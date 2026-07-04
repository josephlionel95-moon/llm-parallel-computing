"""ManualDDP: DistributedDataParallel rebuilt from scratch, decision by decision.

The problem with ch08's manual DP: `average_gradients()` runs only after
backward() returns, so the network idles during backward and the GPUs idle
during the all-reduce. But gradients become ready ONE BY ONE, *during*
backward — the gradient of the last layer exists while the first layer's is
still being computed. DDP's whole design follows from exploiting that:

1. HOOKS   — be told the moment each p.grad is finalized (we use
             register_post_accumulate_grad_hook), so communication can start
             mid-backward instead of after it.
2. BUCKETS — one all-reduce per parameter would pay the per-message latency
             alpha hundreds of times (ch07's alpha-beta model: small messages
             are latency-dominated). So group parameters into ~25 MB buckets
             and all-reduce a bucket only when ALL its grads are ready.
3. REVERSE ORDER — backward computes gradients roughly from the last layer
             back to the first, i.e. in reverse order of model.parameters().
             Buckets built over reversed parameters therefore FILL earliest,
             and the first all-reduce launches as early as possible.
4. ASYNC   — all_reduce(async_op=True) returns a handle immediately; the
             backward pass keeps computing while NCCL/Gloo moves bytes. We
             wait on all handles in finalize(), just before optimizer.step().
5. no_sync — under gradient accumulation, only the LAST micro-batch needs a
             sync; the context manager makes the hooks do nothing so the
             earlier micro-batches just accumulate into p.grad locally.

Usage:

    model = ManualDDP(MiniGPT(cfg), bucket_size_mb=25)
    for x, y in loader:
        loss = model(x, y)[1]
        loss.backward()          # hooks fire, buckets fly, all during this line
        model.finalize()         # wait, average, write back into p.grad
        opt.step(); opt.zero_grad()

Differences from the real torch DDP (dissected in ch09): no autograd-graph
traversal to find unused parameters, no first-iteration bucket rebuilding by
observed ready order, no gradient_as_bucket_view memory aliasing, and
finalize() is explicit rather than hidden in a backward callback. Each
omission is a design decision the chapter examines.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class _Bucket:
    """A group of parameters whose gradients share one all-reduce."""
    index: int
    params: list[nn.Parameter]
    offsets: list[int]                      # start offset of each param in buffer
    buffer: torch.Tensor                    # flat, len = sum of param numels
    pending: int = 0                        # grads still not ready this iteration
    work: Optional[dist.Work] = None        # async all-reduce handle
    t_launch: float = 0.0                   # perf_counter at launch (for timelines)
    t_done: float = 0.0                     # perf_counter after wait()

    def size_bytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()


class ManualDDP(nn.Module):
    """A from-scratch DistributedDataParallel. See module docstring for the
    five design decisions; see ch09 for how the real one differs.

    Args:
        module: the model to replicate. Must be on its final device already.
        bucket_size_mb: target bucket size. Small -> more messages (alpha
            pain); large -> the last bucket finishes late and overlap shrinks
            (the U-curve measured in experiments/ch09/bucket_size_sweep.py).
        broadcast_buffers_from: rank whose params/buffers define the replica
            (None to skip the broadcast, e.g. when the caller already did it).
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0,
                 broadcast_buffers_from: Optional[int] = 0):
        super().__init__()
        assert dist.is_initialized(), "init_process_group before ManualDDP"
        self.module = module
        self.world_size = dist.get_world_size()
        self.require_backward_grad_sync = True   # toggled by no_sync()
        self.bucket_events: list[tuple[int, float, float]] = []  # (idx, t_launch, t_done)

        # (1) Replicas must start identical: broadcast params AND buffers.
        if broadcast_buffers_from is not None:
            for p in module.parameters():
                dist.broadcast(p.detach(), src=broadcast_buffers_from)
            for b in module.buffers():
                dist.broadcast(b.detach(), src=broadcast_buffers_from)

        # (3) Build buckets over parameters in REVERSE registration order.
        # Autograd produces grads roughly last-layer-first, so reverse-order
        # buckets are the ones that complete earliest during backward.
        self.buckets: list[_Bucket] = []
        # param id -> (bucket, slot). Keyed by id() because tensors overload
        # __eq__ elementwise, which breaks dict/list lookups by value.
        self._param_slot: dict[int, tuple[_Bucket, int]] = {}
        cap = int(bucket_size_mb * 2**20)
        params = [p for p in module.parameters() if p.requires_grad]
        group: list[nn.Parameter] = []
        group_bytes = 0
        for p in reversed(params):
            group.append(p)
            group_bytes += p.numel() * p.element_size()
            if group_bytes >= cap:
                self._seal_bucket(group)
                group, group_bytes = [], 0
        if group:
            self._seal_bucket(group)

        # (2) One hook per parameter, fired the moment its grad is finalized
        # (post-accumulate: after grad is summed into p.grad, so gradient
        # accumulation composes correctly with these hooks).
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

        self._reset_state()

    # ------------------------------------------------------------------ setup

    def _seal_bucket(self, group: list[nn.Parameter]) -> None:
        offsets, total = [], 0
        for p in group:
            offsets.append(total)
            total += p.numel()
        ref = group[0]
        bucket = _Bucket(index=len(self.buckets), params=list(group),
                         offsets=offsets,
                         buffer=torch.zeros(total, dtype=ref.dtype, device=ref.device))
        for i, p in enumerate(group):
            self._param_slot[id(p)] = (bucket, i)
        self.buckets.append(bucket)

    def _reset_state(self) -> None:
        for b in self.buckets:
            b.pending = len(b.params)
            b.work = None

    # ------------------------------------------------------------- hot path

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _hook(self, param: nn.Parameter) -> None:
        """Runs inside backward(), once per parameter, when p.grad is final.

        Copies the grad into its bucket slot; when the bucket's last grad
        lands, launches the bucket's all-reduce WITHOUT waiting (4). The
        autograd engine immediately resumes computing earlier layers' grads,
        so bytes move while FLOPs happen — that overlap is DDP's entire
        performance story.
        """
        if not self.require_backward_grad_sync:
            return                            # no_sync(): accumulate locally only
        bucket, i = self._param_slot[id(param)]
        off = bucket.offsets[i]
        bucket.buffer[off:off + param.numel()].copy_(param.grad.reshape(-1))
        bucket.pending -= 1
        if bucket.pending == 0:
            bucket.t_launch = time.perf_counter()
            bucket.work = dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM,
                                          async_op=True)

    def finalize(self) -> None:
        """Call after backward(), before optimizer.step().

        (5) Waits for every in-flight all-reduce, averages, and writes the
        synchronized gradients back into p.grad. Waiting here — not inside
        the hook — is the point: by now most transfers finished during
        backward, so these waits are usually near-free.

        The real DDP hides this in an autograd callback so users never call
        it; we keep it explicit so you can time it (experiments/ch09).
        """
        if not self.require_backward_grad_sync:
            return
        for b in self.buckets:
            if b.work is None:
                stragglers = [i for i, bb in enumerate(self.buckets) if bb.pending > 0]
                raise RuntimeError(
                    f"buckets {stragglers} never launched: some parameters got no "
                    f"gradient this iteration. Real DDP handles this with "
                    f"find_unused_parameters=True (at the cost of traversing the "
                    f"autograd graph every step) — see ch09.")
            b.work.wait()
            b.t_done = time.perf_counter()
            b.buffer /= self.world_size       # SUM -> mean over replicas
            for p, off in zip(b.params, b.offsets):
                p.grad.copy_(b.buffer[off:off + p.numel()].view_as(p))
        self.bucket_events = [(b.index, b.t_launch, b.t_done) for b in self.buckets]
        self._reset_state()

    # ----------------------------------------------------- gradient accumulation

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        """Disable gradient synchronization inside the context.

        for i, micro in enumerate(microbatches):
            ctx = model.no_sync() if i < len(microbatches)-1 else nullcontext()
            with ctx:
                model(micro).loss.backward()   # grads accumulate locally
        model.finalize()                        # only the last backward synced

        Without this, k-step gradient accumulation would pay k all-reduces
        for one optimizer step — k-1 of them pure waste, since all-reduce is
        linear and commutes with local accumulation (ch08 math).
        """
        prev = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = prev
