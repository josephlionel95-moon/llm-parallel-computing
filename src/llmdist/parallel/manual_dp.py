"""Manual data parallelism: the entire idea in three functions.

Data parallelism = every rank holds a FULL model replica, sees a DIFFERENT
shard of each batch, and after backward the ranks AVERAGE their gradients so
every replica takes the identical optimizer step. Chapter 08 derives why
averaging is exactly right (linearity of the gradient of a mean) and why the
replicas therefore never drift.

The training loop shape:

    broadcast_parameters(model)              # once — replicas start identical
    for step, (x, y) in enumerate(loader):   # loader shards by rank
        loss = model(x, y)[1]
        loss.backward()                      # local grads on the local shard
        average_gradients(model)             # <-- the only new line vs 1 GPU
        opt.step(); opt.zero_grad()

This is "naive" DP: the all-reduce starts only AFTER backward finishes, so
communication and computation are strictly serialized. Chapter 09 fixes that
with hooks + buckets (overlap); this module stays deliberately simple so the
semantics are unmistakable.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def broadcast_parameters(model: nn.Module, src: int = 0) -> None:
    """Make every replica identical by broadcasting rank `src`'s state.

    Why needed: each process constructs the model independently, and unless
    every rank seeds RNG identically (fragile — a single extra rand() call on
    one rank desynchronizes everything), the random inits differ. One
    broadcast at startup removes the assumption entirely.

    Buffers too: BatchNorm running stats, our causal mask, etc. — anything
    forward() reads must match, or replicas compute different functions.

    Cost: one broadcast of all parameters, paid once per training run.
    """
    for p in model.parameters():
        dist.broadcast(p.detach(), src=src)
    for b in model.buffers():
        dist.broadcast(b.detach(), src=src)


def average_gradients(model: nn.Module, flat: bool = True) -> None:
    """All-reduce every gradient and divide by world size. Call between
    loss.backward() and optimizer.step().

    After this, every rank's p.grad equals the gradient of the loss averaged
    over the GLOBAL batch — so the update is mathematically identical to a
    single-process step on the concatenated batch (ch08 derivation).

    flat=True packs all grads into ONE buffer and does ONE all-reduce:
        cost ~ 2(p-1)*alpha/step  vs  n_tensors * 2(p-1)*alpha for per-tensor.
    With ~50 parameter tensors and Gloo's per-message latency, flattening is
    an easy multiple-x win on small models — the alpha-beta model (ch07) in
    action. The price: one extra copy of the gradients (transient memory),
    and no overlap with backward either way (that is ch09's job).
    """
    world = dist.get_world_size()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    if flat:
        buf = torch.cat([g.reshape(-1) for g in grads])      # 1 message
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf /= world
        offset = 0
        for g in grads:
            g.copy_(buf[offset:offset + g.numel()].view_as(g))
            offset += g.numel()
    else:
        for g in grads:                                       # n messages
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= world


def shard_indices(n: int, rank: int, world_size: int, epoch: int = 0,
                  shuffle: bool = True, seed: int = 0) -> list[int]:
    """Hand-rolled DistributedSampler: which sample indices does rank r see?

    Contract (same as torch's DistributedSampler):
    1. Every index appears on exactly one rank per epoch (no duplicated or
       skipped gradient contributions — duplicates would silently re-weight
       those samples in the averaged gradient).
    2. All ranks produce the SAME permutation (seeded by `seed + epoch`, not
       by rank!) and then take disjoint slices. If ranks shuffled
       independently, a sample could land on two ranks in one epoch.
    3. Pad by recycling the head so every rank gets ceil(n/world) samples:
       equal shard sizes mean equal numbers of steps, and equal steps mean
       nobody blocks forever in an all-reduce the others never reach.

    Why epoch is an argument (torch: sampler.set_epoch(epoch)): if you forget
    it, the permutation is identical every epoch — each rank revisits the
    SAME shard in the SAME order, quietly hurting generalization. The classic
    silent bug of distributed training.
    """
    if shuffle:
        g = torch.Generator().manual_seed(seed + epoch)   # identical on all ranks
        perm = torch.randperm(n, generator=g).tolist()
    else:
        perm = list(range(n))
    per_rank = (n + world_size - 1) // world_size         # ceil
    perm += perm[: per_rank * world_size - n]             # pad from the head
    return perm[rank::world_size]                          # disjoint strided slices
