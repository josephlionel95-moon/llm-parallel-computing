"""Spawn workers for the ch09 notebook (DDP internals)."""
from __future__ import annotations

import copy
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.parallel.manual_ddp import ManualDDP
from llmdist.parallel.manual_dp import average_gradients, broadcast_parameters
# Slightly bigger than ch08's model so per-bucket communication is visible
# next to backward compute on CPU (~3.3M params -> ~13 MB of fp32 grads).
NB_CFG = GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, d_model=256)


def _batch(rank: int, step: int, bsz: int = 8,
           cfg: GPTConfig = NB_CFG) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(step * 1000 + rank)
    x = torch.randint(0, cfg.vocab_size, (bsz, cfg.block_size), generator=g)
    return x, x.roll(-1, dims=1)


def w_compare_dp_ddp(rank: int, world: int, steps: int, bucket_mb: float,
                     out_path: str) -> None:
    """Step-time comparison: naive manual DP vs ManualDDP vs torch DDP,
    identical model/data/optimizer. Rank 0 saves mean ms/step of each,
    plus a correctness check: ManualDDP grads == torch DDP grads.
    """
    torch.manual_seed(0)
    base = MiniGPT(NB_CFG)

    m_dp = copy.deepcopy(base)
    broadcast_parameters(m_dp)
    m_manual = ManualDDP(copy.deepcopy(base), bucket_size_mb=bucket_mb)
    m_torch = nn.parallel.DistributedDataParallel(
        copy.deepcopy(base), bucket_cap_mb=bucket_mb)

    def dp_step(step: int) -> None:
        x, y = _batch(rank, step)
        m_dp(x, y)[1].backward()
        average_gradients(m_dp)                   # serialized comm
        for p in m_dp.parameters():
            p.grad = None

    def manual_ddp_step(step: int) -> None:
        x, y = _batch(rank, step)
        m_manual(x, y)[1].backward()              # comm overlaps in hooks
        m_manual.finalize()
        for p in m_manual.parameters():
            p.grad = None

    def torch_ddp_step(step: int) -> None:
        x, y = _batch(rank, step)
        m_torch(x, y)[1].backward()               # comm overlaps via Reducer
        for p in m_torch.parameters():
            p.grad = None

    results: dict[str, float] = {}
    for name, fn in (("manual DP (naive)", dp_step),
                     ("ManualDDP", manual_ddp_step),
                     ("torch DDP", torch_ddp_step)):
        fn(0); fn(0)                              # warmup (DDP builds buckets)
        dist.barrier()
        t0 = time.perf_counter()
        for s in range(steps):
            fn(s + 1)
        dist.barrier()
        results[name] = (time.perf_counter() - t0) * 1e3 / steps
        if rank == 0:
            print(f"{name:<18} {results[name]:8.1f} ms/step", flush=True)

    # Correctness: same fresh batch through ManualDDP and torch DDP.
    x, y = _batch(rank, step=999)
    m_manual(x, y)[1].backward(); m_manual.finalize()
    m_torch(x, y)[1].backward()
    worst = 0.0
    for pm, pt in zip(m_manual.module.parameters(), m_torch.module.parameters()):
        worst = max(worst, (pm.grad - pt.grad).abs().max().item())
        assert torch.allclose(pm.grad, pt.grad, atol=1e-6, rtol=1e-5)
    if rank == 0:
        print(f"ManualDDP grads == torch DDP grads (max diff {worst:.2e})  [OK]")
        torch.save({"step_ms": results, "grad_max_diff": worst,
                    "world": world, "bucket_mb": bucket_mb}, out_path)


def w_bucket_sweep(rank: int, world: int, bucket_sizes_mb: tuple,
                   steps: int, out_path: str) -> None:
    """ManualDDP step time as a function of bucket size — the U-curve.

    Tiny buckets: many all-reduces, each paying latency alpha.
    Huge buckets: one all-reduce that can only start when backward is nearly
    done — no overlap left. The optimum sits in between (torch defaults 25MB).
    """
    torch.manual_seed(0)
    base = MiniGPT(NB_CFG)
    results: dict[float, float] = {}
    for mb in bucket_sizes_mb:
        model = ManualDDP(copy.deepcopy(base), bucket_size_mb=mb)

        def step_fn(s: int) -> None:
            x, y = _batch(rank, s)
            model(x, y)[1].backward()
            model.finalize()
            for p in model.parameters():
                p.grad = None

        step_fn(0)                                # warmup
        dist.barrier()
        t0 = time.perf_counter()
        for s in range(steps):
            step_fn(s + 1)
        dist.barrier()
        results[mb] = (time.perf_counter() - t0) * 1e3 / steps
        if rank == 0:
            print(f"bucket {mb:>7.2f} MB  ({len(model.buckets):>3} buckets)  "
                  f"{results[mb]:8.1f} ms/step", flush=True)
    if rank == 0:
        torch.save({"step_ms": results, "world": world}, out_path)


def w_overlap_timeline(rank: int, world: int, out_path: str,
                       bucket_mb: float = 1.0) -> None:
    """Measure real timestamps for the naive-DP vs DDP overlap picture.

    Naive DP:  [backward----------][comm----]      (serialized)
    ManualDDP: [backward----------]                 (comm rides inside)
                  [b3][b2][b1][b0]

    Rank 0 saves events for llmdist.viz.plots.timeline: (lane, kind, t0, t1)
    with a common origin at backward start, in ms. Bucket 'done' times are
    when finalize()'s wait() returned — an upper bound on transfer end (we
    cannot see inside the transport), which still demonstrates that LAUNCHES
    happen mid-backward: that is the overlap.
    """
    torch.manual_seed(0)
    base = MiniGPT(NB_CFG)
    x, y = _batch(rank, step=1, bsz=16)

    # --- naive DP -----------------------------------------------------------
    m_dp = copy.deepcopy(base)
    broadcast_parameters(m_dp)
    m_dp(x, y)[1].backward()                      # warmup autograd allocs
    for p in m_dp.parameters():
        p.grad = None
    dist.barrier()
    t0 = time.perf_counter()
    m_dp(x, y)[1].backward()
    t_bwd_end = time.perf_counter()
    average_gradients(m_dp)
    t_comm_end = time.perf_counter()

    def rel(t: float) -> float:
        return (t - t0) * 1e3

    naive_events = [("naive DP", "compute", 0.0, rel(t_bwd_end)),
                    ("naive DP", "comm", rel(t_bwd_end), rel(t_comm_end))]

    # --- ManualDDP ----------------------------------------------------------
    m_ddp = ManualDDP(copy.deepcopy(base), bucket_size_mb=bucket_mb)
    m_ddp(x, y)[1].backward(); m_ddp.finalize()   # warmup
    for p in m_ddp.parameters():
        p.grad = None
    dist.barrier()
    d0 = time.perf_counter()
    m_ddp(x, y)[1].backward()
    d_bwd_end = time.perf_counter()
    m_ddp.finalize()
    d_fin_end = time.perf_counter()

    def drel(t: float) -> float:
        return (t - d0) * 1e3

    ddp_events = [("ManualDDP", "compute", 0.0, drel(d_bwd_end)),
                  ("ManualDDP", "sync", drel(d_bwd_end), drel(d_fin_end))]
    for idx, t_launch, t_done in m_ddp.bucket_events:
        ddp_events.append((f"  bucket {idx}", "comm", drel(t_launch), drel(t_done)))

    if rank == 0:
        naive_total = rel(t_comm_end)
        ddp_total = drel(d_fin_end)
        print(f"naive DP : backward {rel(t_bwd_end):7.1f} ms + comm "
              f"{rel(t_comm_end) - rel(t_bwd_end):7.1f} ms = {naive_total:7.1f} ms")
        print(f"ManualDDP: backward {drel(d_bwd_end):7.1f} ms, finalize wait "
              f"{drel(d_fin_end) - drel(d_bwd_end):7.1f} ms  = {ddp_total:7.1f} ms")
        print(f"{len(m_ddp.bucket_events)} buckets launched during backward")
        torch.save({"naive": naive_events, "ddp": ddp_events,
                    "naive_total_ms": naive_total, "ddp_total_ms": ddp_total},
                   out_path)
