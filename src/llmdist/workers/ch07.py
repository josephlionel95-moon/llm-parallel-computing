"""Spawn workers for the ch07 notebook (collective communication).

Each worker runs under llmdist.utils.dist.run_distributed (Gloo/CPU by
default) and prints per-rank state or saves measurements from rank 0.
"""
from __future__ import annotations

import time
from typing import Callable

import torch
import torch.distributed as dist

from llmdist.comm import manual_collectives as mc


def ordered_print(rank: int, world: int, *lines: str) -> None:
    """Print rank by rank so multi-process output does not interleave."""
    for r in range(world):
        dist.barrier()
        if r == rank:
            for ln in lines:
                print(ln, flush=True)
    dist.barrier()


def _bench(fn: Callable[[], None], iters: int = 10, warmup: int = 3) -> float:
    """Min wall time (ms) of fn(), barrier-aligned so all ranks co-start.

    min, not mean: collective time is gated by the slowest rank anyway, and
    min filters out OS scheduling noise on shared CPU runners.
    """
    sync_cuda = dist.get_backend() == "nccl"      # NCCL ops are async on stream
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


# --------------------------------------------------------------------- demos

def w_collectives_tour(rank: int, world: int) -> None:
    """Every manual collective on a 4-element tensor, before/after per rank."""
    base = torch.arange(4.0) + 10 * rank          # rank 0: [0..3], rank 1: [10..13]

    t = base.clone()
    mc.broadcast(t, src=0)
    ordered_print(rank, world,
                  f"[broadcast   ] rank {rank}: {base.tolist()} -> {t.tolist()}")

    t = base.clone()
    mc.reduce(t, dst=0)
    ordered_print(rank, world,
                  f"[reduce->0   ] rank {rank}: {base.tolist()} -> {t.tolist()}"
                  + ("   (sum lands here)" if rank == 0 else "   (unchanged)"))

    t = base.clone()
    mc.all_reduce_ring(t)
    ordered_print(rank, world,
                  f"[all_reduce  ] rank {rank}: {base.tolist()} -> {t.tolist()}")

    chunks = [torch.full((4,), float(i)) for i in range(world)] if rank == 0 else None
    out = torch.empty(4)
    mc.scatter(out, chunks, src=0)
    ordered_print(rank, world, f"[scatter     ] rank {rank}: got {out.tolist()}")

    got = mc.gather(base.clone(), dst=0)
    ordered_print(rank, world,
                  f"[gather->0   ] rank {rank}: "
                  + (f"{[g.tolist() for g in got]}" if got else "(nothing, not dst)"))

    out = torch.empty(world, 4)
    mc.all_gather(out, base.clone())
    ordered_print(rank, world, f"[all_gather  ] rank {rank}: {out.tolist()}")

    inp = torch.stack([torch.arange(4.0) + rank + 100 * i for i in range(world)])
    out = torch.empty(4)
    mc.reduce_scatter(out, inp)
    ordered_print(rank, world,
                  f"[reduce_scat ] rank {rank}: rows {inp.tolist()} -> mine {out.tolist()}")


def w_verify_manual(rank: int, world: int, numel: int = 100_000) -> None:
    """Assert manual naive + ring all_reduce match torch, on random data."""
    g = torch.Generator().manual_seed(rank)
    for name, fn in (("naive", mc.all_reduce_naive), ("ring", mc.all_reduce_ring)):
        t = torch.randn(numel, generator=g)
        orig = t.clone()
        fn(t)
        mc.assert_matches_torch_allreduce(t, orig)
        mc.assert_all_ranks_equal(t, atol=1e-5)
    if rank == 0:
        print(f"manual naive + ring all_reduce == torch.dist.all_reduce "
              f"on {numel} floats across {world} ranks  [OK]", flush=True)


# ---------------------------------------------------------------- benchmarks

def w_collective_family_bench(rank: int, world: int, sizes_bytes: tuple,
                              iters: int, out_path: str) -> None:
    """Time torch's broadcast / all_reduce / all_gather / reduce_scatter vs
    message size. Rank 0 saves {collective: {size_bytes: ms}}."""
    device = (torch.device(f"cuda:{rank}") if dist.get_backend() == "nccl"
              else torch.device("cpu"))
    results: dict[str, dict[int, float]] = {}
    for nbytes in sizes_bytes:
        n = max(4, nbytes // 4)                   # fp32 elements
        t = torch.randn(n, device=device)
        gathered = [torch.empty_like(t) for _ in range(world)]
        ops: dict[str, Callable[[], None]] = {
            "broadcast": lambda: dist.broadcast(t, src=0),
            "all_reduce": lambda: dist.all_reduce(t),
            "all_gather": lambda: dist.all_gather(gathered, t),
        }
        if dist.get_backend() == "nccl":
            # Gloo has NO reduce_scatter — a fact worth noticing: our manual
            # ring reduce_scatter (p2p only) runs on Gloo just fine.
            n_rs = (n // world) * world           # needs divisibility
            rs_in = list(torch.randn(n_rs, device=device).chunk(world))
            rs_out = torch.empty(n_rs // world, device=device)
            ops["reduce_scatter"] = lambda: dist.reduce_scatter(rs_out, rs_in)
        for name, fn in ops.items():
            ms = _bench(fn, iters=iters)
            results.setdefault(name, {})[nbytes] = ms
            if rank == 0:
                print(f"{name:<15} {nbytes / 2**10:>10.0f} KB  {ms:8.3f} ms", flush=True)
    if rank == 0:
        torch.save({"world": world, "results": results}, out_path)


def w_manual_vs_torch_bench(rank: int, world: int, sizes_bytes: tuple,
                            iters: int, out_path: str) -> None:
    """manual naive vs manual ring vs dist.all_reduce across sizes."""
    results: dict[str, dict[int, float]] = {}
    for nbytes in sizes_bytes:
        t = torch.randn(max(4, nbytes // 4))
        impls: dict[str, Callable[[], None]] = {
            "manual naive": lambda: mc.all_reduce_naive(t),
            "manual ring": lambda: mc.all_reduce_ring(t),
            "torch dist.all_reduce": lambda: dist.all_reduce(t),
        }
        for name, fn in impls.items():
            ms = _bench(fn, iters=iters)
            results.setdefault(name, {})[nbytes] = ms
            if rank == 0:
                print(f"{name:<22} {nbytes / 2**10:>8.0f} KB  {ms:8.3f} ms", flush=True)
    if rank == 0:
        torch.save({"world": world, "results": results}, out_path)


def fit_alpha_beta(sizes_bytes: list[int], times_ms: list[float]) -> tuple[float, float]:
    """Least-squares fit of T(n) = alpha + beta*n to measured points.

    Returns (alpha_us, beta_ns_per_byte) — i.e. latency in microseconds and
    inverse bandwidth in ns/byte (1 ns/byte = 1 GB/s). Called by notebook and
    experiment after benchmarking; pure NumPy-free implementation.
    """
    n = len(sizes_bytes)
    xs = [float(s) for s in sizes_bytes]
    ys = [t * 1e-3 for t in times_ms]              # seconds
    mx, my = sum(xs) / n, sum(ys) / n
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / \
        max(sum((x - mx) ** 2 for x in xs), 1e-30)
    alpha = my - beta * mx
    return alpha * 1e6, beta * 1e9
