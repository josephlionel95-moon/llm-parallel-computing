"""Manual (p2p-only) all_reduce vs torch.distributed: correctness + speed.

    torchrun --nproc_per_node=2 experiments/ch07/manual_vs_builtin_allreduce.py
    python experiments/ch07/manual_vs_builtin_allreduce.py --nproc 2

Verifies our naive and ring all_reduce against dist.all_reduce, then times
all three across sizes. Expected shape of the result (do not trust, measure):
- naive loses to ring at large N (root moves 2(p-1)N bytes vs 2(p-1)N/p per
  rank on the ring) — at p=2 the gap is small, it grows with p;
- both manual versions pay Python/dispatch overhead per step, so the tuned
  library wins overall; the ring's *scaling* with N is what matches theory.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from llmdist.comm import manual_collectives as mc  # noqa: E402
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402
from llmdist.workers.ch07 import _bench  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--max-mb", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()

    # ---- correctness first: never benchmark wrong code -----------------
    g = torch.Generator().manual_seed(rank)
    for numel in (1, 5, 1000, 12_345):            # incl. sizes not divisible by p
        t = torch.randn(numel, generator=g)
        orig = t.clone()
        mc.all_reduce_ring(t)
        mc.assert_matches_torch_allreduce(t, orig)
        t = orig.clone()
        mc.all_reduce_naive(t)
        mc.assert_matches_torch_allreduce(t, orig)
    if rank == 0:
        print("correctness: manual naive + ring == dist.all_reduce  [OK]\n")

    # ---- speed ----------------------------------------------------------
    sizes = []
    n = 1024
    while n <= args.max_mb * 2**20:
        sizes.append(n)
        n *= 4
    results: dict[str, dict[int, float]] = {}
    for nbytes in sizes:
        t = torch.randn(max(4, nbytes // 4))
        for name, fn in (("manual naive", lambda: mc.all_reduce_naive(t)),
                         ("manual ring", lambda: mc.all_reduce_ring(t)),
                         ("dist.all_reduce", lambda: dist.all_reduce(t))):
            results.setdefault(name, {})[nbytes] = _bench(fn, iters=args.iters)
        if rank == 0:
            line = " | ".join(f"{k} {v[nbytes]:8.3f} ms" for k, v in results.items())
            print(f"{nbytes / 2**10:>9,.0f} KB  {line}", flush=True)

    if rank == 0:
        print(f"\nworld={world}, backend={dist.get_backend()} (CPU Gloo unless NCCL)")
        print("manual implementations pay per-step Python + p2p dispatch overhead;")
        print("compare the SLOPES at large N — that is where the algorithm shows.")
        if args.save_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for name, series in results.items():
                ax.plot(list(series), list(series.values()), "o-", label=name)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("message size (bytes)"); ax.set_ylabel("time (ms)")
            ax.set_title(f"all_reduce implementations, world={world}")
            ax.legend(); ax.grid(alpha=0.3)
            out = os.path.join(ROOT, "visualizations", "ch07_manual_vs_builtin.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.tight_layout(); fig.savefig(out, dpi=120)
            print(f"saved {out}")


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        rank, world, _ = setup_from_env()
        try:
            worker(rank, world)
        finally:
            cleanup()
    else:
        run_distributed(worker, world_size=args.nproc, port=29552)


if __name__ == "__main__":
    main()
