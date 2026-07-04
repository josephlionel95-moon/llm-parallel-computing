"""Benchmark collectives vs message size; fit the alpha-beta cost model.

    torchrun --nproc_per_node=2 experiments/ch07/benchmark_collectives.py
    python experiments/ch07/benchmark_collectives.py --nproc 2   # spawn mode

Runs on 2x T4 (NCCL), 1 GPU or CPU-only (Gloo). Message sizes sweep
1 KB -> 64 MB to expose the two regimes the chapter derives:
  small messages: T ~ alpha            (latency-dominated, flat on log-log)
  large messages: T ~ N * beta         (bandwidth-dominated, slope 1)
The script least-squares fits alpha and beta from the all_reduce curve and
prints the crossover size N* = alpha/beta where the regimes meet.
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

from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402
from llmdist.workers.ch07 import _bench, fit_alpha_beta  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2, help="spawn mode world size")
    ap.add_argument("--min-kb", type=int, default=1)
    ap.add_argument("--max-mb", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def print_table(rows: list[list[str]], header: list[str]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        tb = Table(*header)
        for r in rows:
            tb.add_row(*r)
        Console().print(tb)
    except ImportError:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for r in rows:
            print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def bench_worker(rank: int, world: int) -> None:
    args = parse_args()
    device = (torch.device(f"cuda:{rank}") if dist.get_backend() == "nccl"
              else torch.device("cpu"))
    sizes = []
    n = args.min_kb * 1024
    while n <= args.max_mb * 2**20:
        sizes.append(n)
        n *= 4

    results: dict[str, dict[int, float]] = {}
    for nbytes in sizes:
        numel = max(4, nbytes // 4)
        t = torch.randn(numel, device=device)
        gathered = [torch.empty_like(t) for _ in range(world)]
        ops = {
            "broadcast": lambda: dist.broadcast(t, src=0),
            "all_reduce": lambda: dist.all_reduce(t),
            "all_gather": lambda: dist.all_gather(gathered, t),
        }
        if dist.get_backend() == "nccl":          # Gloo has no reduce_scatter
            n_rs = (numel // world) * world
            rs_in = list(torch.randn(n_rs, device=device).chunk(world))
            rs_out = torch.empty(n_rs // world, device=device)
            ops["reduce_scatter"] = lambda: dist.reduce_scatter(rs_out, rs_in)
        iters = max(3, args.iters if nbytes < 2**24 else args.iters // 2)
        for name, fn in ops.items():
            results.setdefault(name, {})[nbytes] = _bench(fn, iters=iters)

    if rank != 0:
        return

    rows = []
    for nbytes in sizes:
        # Bus bandwidth a la nccl-tests: algorithm bytes / time. For ring
        # all_reduce the wire factor is 2(p-1)/p (chapter derivation).
        ar_ms = results["all_reduce"][nbytes]
        factor = 2 * (world - 1) / world
        busbw = nbytes * factor / (ar_ms * 1e-3) / 1e9
        rows.append([f"{nbytes / 2**10:,.0f} KB"]
                    + [f"{results[k][nbytes]:.3f}" for k in results]
                    + [f"{busbw:.2f}"])
    print(f"\nbackend={dist.get_backend()}  world={world}  (times in ms, min of runs)")
    print_table(rows, ["size"] + list(results) + ["all_reduce busbw GB/s"])

    xs = sizes
    ys = [results["all_reduce"][s] for s in sizes]
    alpha_us, beta_ns = fit_alpha_beta(xs, ys)
    print("\nalpha-beta fit of all_reduce:  T(N) = alpha + beta*N")
    print(f"  alpha ~ {alpha_us:8.1f} us   (per-message latency)")
    print(f"  beta  ~ {beta_ns:8.3f} ns/B  (~{1 / max(beta_ns, 1e-9):.2f} GB/s effective)")
    print(f"  crossover N* = alpha/beta ~ {alpha_us * 1e3 / max(beta_ns, 1e-9) / 2**10:,.0f} KB")
    print("  below N*: latency-dominated (batch your messages!);"
          " above: bandwidth-dominated")

    if args.save_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, series in results.items():
            ax.plot(list(series), list(series.values()), "o-", label=name)
        fit = [(alpha_us * 1e-3 + beta_ns * 1e-6 * s) for s in sizes]
        ax.plot(sizes, fit, "k--", alpha=0.6, label="alpha+beta*N fit (all_reduce)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("message size (bytes)"); ax.set_ylabel("time (ms)")
        ax.set_title(f"Collectives vs size ({dist.get_backend()}, world={world}): "
                     f"flat = latency regime, slope 1 = bandwidth regime")
        ax.legend(); ax.grid(alpha=0.3)
        out = os.path.join(ROOT, "visualizations", "ch07_collective_bench.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"saved {out}")


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:                      # torchrun path
        rank, world, _ = setup_from_env()
        try:
            bench_worker(rank, world)
        finally:
            cleanup()
    else:                                         # plain python: self-spawn
        run_distributed(bench_worker, world_size=args.nproc, port=29551)


if __name__ == "__main__":
    main()
