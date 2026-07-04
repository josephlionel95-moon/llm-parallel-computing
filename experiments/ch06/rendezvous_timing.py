"""Rendezvous cost vs world size, measured with spawned CPU/Gloo processes.

Times three components for world sizes 2..N:
  spawn      — starting fresh Python interpreters (mp.spawn overhead)
  rendezvous — init_process_group: TCPStore check-in + Gloo connection setup
  first coll — the first all_reduce (lazy per-group communicator setup)
plus the second collective, to show setup cost is paid once and cached.

Expectation (derivation, not benchmark): total ≈ c0 + c1*p — linear in p with
a large constant (interpreter + import cost per process). Init is SECONDS per
JOB, amortized over days of training; per-step costs (ch07's alpha) are what
you optimize.

Run:  python experiments/ch06/rendezvous_timing.py [--max-world 8] [--save-plots]
Works anywhere: CPU-only by design (Gloo).
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llmdist.utils.env_check import detect


def _worker(rank: int, world_size: int, port: int, q) -> None:  # type: ignore[no-untyped-def]
    t_start = time.perf_counter()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=60))
    t_init = time.perf_counter()

    x = torch.ones(1)
    dist.all_reduce(x)
    t_first = time.perf_counter()

    dist.all_reduce(x)
    t_second = time.perf_counter()

    q.put((rank, t_init - t_start, t_first - t_init, t_second - t_first))
    dist.destroy_process_group()


def print_table(rows: list[list[str]], header: list[str]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(*header)
        for r in rows:
            t.add_row(*r)
        Console().print(t)
    except ImportError:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        print(fmt.format(*("-" * w for w in widths)))
        for r in rows:
            print(fmt.format(*r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-world", type=int, default=8)
    ap.add_argument("--save-plots", action="store_true")
    args = ap.parse_args()

    print(detect().banner())
    print("\nbackend: gloo on CPU (deliberately — rendezvous cost, not link speed)\n")

    sizes = [w for w in (2, 4, 8, 16) if w <= args.max_world]
    rows, totals = [], []
    ctx = mp.get_context("spawn")
    for i, ws in enumerate(sizes):
        q = ctx.SimpleQueue()
        t0 = time.perf_counter()
        mp.spawn(_worker, args=(ws, 29600 + i, q), nprocs=ws, join=True)
        total = time.perf_counter() - t0

        per_rank = [q.get() for _ in range(ws)]
        init_max = max(r[1] for r in per_rank)          # slowest rank gates the group
        first_max = max(r[2] for r in per_rank)
        second_max = max(r[3] for r in per_rank)
        spawn_est = total - init_max - first_max - second_max
        rows.append([str(ws), f"{total:.2f}", f"{max(spawn_est, 0):.2f}",
                     f"{init_max:.3f}", f"{first_max * 1e3:.1f}",
                     f"{second_max * 1e3:.3f}"])
        totals.append(total)
        print(f"world {ws}: total {total:.2f} s")

    print()
    print_table(rows, ["world", "total s", "~spawn s", "init s (max rank)",
                       "1st all_reduce ms", "2nd all_reduce ms"])
    print("\nread the last two columns: the FIRST collective pays lazy per-group")
    print("connection/communicator setup; the second is orders of magnitude")
    print("cheaper. This is why every benchmark in this course warms up first.")

    if args.save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sizes, totals, "o-")
        ax.set_xlabel("world size")
        ax.set_ylabel("spawn + rendezvous + teardown (s)")
        ax.set_title("Rendezvous cost vs world size (CPU/Gloo)")
        fig.tight_layout()
        fig.savefig(out / "ch06_rendezvous_timing.png", dpi=120)
        print(f"saved {out / 'ch06_rendezvous_timing.png'}")


if __name__ == "__main__":
    main()
