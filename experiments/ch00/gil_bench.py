"""The GIL, measured: threads vs processes on Python-bound vs C-bound work.

Four quadrants:
                      | threads          | processes
  pure-Python compute | ~1x (GIL-bound)  | ~min(k, cores)x
  C kernel (matmul)   | scales (GIL      | scales (but pays
                      |  released)       |  process startup + IPC)

Why it matters: this asymmetry is the reason DataLoader workers and torchrun
ranks are *processes*. Distributed training is process-based because arbitrary
Python cannot be parallelized with threads.

Run:  python experiments/ch00/gil_bench.py [--n 2000000] [--workers 4]
CPU-only; no GPU required. Uses the 'spawn' start method (the one CUDA needs).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch


def py_work(n: int) -> int:
    """Pure-Python loop: each iteration is bytecode, executed holding the GIL."""
    s = 0
    for i in range(n):
        s += i * i
    return s


def torch_work(size: int) -> float:
    """C-kernel work: matmul releases the GIL while it runs."""
    a = torch.randn(size, size)
    return float((a @ a).sum())


def run_pool(pool_cls, fn, chunks, ctx=None) -> float:
    kwargs = {"mp_context": ctx} if (ctx is not None and pool_cls is ProcessPoolExecutor) else {}
    t0 = time.perf_counter()
    with pool_cls(max_workers=len(chunks), **kwargs) as ex:
        list(ex.map(fn, chunks))
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000, help="pure-Python loop length")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--matsize", type=int, default=384, help="matmul size for C-kernel test")
    args = ap.parse_args()
    k = args.workers
    ctx = mp.get_context("spawn")  # the start method CUDA requires; be honest about its cost

    # intra-op threads = 1 so OUR thread/process pool is the only parallelism
    torch.set_num_threads(1)

    print(f"cores available: {mp.cpu_count()}, workers: {k}, start method: spawn\n")
    rows = []

    # -- quadrant 1/2: pure Python ------------------------------------------
    t0 = time.perf_counter()
    py_work(args.n)
    serial_py = time.perf_counter() - t0
    chunks = [args.n // k] * k
    t_thr = run_pool(ThreadPoolExecutor, py_work, chunks)
    t_prc = run_pool(ProcessPoolExecutor, py_work, chunks, ctx=ctx)
    rows.append(("pure Python", serial_py, t_thr, t_prc))

    # -- quadrant 3/4: C kernel ----------------------------------------------
    reps = 8 * k
    t0 = time.perf_counter()
    for _ in range(reps):
        torch_work(args.matsize)
    serial_k = time.perf_counter() - t0
    sizes = [args.matsize] * reps

    def thr_kernel() -> float:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as ex:
            list(ex.map(torch_work, sizes))
        return time.perf_counter() - t0

    t_thr_k = thr_kernel()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=k, mp_context=ctx) as ex:
        list(ex.map(torch_work, sizes))
    t_prc_k = time.perf_counter() - t0
    rows.append(("torch matmul", serial_k, t_thr_k, t_prc_k))

    hdr = (f"{'workload':<14} {'serial s':>9} {'threads s':>10} {'thr speedup':>12} "
           f"{'procs s':>9} {'proc speedup':>13}")
    print(hdr)
    print("-" * len(hdr))
    for name, s, t, p in rows:
        print(f"{name:<14} {s:>9.2f} {t:>10.2f} {s / t:>11.2f}x {p:>9.2f} {s / p:>12.2f}x")

    print(
        "\nReading the table:\n"
        "  - pure Python + threads ~1x (or below: GIL handoff overhead).\n"
        "  - pure Python + processes ~min(workers, cores)x, minus spawn startup.\n"
        "  - matmul + threads scales: the kernel releases the GIL.\n"
        "  - process startup with spawn is expensive (fresh interpreter + imports);\n"
        "    that fixed cost is why torchrun starts ranks ONCE, not per step."
    )


if __name__ == "__main__":
    main()
