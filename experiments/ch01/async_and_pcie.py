"""Async execution, kernel-launch overhead, and PCIe bandwidth (pageable vs pinned).

Three demos:
  1. The time.time() lie: unsynchronized timing of an async launch vs the truth
     (torch.cuda.Event timing, as wrapped by llmdist.profiling.timers.timeit).
  2. Kernel launch overhead: microseconds per launch, measured over 1000 tiny ops.
  3. H2D/D2H bandwidth: pageable vs pinned host memory, several sizes.

Why it matters downstream: DDP overlaps NCCL comms with backward compute using
exactly the async-stream machinery demo 1 exposes; and on Kaggle's 2x T4 the
GPUs communicate over the same PCIe bus demo 3 measures.

Run:  python experiments/ch01/async_and_pcie.py [--mb 256]
CPU-only machines get explanatory skip messages (nothing here exists on CPU).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.profiling.timers import bandwidth_gbps, timeit
from llmdist.utils import env_check


def demo_async_lie(device: torch.device) -> None:
    print("=" * 70)
    print("Demo 1: the time.time() lie")
    a = torch.randn(4096, 4096, device=device)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    _ = a @ a                                # returns after ENQUEUE
    lie_ms = (time.perf_counter() - t0) * 1e3
    torch.cuda.synchronize()                 # drain before honest measurement

    truth = timeit(lambda: a @ a, iters=10, warmup=3, device=device)
    print(f"  unsynchronized wall clock : {lie_ms:9.4f} ms   (launch only)")
    print(f"  CUDA-event mean           : {truth['mean_ms']:9.4f} ms   (the kernel)")
    print(f"  -> the naive number was {truth['mean_ms'] / max(lie_ms, 1e-9):.0f}x too small")


def demo_launch_overhead(device: torch.device) -> None:
    print("=" * 70)
    print("Demo 2: kernel launch overhead")
    tiny = torch.randn(8, device=device)
    n = 1000
    for _ in range(50):
        tiny.add_(1.0)                       # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        tiny.add_(1.0)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"  {n} tiny in-place adds -> {us:6.2f} us/launch")
    print(f"  a step with 2000 kernels pays >= {2000 * us / 1e3:5.1f} ms of launch cost;")
    print("  this is what torch.compile fusion and CUDA Graphs eliminate.")


def demo_pcie(device: torch.device, mb: int) -> None:
    print("=" * 70)
    print(f"Demo 3: PCIe H2D/D2H bandwidth, {mb} MiB payload")
    n = mb * 2**20 // 4
    pageable = torch.randn(n)
    pinned = torch.randn(n).pin_memory()     # allocate ONCE, outside timing
    dev_buf = torch.empty(n, device=device)

    rows = [
        ("H2D pageable", lambda: dev_buf.copy_(pageable)),
        ("H2D pinned  ", lambda: dev_buf.copy_(pinned, non_blocking=True)),
        ("D2H pageable", lambda: pageable.copy_(dev_buf)),
        ("D2H pinned  ", lambda: pinned.copy_(dev_buf, non_blocking=True)),
    ]
    print(f"  {'direction':<14} {'ms':>9} {'GB/s':>8}")
    for name, fn in rows:
        st = timeit(fn, iters=10, warmup=3, device=device)
        print(f"  {name:<14} {st['mean_ms']:>9.3f} {bandwidth_gbps(n * 4, st['mean_ms']):>8.2f}")
    print("  Theoretical PCIe 3.0 x16 ~ 15.75 GB/s; pinned should approach 10-13,")
    print("  pageable sits well below (extra staging memcpy). HBM itself: 320 GB/s")
    print("  on T4 -> host traffic is ~25-100x slower than device memory.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256, help="PCIe test payload in MiB")
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    if env.num_gpus == 0:
        print("[skip] This experiment measures CUDA-specific behavior (async launch,")
        print("       launch overhead, PCIe DMA) and has no CPU equivalent.")
        print("       Expected on a T4: launch ~3-10 us; pageable H2D ~3-6 GB/s;")
        print("       pinned H2D ~10-13 GB/s; unsynchronized timings ~1000x too small.")
        return

    device = torch.device("cuda:0")
    demo_async_lie(device)
    demo_launch_overhead(device)
    demo_pcie(device, args.mb)


if __name__ == "__main__":
    main()
