"""Timing utilities that are correct in the presence of asynchronous CUDA.

The #1 profiling mistake (ch01/ch04): timing a kernel launch with time.time().
Kernel launches return immediately; without synchronization you measure the
launch overhead (~10 us), not the kernel. We use CUDA events on GPU and
perf_counter on CPU, and always warm up before measuring.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import torch


def timeit(fn: Callable[[], object], iters: int = 20, warmup: int = 5,
           device: torch.device | None = None) -> dict[str, float]:
    """Time fn() and return {'mean_ms', 'std_ms', 'min_ms'}.

    Uses CUDA events when a CUDA device is involved, else perf_counter.
    """
    use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize(device)
    times_ms: list[float] = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms": statistics.fmean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "min_ms": min(times_ms),
    }


def bandwidth_gbps(nbytes: int, ms: float, factor: float = 1.0) -> float:
    """Convert bytes moved in `ms` milliseconds to GB/s.

    `factor` encodes algorithm bytes: e.g. ring AllReduce moves
    2*(p-1)/p * N bytes per rank, so pass factor=2*(p-1)/p to report
    'bus bandwidth' the way nccl-tests does.
    """
    return (nbytes * factor) / (ms * 1e-3) / 1e9


class Throughput:
    """Tokens/sec and samples/sec meter for training loops."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._t0 = time.perf_counter()
        self.tokens = 0
        self.samples = 0

    def update(self, tokens: int, samples: int) -> None:
        self.tokens += tokens
        self.samples += samples

    def report(self) -> dict[str, float]:
        dt = time.perf_counter() - self._t0
        return {"tokens_per_s": self.tokens / dt,
                "samples_per_s": self.samples / dt,
                "elapsed_s": dt}
