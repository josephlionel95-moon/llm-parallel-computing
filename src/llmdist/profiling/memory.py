"""Memory measurement utilities.

Key distinction taught in ch02/ch04:
  memory_allocated  = bytes in live tensors
  memory_reserved   = bytes the caching allocator holds from CUDA (>= allocated)
  max_memory_*      = high-water marks since last reset

The gap between allocated and reserved is the CUDA cache; nvidia-smi shows
reserved + context, which is why it always looks scarier than PyTorch's numbers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


def mb(x: int | float) -> float:
    return x / 2**20


@dataclass
class MemSnapshot:
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float

    def __sub__(self, other: "MemSnapshot") -> "MemSnapshot":
        return MemSnapshot(self.allocated_mb - other.allocated_mb,
                           self.reserved_mb - other.reserved_mb,
                           self.max_allocated_mb - other.max_allocated_mb)

    def __str__(self) -> str:
        return (f"allocated {self.allocated_mb:9.1f} MB | reserved {self.reserved_mb:9.1f} MB"
                f" | peak {self.max_allocated_mb:9.1f} MB")


def snapshot(device: torch.device | int | None = None) -> MemSnapshot:
    if not torch.cuda.is_available():
        return MemSnapshot(0.0, 0.0, 0.0)
    return MemSnapshot(mb(torch.cuda.memory_allocated(device)),
                       mb(torch.cuda.memory_reserved(device)),
                       mb(torch.cuda.max_memory_allocated(device)))


@contextmanager
def track(label: str = "", device: torch.device | int | None = None, verbose: bool = True):
    """Measure the memory delta and peak of a code region.

    with track("forward"):
        out = model(x)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before = snapshot(device)
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    after = snapshot(device)
    if verbose:
        d = after - before
        print(f"[{label:<24}] Δalloc {d.allocated_mb:+9.1f} MB | "
              f"peak during region {after.max_allocated_mb:9.1f} MB")


def model_memory_ledger(model: torch.nn.Module, optimizer_bytes_per_param: int = 8,
                        dtype_bytes: int = 4) -> dict[str, float]:
    """Predicted steady-state training memory (MB), the ledger derived in ch04:
    params + grads (dtype_bytes each) + optimizer states (8 B/param for Adam FP32
    m and v; more with master weights)."""
    n = sum(p.numel() for p in model.parameters())
    return {
        "params_mb": mb(n * dtype_bytes),
        "grads_mb": mb(n * dtype_bytes),
        "optimizer_mb": mb(n * optimizer_bytes_per_param),
        "total_static_mb": mb(n * (2 * dtype_bytes + optimizer_bytes_per_param)),
        "num_params": n,
    }
