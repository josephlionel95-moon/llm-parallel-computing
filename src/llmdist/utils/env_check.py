"""Hardware/environment detection used by every notebook and experiment.

Why this exists: the course must run on Kaggle (2x T4), Colab (1x T4), local
machines, and CPU-only CI. Rather than sprinkling `if torch.cuda.is_available()`
everywhere, every entry point calls `detect()` once and branches on the result.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field


@dataclass
class Env:
    """Snapshot of the execution environment relevant to distributed training."""

    platform_name: str          # "kaggle" | "colab" | "local"
    num_gpus: int
    gpu_names: list[str] = field(default_factory=list)
    gpu_mem_gb: list[float] = field(default_factory=list)
    torch_version: str = ""
    cuda_version: str | None = None
    nccl_available: bool = False

    @property
    def multi_gpu(self) -> bool:
        return self.num_gpus >= 2

    @property
    def backend(self) -> str:
        """Preferred torch.distributed backend for this machine.

        NCCL only works between CUDA devices; Gloo works everywhere (CPU tensors),
        which is why our in-notebook simulations always use Gloo.
        """
        return "nccl" if self.num_gpus >= 1 and self.nccl_available else "gloo"

    def banner(self) -> str:
        lines = [
            f"platform      : {self.platform_name}",
            f"python        : {platform.python_version()}",
            f"torch         : {self.torch_version} (CUDA {self.cuda_version})",
            f"gpus          : {self.num_gpus}",
        ]
        for i, (n, m) in enumerate(zip(self.gpu_names, self.gpu_mem_gb)):
            lines.append(f"  [{i}] {n} ({m:.1f} GB)")
        lines.append(f"backend       : {self.backend}")
        if self.num_gpus == 0:
            lines.append("mode          : CPU simulation (Gloo) - semantics identical, no NCCL speed")
        elif self.num_gpus == 1:
            lines.append("mode          : single GPU - multi-GPU cells fall back to CPU/Gloo simulation")
        else:
            lines.append("mode          : real multi-GPU (NCCL)")
        return "\n".join(lines)


def detect() -> Env:
    """Detect platform and GPUs. Import of torch is deferred so this module
    can be imported for its dataclass without a heavy dependency."""
    import torch

    if os.path.exists("/kaggle"):
        name = "kaggle"
    elif "google.colab" in sys.modules or os.path.exists("/content"):
        name = "colab"
    else:
        name = "local"

    n = torch.cuda.device_count()
    names, mems = [], []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        names.append(p.name)
        mems.append(p.total_memory / 2**30)

    return Env(
        platform_name=name,
        num_gpus=n,
        gpu_names=names,
        gpu_mem_gb=mems,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        nccl_available=torch.distributed.is_nccl_available() and n > 0,
    )


if __name__ == "__main__":
    print(detect().banner())
