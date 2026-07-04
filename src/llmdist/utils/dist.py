"""Distributed setup helpers.

Two ways to launch distributed code in this course:

1. `torchrun --nproc_per_node=N script.py`
   torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT as
   environment variables and starts N copies of the script. Scripts call
   `setup_from_env()`.

2. Inside a notebook: `run_distributed(fn, world_size=2)`
   Uses torch.multiprocessing.spawn so multi-process experiments work in a single
   notebook cell. Defaults to Gloo/CPU so it runs anywhere; pass backend="nccl"
   on a multi-GPU machine.

Both paths end in the same place: `dist.init_process_group`, one process per rank,
all agreeing on a rendezvous (TCP store at MASTER_ADDR:MASTER_PORT).
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_from_env() -> tuple[int, int, torch.device]:
    """Initialize the process group from torchrun's environment variables.

    Returns (rank, world_size, device). Binds each rank to its LOCAL_RANK GPU
    when CUDA is available — forgetting this binding is the #1 cause of
    'all ranks use GPU 0' bugs.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available() and torch.cuda.device_count() > local_rank:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=120))
    return rank, world_size, device


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _worker(rank: int, world_size: int, backend: str, port: int,
            fn: Callable[..., Any], args: tuple) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=120))
    if backend == "nccl":
        torch.cuda.set_device(rank)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Callable[..., Any], world_size: int = 2,
                    backend: str = "gloo", port: int = 29500,
                    args: tuple = ()) -> None:
    """Run `fn(rank, world_size, *args)` in `world_size` fresh processes.

    Designed for notebook cells: spawn (not fork) gives each worker a clean
    CUDA context, and joins before returning so cell output is complete.
    """
    mp.spawn(_worker, args=(world_size, backend, port, fn, args),
             nprocs=world_size, join=True)


def rank0_print(*a: Any, **kw: Any) -> None:
    """Print only on rank 0 (or when not distributed)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, **kw)
