"""Hello, distributed world — the canonical first program (README quick start).

Every distributed program must prove four things: discover identity, bind a
device, pass a collective sanity check, shut down cleanly. This script does
exactly those four things and nothing else.

Launch (any of):
  torchrun --standalone --nproc_per_node=2 experiments/ch06/hello_distributed.py
  torchrun --nproc_per_node=2 experiments/ch06/hello_distributed.py   # needs MASTER_* or defaults
  python experiments/ch06/hello_distributed.py        # single-process fallback (world of 1)

On GPUs it uses NCCL and binds cuda:{LOCAL_RANK}; without GPUs it degrades to
Gloo on CPU — identical semantics, no NCCL speed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.distributed as dist

from llmdist.utils.dist import cleanup, setup_from_env


def main() -> None:
    if "RANK" not in os.environ:
        # Plain `python` launch: become a world of one so the same code path
        # (rendezvous included) still runs. torchrun would have set these.
        print("[fallback] RANK not set — no launcher detected; running world_size=1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

    # 1. identity + device binding (torch.cuda.set_device happens inside)
    rank, world_size, device = setup_from_env()
    backend = dist.get_backend()
    dev_name = (torch.cuda.get_device_name(device)
                if device.type == "cuda" else "CPU")
    print(f"[rank {rank}/{world_size}] pid={os.getpid()} backend={backend} "
          f"device={device} ({dev_name})")

    # 2. warmup all_reduce — the first collective triggers lazy communicator/
    #    channel construction (and catches rendezvous, binding, and fabric
    #    problems one line into the job instead of 40 minutes into training).
    t = torch.ones(1, device=device) * (rank + 1)
    dist.all_reduce(t)  # default op: SUM -> 1 + 2 + ... + p = p(p+1)/2

    expected = world_size * (world_size + 1) / 2
    ok = abs(t.item() - expected) < 1e-6
    print(f"[rank {rank}/{world_size}] all_reduce(1..{world_size}) = {t.item():.0f} "
          f"(expected {expected:.0f}) -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"rank {rank}: all_reduce sanity check FAILED")

    # 3. a barrier so the printouts above complete before anyone exits
    dist.barrier()
    if rank == 0:
        print(f"\nworld of {world_size} rank(s) alive and agreeing. "
              f"Chapter 07 opens the full collective toolbox.")

    # 4. clean shutdown — leaked process groups hold the rendezvous port
    cleanup()


if __name__ == "__main__":
    main()
