"""A museum of distributed failure modes — deliberately broken, safely bounded.

Each --mode reproduces one classic failure with a short timeout, prints the
raw error, and explains symptom -> cause -> fix. Break things here, where it
costs 15 seconds, instead of in a training run where it costs an afternoon.

Modes:
  hang       WORLD_SIZE says 2, only 1 process shows up -> rendezvous timeout
  port       two stores bind the same MASTER_PORT -> address already in use
  nccl_cpu   NCCL backend handed a CPU tensor -> immediate backend error
  mismatch   rank 0 skips a collective others entered -> collective timeout
  gpu0       explanation (and demo if >=2 GPUs) of the 'everyone on GPU 0' bug

Run:  python experiments/ch06/failure_modes.py --mode hang
      python experiments/ch06/failure_modes.py --all
CPU-safe: all modes except gpu0's live demo run on Gloo/CPU; nccl_cpu degrades
to an explanation when no GPU/NCCL is present.
"""
from __future__ import annotations

import argparse
import datetime
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TIMEOUT_S = 12


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"FAILURE MODE: {title}")
    print("=" * 72)


def explain(symptom: str, cause: str, fix: str) -> None:
    print(f"\n  SYMPTOM : {symptom}")
    print(f"  CAUSE   : {cause}")
    print(f"  FIX     : {fix}")


# ---------------------------------------------------------------- mode: hang
def _lonely_worker(rank: int) -> None:
    # Claims WORLD_SIZE=2 but is launched alone: it waits in the rendezvous
    # for a partner who never comes, until the timeout fires.
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29700",
                      RANK="0", WORLD_SIZE="2")
    t0 = time.perf_counter()
    try:
        dist.init_process_group("gloo", rank=0, world_size=2,
                                timeout=datetime.timedelta(seconds=TIMEOUT_S))
        print("  [worker] init returned (should NOT happen)")
        dist.destroy_process_group()
    except Exception as e:
        print(f"  [worker] init_process_group raised after "
              f"{time.perf_counter() - t0:.1f} s:\n           "
              f"{type(e).__name__}: {str(e)[:140]}")


def mode_hang() -> None:
    banner("mismatched world size -> rendezvous hang")
    print(f"launching ONE process that claims WORLD_SIZE=2 (timeout {TIMEOUT_S} s)...")
    mp.spawn(_lonely_worker, nprocs=1, join=True)
    explain(
        "all launched ranks freeze inside init_process_group, then time out",
        "WORLD_SIZE promises p processes; fewer checked into the TCPStore "
        "(a rank crashed before init, or the launch count was wrong)",
        "launch counts must equal WORLD_SIZE; if a rank died, read ITS traceback "
        "first; keep dev timeouts short (ours: 120 s) so this fails in minutes, "
        "not NCCL's default 30-minute silence",
    )


# ---------------------------------------------------------------- mode: port
def mode_port() -> None:
    banner("MASTER_PORT already in use")
    port = 29701
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", port))     # simulate a zombie run holding the port
    holder.listen(1)
    print(f"a previous 'job' (this script) is holding 127.0.0.1:{port};")
    print("now rank 0 of a new job tries to host its TCPStore there...")
    try:
        dist.TCPStore("127.0.0.1", port, 1, True,
                      timeout=datetime.timedelta(seconds=TIMEOUT_S))
        print("  store created (unexpected on this platform)")
    except Exception as e:
        print(f"  TCPStore raised: {type(e).__name__}: {str(e)[:140]}")
    finally:
        holder.close()
    explain(
        "'address already in use' seconds after launching",
        "the rendezvous TCPStore is a real server socket owned by rank 0; a "
        "crashed/zombie previous run (or another student's job) still holds the port",
        "kill leftovers (pkill -f train.py), pick another --master_port, or use "
        "torchrun --standalone which picks a free port for you",
    )


# ------------------------------------------------------------ mode: nccl_cpu
def _nccl_cpu_worker(rank: int, world_size: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29702",
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=TIMEOUT_S))
    torch.cuda.set_device(rank)
    t_cpu = torch.ones(1)                      # CPU tensor — the bug
    try:
        dist.all_reduce(t_cpu)
        print(f"  [rank {rank}] all_reduce on CPU tensor succeeded (unexpected)")
    except Exception as e:
        print(f"  [rank {rank}] all_reduce(cpu_tensor) under NCCL raised:\n"
              f"             {type(e).__name__}: {str(e)[:140]}")
    dist.destroy_process_group()


def mode_nccl_cpu() -> None:
    banner("NCCL backend handed a CPU tensor")
    if torch.cuda.device_count() >= 2 and dist.is_nccl_available():
        mp.spawn(_nccl_cpu_worker, args=(2,), nprocs=2, join=True)
    else:
        print("(no 2-GPU NCCL here — showing the story instead of the stack trace)")
        print("  dist.init_process_group('nccl'); t = torch.ones(1)  # CPU!")
        print("  dist.all_reduce(t)  ->  RuntimeError: Tensors must be CUDA and dense")
    explain(
        "immediate error naming the tensor / 'Tensors must be CUDA and dense'",
        "NCCL's collectives are CUDA kernels moving GPU memory; a CPU tensor has "
        "no address on any GPU",
        "move the tensor to this rank's device first (t.to(f'cuda:{local_rank}')), "
        "or use Gloo for CPU tensors — match backend to tensor device, always",
    )


# ------------------------------------------------------------ mode: mismatch
def _mismatch_worker(rank: int, world_size: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29703",
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=TIMEOUT_S))
    t = torch.ones(1)
    try:
        if rank == 0:
            print("  [rank 0] skipping the all_reduce (a collective inside "
                  "`if rank != 0:` — the bug) and idling...")
            time.sleep(TIMEOUT_S + 3)          # rank 0 does 'other work'
        else:
            print(f"  [rank {rank}] entering all_reduce, waiting for rank 0...")
            dist.all_reduce(t)
            print(f"  [rank {rank}] finished (should NOT happen)")
    except Exception as e:
        print(f"  [rank {rank}] collective raised after timeout:\n"
              f"             {type(e).__name__}: {str(e)[:140]}")
    finally:
        dist.destroy_process_group()


def mode_mismatch() -> None:
    banner("mismatched collective -> mid-job hang")
    print(f"rank 0 skips an all_reduce that rank 1 enters (timeout {TIMEOUT_S} s)...")
    try:
        mp.spawn(_mismatch_worker, args=(2,), nprocs=2, join=True)
    except Exception as e:
        print(f"  spawn surfaced: {type(e).__name__}")
    explain(
        "job hangs at a random step; on GPUs, all at 100% util doing nothing "
        "(spinning inside the collective kernel)",
        "collectives are appointments: EVERY rank must execute EVERY collective, "
        "same order, same shapes; a collective inside rank-conditional code (or "
        "uneven data ending one rank's epoch early) strands the others",
        "hoist collectives out of `if rank == ...` blocks; even out data shards; "
        "TORCH_DISTRIBUTED_DEBUG=DETAIL logs each rank's calls so the divergence "
        "is visible; TORCH_NCCL_ASYNC_ERROR_HANDLING=1 turns NCCL hangs into errors",
    )


# ---------------------------------------------------------------- mode: gpu0
def mode_gpu0() -> None:
    banner("everyone on GPU 0 (missing torch.cuda.set_device)")
    print("CUDA's per-process 'current device' defaults to 0. If N ranks on a node")
    print("skip torch.cuda.set_device(LOCAL_RANK), all N allocate on GPU 0:")
    print("  nvidia-smi: GPU 0 holds N processes and N x memory; GPUs 1..N-1 idle;")
    print("  NCCL: 'Duplicate GPU detected: rank 0 and 1 both on CUDA device 0'.")
    explain(
        "GPU 0 OOMs while other GPUs sit idle, or NCCL aborts with duplicate-GPU",
        "device binding is per-process state that nobody set; RANK-based binding "
        "(cuda:{RANK}) breaks the same way on multi-node (invalid ordinal)",
        "torch.cuda.set_device(int(os.environ['LOCAL_RANK'])) before creating any "
        "tensor — one line; our llmdist.utils.dist.setup_from_env does it for you",
    )
    if torch.cuda.device_count() >= 2:
        print("\n(you have >=2 GPUs: try editing hello_distributed.py to remove the")
        print(" set_device call inside setup_from_env and watch nvidia-smi — then")
        print(" put it back.)")


MODES = {"hang": mode_hang, "port": mode_port, "nccl_cpu": mode_nccl_cpu,
         "mismatch": mode_mismatch, "gpu0": mode_gpu0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), default=None)
    ap.add_argument("--all", action="store_true", help="run every mode in sequence")
    args = ap.parse_args()

    if not args.all and args.mode is None:
        ap.print_help()
        print("\npick a way to fail, e.g.:  --mode hang")
        return

    modes = list(MODES) if args.all else [args.mode]
    for m in modes:
        MODES[m]()
    print("\nall demonstrations bounded their own damage. Real jobs default to a")
    print("30-MINUTE NCCL timeout — now you know why we always set it to 120 s.")


if __name__ == "__main__":
    main()
