"""Shared-memory tensors: zero-copy data exchange between processes.

Demonstrates the mechanism underneath DataLoader worker->main transfers and
torch.multiprocessing queues: a CPU tensor whose storage lives in an OS
shared-memory segment. Children mutate it; the parent sees the writes with
no message passing and no serialization of the payload.

Also times "send a big tensor to a child": shared-memory handle vs pickled
bytes, to show the zero-copy path is ~independent of tensor size.

Run:  python experiments/ch00/shared_memory_tensor.py [--mb 64]
CPU-only; uses spawn (the CUDA-safe start method).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.multiprocessing as mp


def child_write(rank: int, world: int, t: torch.Tensor, barrier) -> None:
    """Each child writes its rank into its own slot of the SHARED storage.

    Note what was pickled to get here: the *handle* to the shm segment plus
    tensor metadata - not the payload bytes.
    """
    print(f"  [child {rank}] data_ptr={t.data_ptr():#x} (my mapping of the same pages)")
    t[rank] = float(100 + rank)
    barrier.wait()  # synchronization is OUR job: shm gives none for free


def child_checksum(t: torch.Tensor, q) -> None:
    q.put(float(t.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=64, help="size of the transfer-timing tensor")
    ap.add_argument("--world", type=int, default=4)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")

    # ---- Part 1: in-place mutation across processes -------------------------
    print("Part 1: children mutate the parent's tensor in place")
    t = torch.zeros(args.world)
    t.share_memory_()
    print(f"  [parent ] data_ptr={t.data_ptr():#x}  is_shared={t.is_shared()}  before={t.tolist()}")

    barrier = ctx.Barrier(args.world)
    procs = [ctx.Process(target=child_write, args=(r, args.world, t, barrier))
             for r in range(args.world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"  [parent ] after children exit: {t.tolist()}")
    assert t.tolist() == [float(100 + r) for r in range(args.world)], "lost an update?!"

    # ---- Part 2: cost of moving a big tensor to a child ---------------------
    print(f"\nPart 2: hand a {args.mb} MiB tensor to a child process")
    n = args.mb * 2**20 // 4
    big = torch.randn(n)

    # (a) shared memory: payload never crosses; only a handle does.
    big.share_memory_()
    q = ctx.SimpleQueue()
    t0 = time.perf_counter()
    p = ctx.Process(target=child_checksum, args=(big, q))
    p.start()
    checksum = q.get()
    p.join()
    t_shm = time.perf_counter() - t0

    # (b) pickled bytes: serialize the payload in-process (lower bound on the
    #     cost of NOT using shm - a real pipe transfer adds copies on top).
    import pickle

    t0 = time.perf_counter()
    blob = pickle.dumps(big.clone())  # clone: pickle a non-shared tensor
    _ = pickle.loads(blob)
    t_pkl = time.perf_counter() - t0

    print(f"  shm handle + child start + checksum : {t_shm * 1e3:8.1f} ms "
          f"(dominated by process startup, not size)")
    print(f"  pickle round-trip of the payload    : {t_pkl * 1e3:8.1f} ms "
          f"({len(blob) / 2**20:.0f} MiB serialized)")
    print(f"  child-computed checksum             : {checksum:.1f}")
    print(
        "\nTakeaway: share_memory_() moves *handles*, pickle moves *bytes*.\n"
        "Re-run with --mb 256: the pickle time scales linearly; the shm path barely moves.\n"
        "DataLoader returns every batch to you through exactly this shm mechanism."
    )


if __name__ == "__main__":
    main()
