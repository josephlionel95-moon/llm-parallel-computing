"""Sharded checkpointing with torch.distributed.checkpoint: save on 2 ranks,
resume on 1 — resharding as a non-event.

    python experiments/ch19/sharded_dcp.py
    torchrun --nproc_per_node=2 experiments/ch19/sharded_dcp.py --phase save
    torchrun --nproc_per_node=1 experiments/ch19/sharded_dcp.py --phase load

Phase "save": world_size=2, every rank calls dcp.save collectively — each
rank writes its own .distcp file plus a shared .metadata describing which
slices of which logical tensor live where (with replicated DDP-style state
DCP dedups, so the write is rank-0-heavy; with FSDP state it is truly
parallel — the API is IDENTICAL, which is the point).

Phase "load": world_size=1, a fresh model declares what it needs (its own
state_dict as destination tensors) and dcp.load routes bytes from whatever
layout the metadata records. World size changed 2 -> 1 and nothing cared.

Run without torchrun and the script spawns both phases itself (Gloo/CPU),
then verifies every tensor of the reloaded model against a reference file.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.utils.dist import run_distributed  # noqa: E402

# The DCP API has moved twice — a nice fossil record of the feature maturing:
#   1.13: torch.distributed._shard.checkpoint (prototype)
#   2.0 : torch.distributed.checkpoint, save_state_dict(storage_writer=...)
#   2.2+: dcp.save(state, checkpoint_id=...)   <- what you should write today
try:
    import torch.distributed.checkpoint as dcp

    def dcp_save(state: dict, path: str) -> None:
        if hasattr(dcp, "save"):                       # torch >= 2.2
            dcp.save(state, checkpoint_id=path)
        else:                                          # torch 2.0 / 2.1
            dcp.save_state_dict(state, storage_writer=dcp.FileSystemWriter(path))

    def dcp_load(state: dict, path: str) -> None:
        if hasattr(dcp, "load"):
            dcp.load(state, checkpoint_id=path)
        else:
            dcp.load_state_dict(state, storage_reader=dcp.FileSystemReader(path))
except ModuleNotFoundError:                            # torch 1.13 prototype
    from torch.distributed._shard.checkpoint import (
        FileSystemReader, FileSystemWriter, load_state_dict, save_state_dict)

    def dcp_save(state: dict, path: str) -> None:
        save_state_dict(state_dict=state, storage_writer=FileSystemWriter(path))

    def dcp_load(state: dict, path: str) -> None:
        load_state_dict(state_dict=state, storage_reader=FileSystemReader(path))

CFG = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, d_model=64)


def build_model(seed: int) -> MiniGPT:
    torch.manual_seed(seed)
    return MiniGPT(CFG)


def save_worker(rank: int, world: int, ckpt_dir: str, ref_path: str) -> None:
    """All ranks build identical replicas (broadcast-equivalent: same seed),
    then call dcp.save COLLECTIVELY — this is not rank-0-saves-for-everyone,
    it is everyone-saves-their-part."""
    model = build_model(seed=42)
    if rank == 0:  # reference copy for verification, via the ch19 plain path
        torch.save(model.state_dict(), ref_path)
    dcp_save({"model": model.state_dict()}, ckpt_dir)
    dist.barrier()
    if rank == 0:
        files = sorted(os.listdir(ckpt_dir))
        print(f"[save, world={world}] wrote: {files}")


def load_worker(rank: int, world: int, ckpt_dir: str, ref_path: str) -> None:
    """Different world size, different (wrong-seed) weights. dcp.load is
    IN-PLACE: we hand it destination tensors, it fills them from the shards."""
    model = build_model(seed=7)                    # wrong weights on purpose
    state = {"model": model.state_dict()}          # destination layout
    dcp_load(state, ckpt_dir)
    model.load_state_dict(state["model"])

    ref = torch.load(ref_path, map_location="cpu")
    bad = [k for k, v in model.state_dict().items()
           if not torch.equal(v, ref[k])]
    assert not bad, f"tensors differ after reshard-load: {bad}"
    print(f"[load, world={world}] all {len(ref)} tensors bitwise-equal "
          "to the world=2 save. Resharding 2->1: a non-event. ✓")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["save", "load", "both"], default="both")
    ap.add_argument("--dir", default=None, help="checkpoint directory")
    args = ap.parse_args()

    workdir = args.dir or tempfile.mkdtemp(prefix="ch19_dcp_")
    ckpt_dir = os.path.join(workdir, "dcp_ckpt")
    ref_path = os.path.join(workdir, "reference.pt")

    if "RANK" in os.environ:  # launched by torchrun
        from llmdist.utils.dist import cleanup, setup_from_env
        rank, world, _ = setup_from_env()
        if args.phase in ("save", "both"):
            save_worker(rank, world, ckpt_dir, ref_path)
        if args.phase in ("load", "both"):
            load_worker(rank, world, ckpt_dir, ref_path)
        cleanup()
        return

    # self-contained: spawn save with world=2, then load with world=1
    print(f"checkpoint dir: {ckpt_dir}")
    run_distributed(save_worker, world_size=2, backend="gloo", port=29512,
                    args=(ckpt_dir, ref_path))
    run_distributed(load_worker, world_size=1, backend="gloo", port=29513,
                    args=(ckpt_dir, ref_path))


if __name__ == "__main__":
    main()
