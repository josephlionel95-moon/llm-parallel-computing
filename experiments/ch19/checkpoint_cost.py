"""Checkpoint size & time vs the 12-bytes/param ledger.

    python experiments/ch19/checkpoint_cost.py
    python experiments/ch19/checkpoint_cost.py --d-models 128 256 512

For each model size: take one optimizer step (Adam state is born lazily —
chapter 04!), assemble the full training state, save it atomically, and
compare the file size against two predictions:

  exact ledger : sum of numel*element_size over every tensor in the state
  12 B/param   : the back-of-envelope rule (4 weights + 4 exp_avg + 4 exp_avg_sq)

Also times the two halves of a save — snapshot (clone to CPU) vs write
(serialize+fsync) — because the async checkpointing of the Limitations
section only overlaps the second half.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from llmdist.checkpoint import (  # noqa: E402
    load_checkpoint, save_checkpoint, training_state_dict)
from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d-models", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--n-layer", type=int, default=4)
    return ap.parse_args()


def tensor_bytes(x: object) -> int:
    """Recursively sum bytes of every tensor in a nested container."""
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(tensor_bytes(v) for v in x)
    return 0


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    mb = 1 / 2**20
    hdr = (f"{'d_model':>7} {'params':>10} {'file MB':>8} {'ledger MB':>9} "
           f"{'12B/p MB':>8} {'snap ms':>8} {'write ms':>8} {'load ms':>8}")
    print(hdr + "\n" + "-" * len(hdr))

    for d in args.d_models:
        cfg = GPTConfig(vocab_size=8192, block_size=128, n_layer=args.n_layer,
                        n_head=4, d_model=d)
        model = MiniGPT(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)

        # one real step so Adam's m, v exist (they are created lazily)
        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size + 1))
        _, loss = model(idx[:, :-1].contiguous(), targets=idx[:, 1:].contiguous())
        loss.backward()
        opt.step()

        t0 = time.perf_counter()
        state = training_state_dict(model, opt, step=1)      # snapshot half
        t1 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "step_00000001.pt")
            nbytes = save_checkpoint(path, state)            # write half
            t2 = time.perf_counter()
            fresh = MiniGPT(cfg)
            fresh_opt = torch.optim.Adam(fresh.parameters(), lr=3e-4)
            load_checkpoint(path, fresh, fresh_opt)
            t3 = time.perf_counter()

        n = model.num_params()
        ledger = tensor_bytes(state)          # exact: every tensor in the file
        rule = 12 * n                          # the mental-arithmetic version
        print(f"{d:>7} {n:>10,} {nbytes * mb:>8.1f} {ledger * mb:>9.1f} "
              f"{rule * mb:>8.1f} {(t1 - t0) * 1e3:>8.1f} "
              f"{(t2 - t1) * 1e3:>8.1f} {(t3 - t2) * 1e3:>8.1f}")

    print("\nReading the table:")
    print(" - file ≈ ledger (zip framing costs <1%): the state IS the bytes")
    print(" - ledger > 12B/param because of weight tying: head.weight IS")
    print("   tok_emb.weight, so parameters() (and 12·N) count the embedding")
    print("   once, but state_dict() has BOTH keys and our per-key clone()")
    print("   materializes it twice in the file. Buffers + RNG add smallprint.")
    print("   Exercise: dedup tied weights at save time and re-run this table.")
    print(" - snapshot is the part async checkpointing CANNOT overlap;")
    print("   write is the part it CAN.")


if __name__ == "__main__":
    main()
