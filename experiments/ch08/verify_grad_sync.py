"""Prove DP correctness: averaged shard gradients == large-batch gradients.

    torchrun --nproc_per_node=2 experiments/ch08/verify_grad_sync.py
    python experiments/ch08/verify_grad_sync.py --nproc 2

ch08 derives that data parallelism with gradient averaging computes EXACTLY
the gradient of the loss over the concatenated global batch (linearity of
differentiation + mean-of-means with equal shards). This script checks the
claim numerically per parameter, and prints the residual — which is NOT
zero, because floating-point addition is performed in a different order on
the two paths (all-reduce tree/ring order vs one big batch's kernel order).
That residual is the "bitwise-ish" caveat, quantified.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.parallel.manual_dp import average_gradients  # noqa: E402
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402
from llmdist.workers.ch08 import make_batch, synthetic_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=4, help="per-rank batch size")
    ap.add_argument("--atol", type=float, default=1e-5)
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=4, d_model=128)
    torch.manual_seed(0)                          # identical init on all ranks
    model = MiniGPT(cfg)
    reference = MiniGPT(cfg)
    reference.load_state_dict(model.state_dict())

    stream = synthetic_tokens(n_tokens=16_384)
    shards = [list(range(r * args.bsz, (r + 1) * args.bsz)) for r in range(world)]

    # Path A: distributed — my shard, backward, average across ranks.
    x, y = make_batch(stream, shards[rank], cfg.block_size)
    model(x, y)[1].backward()
    average_gradients(model)

    # Path B: single process — everyone's samples as one batch. Every rank
    # can replay this because the data pipeline is deterministic.
    flat_idx = [i for s in shards for i in s]
    xg, yg = make_batch(stream, flat_idx, cfg.block_size)
    reference(xg, yg)[1].backward()

    rows: list[tuple[str, float, float]] = []
    ok = True
    for (name, p), q in zip(model.named_parameters(), reference.parameters()):
        diff = (p.grad - q.grad).abs().max().item()
        scale = q.grad.abs().max().item()
        rows.append((name, diff, scale))
        ok &= torch.allclose(p.grad, q.grad, atol=args.atol, rtol=1e-4)

    if rank == 0:
        rows.sort(key=lambda r: -r[1])
        print(f"{'parameter':<28} {'max |dp-single|':>16} {'max |grad|':>12}")
        for name, diff, scale in rows[:8]:
            print(f"{name:<28} {diff:>16.3e} {scale:>12.3e}")
        print(f"... ({len(rows)} parameters total)")
        worst = rows[0][1]
        print(f"\nverdict: {'MATCH' if ok else 'MISMATCH'} within atol={args.atol}")
        print(f"worst residual {worst:.3e} — nonzero because reduction ORDER "
              f"differs between the two paths; float addition is not associative.")
        print("Same reason two 'identical' runs of torch DDP can differ in the "
              "last bits: correctness tolerances, not equality signs.")
        assert ok, "gradient sync verification FAILED"


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        rank, world, _ = setup_from_env()
        try:
            worker(rank, world)
        finally:
            cleanup()
    else:
        run_distributed(worker, world_size=args.nproc, port=29554)


if __name__ == "__main__":
    main()
