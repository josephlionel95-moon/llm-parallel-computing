"""Train MiniGPT with MANUAL data parallelism on a synthetic corpus.

    torchrun --nproc_per_node=2 experiments/ch08/manual_dp_train.py
    python experiments/ch08/manual_dp_train.py --nproc 2
    python experiments/ch08/manual_dp_train.py --nproc 1   # baseline for speedup

The entire DP machinery is three calls from llmdist.parallel.manual_dp:
broadcast_parameters (once), shard_indices (per epoch), average_gradients
(per step). Prints loss curve, per-phase timing, comm fraction, tokens/s.
Run with --nproc 1 then --nproc 2 and compare tokens/s: the gap between 2x
and what you get IS the communication cost (plus any CPU contention).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.parallel.manual_dp import (  # noqa: E402
    average_gradients, broadcast_parameters, shard_indices)
from llmdist.utils.dist import cleanup, run_distributed, setup_from_env  # noqa: E402
from llmdist.workers.ch08 import make_batch, synthetic_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bsz", type=int, default=8, help="PER-RANK micro batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--per-tensor", action="store_true",
                    help="one all_reduce per grad tensor instead of one flat")
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def worker(rank: int, world: int) -> None:
    args = parse_args()
    device = (torch.device(f"cuda:{rank}") if dist.get_backend() == "nccl"
              else torch.device("cpu"))
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=4, d_model=128)

    torch.manual_seed(0)
    model = MiniGPT(cfg).to(device)
    broadcast_parameters(model)                   # replicas start identical
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = model.num_params()

    stream = synthetic_tokens()                   # deterministic, no download
    n_windows = len(stream) - cfg.block_size - 1
    phase = {"forward": 0.0, "backward": 0.0, "comm": 0.0, "optimizer": 0.0}
    losses: list[float] = []
    t_start = time.perf_counter()

    for step in range(args.steps):
        shard = shard_indices(n_windows, rank, world, epoch=step)
        x, y = make_batch(stream, shard[:args.bsz], cfg.block_size)
        x, y = x.to(device), y.to(device)

        t0 = time.perf_counter()
        _, loss = model(x, y)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        average_gradients(model, flat=not args.per_tensor)
        t3 = time.perf_counter()
        opt.step(); opt.zero_grad()
        t4 = time.perf_counter()
        phase["forward"] += t1 - t0
        phase["backward"] += t2 - t1
        phase["comm"] += t3 - t2
        phase["optimizer"] += t4 - t3

        loss_g = loss.detach().clone()
        dist.all_reduce(loss_g)
        losses.append(loss_g.item() / world)
        if rank == 0 and (step % max(1, args.steps // 10) == 0 or
                          step == args.steps - 1):
            print(f"step {step:>4}  loss {losses[-1]:.4f}", flush=True)

    elapsed = time.perf_counter() - t_start
    if rank == 0:
        total = sum(phase.values())
        tok_s = args.steps * args.bsz * cfg.block_size * world / elapsed
        print(f"\nmodel: {n_params:,} params ({n_params * 4 / 2**20:.1f} MB fp32 grads "
              f"all-reduced per step, mode={'per-tensor' if args.per_tensor else 'flat'})")
        print(f"world={world} backend={dist.get_backend()} device={device.type}")
        for k, v in phase.items():
            print(f"  {k:<10} {1e3 * v / args.steps:8.1f} ms/step ({100 * v / total:4.1f}%)")
        print(f"comm fraction: {100 * phase['comm'] / total:.1f}% "
              f"— this whole slice is serialized after backward (fixed in ch09)")
        print(f"throughput: {tok_s:,.0f} tokens/s global "
              f"({tok_s / world:,.0f} per process)")
        if args.save_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(losses)
            ax.set_xlabel("step"); ax.set_ylabel("global mean loss")
            ax.set_title(f"Manual DP, MiniGPT on synthetic bigram corpus "
                         f"(world={world}, global batch={args.bsz * world})")
            ax.grid(alpha=0.3)
            out = os.path.join(ROOT, "visualizations", f"ch08_dp_loss_w{world}.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.tight_layout(); fig.savefig(out, dpi=120)
            print(f"saved {out}")


def main() -> None:
    args = parse_args()
    if "RANK" in os.environ:
        rank, world, _ = setup_from_env()
        try:
            worker(rank, world)
        finally:
            cleanup()
    else:
        run_distributed(worker, world_size=args.nproc, port=29553)


if __name__ == "__main__":
    main()
