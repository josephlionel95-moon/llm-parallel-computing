"""Record the ORDER in which parameter gradients become ready during backward.

Foreshadows DDP bucketing (ch09): DDP attaches per-parameter hooks; when a
bucket's parameters are all 'ready' it launches an async AllReduce - possible
only because gradients arrive roughly in REVERSE forward order (tail first),
while earlier layers are still computing.

This script hooks every MiniGPT parameter, runs one backward, prints arrival
order + timestamps, and computes a rank correlation against reversed forward
order. Watch for the weight-tying anomaly: tok_emb.weight == head.weight, so it
accumulates only after BOTH uses contribute -> lands near the end.

Run:  python experiments/ch02/grad_arrival_order.py [--layers 4]
CPU-only friendly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, no scipy needed."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    torch.manual_seed(0)

    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=args.layers,
                    n_head=4, d_model=64)
    model = MiniGPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 64))

    names = [n for n, _ in model.named_parameters()]
    fwd_index = {n: i for i, n in enumerate(names)}
    arrivals: list[tuple[str, float]] = []
    t_start = time.perf_counter()

    def make_hook(name: str):
        def hook(*_args) -> None:
            arrivals.append((name, (time.perf_counter() - t_start) * 1e3))
        return hook

    handles = []
    for name, p in model.named_parameters():
        if hasattr(p, "register_post_accumulate_grad_hook"):
            handles.append(p.register_post_accumulate_grad_hook(make_hook(name)))
        else:  # torch < 2.1 fallback: fires on grad computation for the leaf
            handles.append(p.register_hook(make_hook(name)))

    _, loss = model(idx, targets=idx)
    t_start = time.perf_counter()
    loss.backward()
    for h in handles:
        h.remove()

    print(f"{'arrival':>7} {'t (ms)':>8}  {'fwd idx':>7}  parameter")
    print("-" * 64)
    for i, (name, t_ms) in enumerate(arrivals):
        tied = "  <- TIED to head: waits for both uses" if name == "tok_emb.weight" else ""
        print(f"{i:>7} {t_ms:>8.3f}  {fwd_index[name]:>7}  {name}{tied}")

    rho = spearman([float(i) for i in range(len(arrivals))],
                   [float(fwd_index[n]) for n, _ in arrivals])
    print(f"\nSpearman(arrival rank, forward index) = {rho:+.3f}")
    print("Expect strongly NEGATIVE (reverse order). This is the property DDP's")
    print("bucketing exploits: it AllReduces late-layer buckets while early layers")
    print("still compute. The tied embedding is the canonical exception - DDP must")
    print("place it as if it were 'last', whatever its forward position.")


if __name__ == "__main__":
    main()
