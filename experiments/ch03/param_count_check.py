"""Verify the closed-form parameter count against the live model, over a grid.

Formula (derived in docs/ch03):
    N = (V + T)d + L(12d^2 + 13d) + 2d          (bias=True; 13d -> 4d without)
implemented as MiniGPT.param_formula(); num_params() counts actual tensors
(weight tying dedups the head). Any model edit that breaks the match makes
this script fail loudly - the formula is a unit test for the architecture.

Run:  python experiments/ch03/param_count_check.py
CPU-only friendly, <10 s.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check


def breakdown(model: MiniGPT) -> dict[str, int]:
    c = model.cfg
    d = c.d_model
    return {
        "embeddings (V+T)d": (c.vocab_size + c.block_size) * d,
        "blocks L(12d^2+bias+4d)": c.n_layer * (12 * d * d + (9 * d if c.bias else 0) + 4 * d),
        "final LN 2d": 2 * d,
    }


def main() -> None:
    env = env_check.detect()
    print(env.banner(), "\n")
    torch.manual_seed(0)

    grid = [
        GPTConfig(),  # course default
        GPTConfig(vocab_size=1000, block_size=64, n_layer=2, n_head=2, d_model=64),
        GPTConfig(vocab_size=50257, block_size=128, n_layer=2, n_head=4, d_model=128),
        GPTConfig(vocab_size=512, block_size=32, n_layer=6, n_head=8, d_model=256),
        GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, d_model=64, bias=False),
    ]

    hdr = (f"{'V':>6} {'T':>5} {'L':>3} {'d':>5} {'bias':>5} "
           f"{'formula':>12} {'measured':>12} {'match':>6} {'emb %':>6}")
    print(hdr)
    print("-" * len(hdr))
    all_ok = True
    for cfg in grid:
        m = MiniGPT(cfg)
        f, n = m.param_formula(), m.num_params()
        ok = f == n
        all_ok &= ok
        emb_pct = 100 * (cfg.vocab_size + cfg.block_size) * cfg.d_model / n
        print(f"{cfg.vocab_size:>6} {cfg.block_size:>5} {cfg.n_layer:>3} {cfg.d_model:>5} "
              f"{str(cfg.bias):>5} {f:>12,} {n:>12,} {'OK' if ok else 'FAIL':>6} {emb_pct:>5.1f}%")

    print("\nComponent breakdown for the course default config:")
    m = MiniGPT(GPTConfig())
    for k, v in breakdown(m).items():
        print(f"  {k:<26} {v:>12,}")

    # Weight tying: one storage, one parameter, two FLOP appearances.
    print(f"\nweight tying: head.weight is tok_emb.weight -> "
          f"{m.head.weight is m.tok_emb.weight}; "
          f"unique parameter tensors = {sum(1 for _ in m.parameters())}")

    # Folk rule check: 12Ld^2 share grows with scale.
    print("\nfolk rule N ~ 12Ld^2:")
    for L, d in [(4, 256), (12, 768), (48, 1600), (96, 12288)]:
        full = 12 * L * d * d + (50257 + 2048) * d
        print(f"  L={L:>3}, d={d:>6}: blocks/{('N'):<2} = {100 * 12 * L * d * d / full:5.1f}%")

    assert all_ok, "param_formula() diverged from num_params() - investigate!"
    print("\nAll configs match. The architecture and the math agree.")


if __name__ == "__main__":
    main()
