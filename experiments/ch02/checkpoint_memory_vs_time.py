"""Activation checkpointing on MiniGPT: memory saved vs time paid.

Mechanism (ch02): checkpoint(fn, x) saves only x (+RNG state) in forward and
re-runs fn during backward. Predicted cost: ~4/3x FLOPs for the checkpointed
region (fwd 2N : bwd 4N : recompute +2N). Predicted savings: the region's
intermediate activations are never stored.

Measurement is CPU-safe: saved-for-backward bytes are weighed exactly with
torch.autograd.graph.saved_tensors_hooks (dedup by storage), so no CUDA memory
counters are needed. On GPU we ALSO report max_memory_allocated deltas.

Run:  python experiments/ch02/checkpoint_memory_vs_time.py [--layers 4]
             [--d-model 128] [--seq 128] [--batch 4] [--iters 5]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils import env_check


def forward_loss(model: MiniGPT, idx: torch.Tensor, use_ckpt: bool) -> torch.Tensor:
    """MiniGPT forward with optional per-block checkpointing.

    We re-implement the top-level forward so each Block can be wrapped; the
    blocks themselves are untouched library code.
    """
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    x = model.drop(model.tok_emb(idx) + model.pos_emb(pos))
    for blk in model.blocks:
        x = checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
    logits = model.head(model.ln_f(x))
    return F.cross_entropy(logits.view(-1, logits.size(-1)), idx.view(-1))


def measure(model: MiniGPT, idx: torch.Tensor, use_ckpt: bool, iters: int,
            device: torch.device) -> dict[str, float]:
    saved_bytes = 0.0
    times = []
    cuda = device.type == "cuda"
    for it in range(iters + 1):  # +1 warmup
        seen: dict[int, int] = {}

        def pack(t: torch.Tensor) -> torch.Tensor:
            base = t if t._base is None else t._base
            seen[id(base)] = base.numel() * base.element_size()
            return t

        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = forward_loss(model, idx, use_ckpt)
        loss.backward()
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        model.zero_grad(set_to_none=True)
        if it > 0:
            times.append(dt)
            saved_bytes = sum(seen.values())
    out = {"saved_mb": saved_bytes / 2**20, "ms": statistics.median(times)}
    if cuda:
        out["peak_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    torch.manual_seed(0)

    cfg = GPTConfig(vocab_size=512, block_size=args.seq, n_layer=args.layers,
                    n_head=4, d_model=args.d_model)
    model = MiniGPT(cfg).to(device)
    idx = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    print(f"MiniGPT: L={args.layers}, d={args.d_model}, T={args.seq}, B={args.batch}, "
          f"params={model.num_params():,}\n")

    base = measure(model, idx, use_ckpt=False, iters=args.iters, device=device)
    ckpt = measure(model, idx, use_ckpt=True, iters=args.iters, device=device)

    cols = ["saved-for-bwd MB", "fwd+bwd ms"] + (["CUDA peak MB"] if "peak_mb" in base else [])
    print(f"{'config':<16}" + "".join(f"{c:>18}" for c in cols))
    for name, r in [("baseline", base), ("checkpointed", ckpt)]:
        row = f"{name:<16}{r['saved_mb']:>18.1f}{r['ms']:>18.1f}"
        if "peak_mb" in r:
            row += f"{r['peak_mb']:>18.1f}"
        print(row)

    print(f"\nactivation memory ratio : {base['saved_mb'] / max(ckpt['saved_mb'], 1e-9):.1f}x smaller")
    print(f"time ratio              : {ckpt['ms'] / base['ms']:.2f}x slower "
          f"(theory: ~1.33x for the recomputed region; embeddings/head are not "
          f"recomputed, so expect a bit less)")
    print("\nSweep idea: --layers 8 --seq 256 makes the trade starker; the saved-MB")
    print("column should scale ~linearly in layers*seq*batch (ch03 derives the formula).")


if __name__ == "__main__":
    main()
