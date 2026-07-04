"""FLOPs estimate vs measured time: how honest is 6N (+ attention)?

We compute MiniGPT.flops_per_token() = 6N + 12*L*T*d, time forward and
forward+backward (CUDA events on GPU via llmdist timeit), and report:
  - achieved FLOP/s and, on a known GPU, % of peak (MFU);
  - the fwd : fwd+bwd time ratio vs the theoretical 1:3 (fwd 2N, bwd 4N);
  - scaling: time per token should grow ~linearly with flops_per_token
    across model widths - the accountant's formulas predict wall-clock.

Run:  python experiments/ch03/flops_vs_time.py [--widths 128,256,512] [--iters 5]
CPU-safe (small widths; on CPU the lesson is ratios and scaling, not MFU).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.timers import timeit
from llmdist.utils import env_check

SPEC_PEAK_FP32_TFLOPS = {"Tesla T4": 8.1, "A100": 19.5}  # vendor specs, not measured


def peak_for(name: str) -> float | None:
    for k, v in SPEC_PEAK_FP32_TFLOPS.items():
        if k.lower() in name.lower():
            return v
    return None


def bench_width(d: int, batch: int, seq: int, iters: int,
                device: torch.device) -> dict[str, float]:
    cfg = GPTConfig(vocab_size=2048, block_size=seq, n_layer=4,
                    n_head=max(2, d // 64), d_model=d)
    model = MiniGPT(cfg).to(device)
    idx = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)

    def fwd():
        with torch.no_grad():
            model(idx, targets=idx)

    def fwd_bwd():
        _, loss = model(idx, targets=idx)
        loss.backward()
        model.zero_grad(set_to_none=True)

    t_f = timeit(fwd, iters=iters, warmup=2, device=device)
    t_fb = timeit(fwd_bwd, iters=iters, warmup=2, device=device)
    tokens = batch * seq
    fpt = model.flops_per_token()
    return {
        "d": d,
        "flops_per_tok": fpt,
        "fwd_ms": t_f["mean_ms"],
        "fwdbwd_ms": t_fb["mean_ms"],
        "gflops": fpt * tokens / (t_fb["mean_ms"] * 1e-3) / 1e9,
        "ratio": t_fb["mean_ms"] / t_f["mean_ms"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=str, default=None, help="comma list of d_model")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    on_gpu = env.num_gpus >= 1
    device = torch.device("cuda:0" if on_gpu else "cpu")
    torch.manual_seed(0)

    if args.widths:
        widths = [int(w) for w in args.widths.split(",")]
    else:
        widths = [128, 256, 512, 1024] if on_gpu else [64, 128, 256]

    peak = peak_for(env.gpu_names[0]) if on_gpu else None
    mfu_hdr = f" {'MFU(fp32)':>10}" if peak else ""
    hdr = (f"{'d_model':>8} {'MFLOPs/tok':>11} {'fwd ms':>9} {'fwd+bwd ms':>11} "
           f"{'ratio':>6} {'GFLOP/s':>9}{mfu_hdr}")
    print(f"batch={args.batch}, seq={args.seq}, layers=4\n")
    print(hdr)
    print("-" * len(hdr))
    for d in widths:
        r = bench_width(d, args.batch, args.seq, args.iters, device)
        row = (f"{r['d']:>8} {r['flops_per_tok'] / 1e6:>11.1f} {r['fwd_ms']:>9.2f} "
               f"{r['fwdbwd_ms']:>11.2f} {r['ratio']:>6.2f} {r['gflops']:>9.1f}")
        if peak:
            row += f" {100 * r['gflops'] / (peak * 1e3):>9.1f}%"
        print(row)

    print(
        "\nHow to read this:\n"
        "  - ratio column: theory says fwd:fwd+bwd = 1:3 (backward = two matmuls\n"
        "    per forward matmul). Small models smear it with overhead; it should\n"
        "    approach 3 as d grows and compute dominates (ch01 regimes).\n"
        "  - GFLOP/s should RISE with d: bigger matmuls climb the roofline.\n"
        "  - MFU (on GPU): tiny models sit at single digits - that is launch and\n"
        "    memory-bound overhead, not a broken GPU. GPT-scale training targets\n"
        "    40-60% MFU. Never invent these numbers: this script measures them."
    )


if __name__ == "__main__":
    main()
