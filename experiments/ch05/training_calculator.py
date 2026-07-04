"""Can I train model X on GPU Y — and how long will it take?

Pure arithmetic from ch04 (memory ledger: 16 B/param Adam static) and ch05
(compute: C = 6*N*D, Chinchilla D = 20*N, wall-clock = C / (peak * MFU)).
No GPU required — this is a calculator, and every intermediate is printed so
you can check the division.

Examples:
  python experiments/ch05/training_calculator.py                       # tour of standard models/GPUs
  python experiments/ch05/training_calculator.py --model 7e9 --gpu A100-80GB
  python experiments/ch05/training_calculator.py --model 175e9 --tokens 300e9 --gpu T4 --n-gpus 1
  python experiments/ch05/training_calculator.py --model 70e9 --gpu H100-80GB --target-days 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# name: (memory_gb, peak dense 16-bit TFLOP/s). T4 = fp16 tensor cores (no bf16
# on Turing); A100/H100 = bf16 dense (H100's 1979 figure is 2:4 sparsity — we
# use the dense number, the one training actually sees).
GPUS: dict[str, tuple[float, float]] = {
    "T4":        (16.0,  65.0),
    "A100-40GB": (40.0, 312.0),
    "A100-80GB": (80.0, 312.0),
    "H100-80GB": (80.0, 989.0),
}

STANDARD_MODELS: dict[str, float] = {
    "MiniGPT": 5.3e6, "GPT-2-124M": 124e6, "GPT-2-1.5B": 1.5e9,
    "7B": 7e9, "13B": 13e9, "70B": 70e9, "GPT-3-175B": 175e9,
}


def memory_report(n: float, bytes_per_param: float, gpu_mem: float, n_gpus: int,
                  sharded: bool) -> list[str]:
    need_gb = n * bytes_per_param / 1e9
    per_gpu = need_gb / n_gpus if sharded else need_gb
    fits = per_gpu <= gpu_mem * 0.95  # keep 5% headroom for context/workspace
    lines = [
        f"  static memory  = {n:.3g} params x {bytes_per_param:g} B/param = {need_gb:,.1f} GB",
        f"  per-GPU burden = {per_gpu:,.1f} GB "
        + ("(fully sharded across ranks — ZeRO-3/FSDP semantics)" if sharded and n_gpus > 1
           else "(replicated: plain DDP gives NO memory relief)"),
        f"  capacity       = {gpu_mem:.0f} GB/GPU -> "
        + ("FITS (before activations — budget those with ch03 formulas)" if fits
           else f"DOES NOT FIT (short {per_gpu - gpu_mem:,.1f} GB/GPU)"),
    ]
    return lines


def compute_report(n: float, tokens: float, peak_tflops: float, mfu: float,
                   n_gpus: int, target_days: float | None) -> list[str]:
    flops = 6.0 * n * tokens
    rate = peak_tflops * 1e12 * mfu * n_gpus
    seconds = flops / rate
    years = seconds / SECONDS_PER_YEAR
    lines = [
        f"  C = 6*N*D      = 6 x {n:.3g} x {tokens:.3g} = {flops:.3g} FLOPs",
        f"  cluster rate   = {n_gpus} GPU(s) x {peak_tflops:g} TF x {mfu:.0%} MFU "
        f"= {rate/1e12:,.1f} TFLOP/s",
        f"  wall-clock     = {seconds:,.0f} s = {seconds/86400:,.1f} days = {years:,.2f} GPU-years"
        + ("" if n_gpus == 1 else f" of cluster time ({years*n_gpus:,.1f} GPU-years total)"),
    ]
    if target_days is not None:
        need = flops / (peak_tflops * 1e12 * mfu * target_days * 86400)
        lines.append(f"  for {target_days:g}-day run -> need {need:,.0f} GPUs "
                     f"(communication ignored — ch07 tells you the real bill)")
    return lines


def analyze(n: float, gpu: str, n_gpus: int, tokens: float | None,
            bytes_per_param: float, mfu: float, sharded: bool,
            target_days: float | None) -> None:
    mem_gb, tflops = GPUS[gpu]
    d = tokens if tokens is not None else 20.0 * n
    print(f"\n=== {n:.3g} params on {n_gpus} x {gpu} "
          f"(D = {d:.3g} tokens{', Chinchilla 20N' if tokens is None else ''}) ===")
    print("[memory wall]")
    for line in memory_report(n, bytes_per_param, mem_gb, n_gpus, sharded):
        print(line)
    print("[compute wall]")
    for line in compute_report(n, d, tflops, mfu, n_gpus, target_days):
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=float, default=None,
                    help="parameter count, e.g. 7e9 (default: tour of standard models)")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="A100-80GB")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--tokens", type=float, default=None,
                    help="training tokens D (default: Chinchilla 20N)")
    ap.add_argument("--bytes-per-param", type=float, default=16.0,
                    help="static B/param: 16 Adam, 12 SGD+momentum, ~10 8-bit Adam")
    ap.add_argument("--mfu", type=float, default=0.40)
    ap.add_argument("--sharded", action="store_true",
                    help="assume static bytes divide across ranks (ZeRO-3/FSDP)")
    ap.add_argument("--target-days", type=float, default=None,
                    help="also compute GPUs needed to finish in this many days")
    args = ap.parse_args()

    print("Assumptions are printed inline; change them with flags. MFU=40% models a")
    print("well-tuned large run — measure YOURS (ch26) before trusting a plan.")

    if args.model is not None:
        analyze(args.model, args.gpu, args.n_gpus, args.tokens,
                args.bytes_per_param, args.mfu, args.sharded, args.target_days)
    else:
        for name, n in STANDARD_MODELS.items():
            print(f"\n#### {name}")
            analyze(n, args.gpu, args.n_gpus, args.tokens, args.bytes_per_param,
                    args.mfu, args.sharded, args.target_days)
        print("\nHeadlines to memorize: 7B needs ~112 GB static (no single GPU);")
        print("GPT-3 (300B tokens) on one T4 ~ 4 centuries; Chinchilla makes cost")
        print("quadratic in N (C = 120 N^2).")


if __name__ == "__main__":
    main()
