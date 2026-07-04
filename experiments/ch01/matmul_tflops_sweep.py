"""Matmul TFLOPS sweep: size x dtype, plus the tensor-core alignment cliff.

What it shows:
  - achieved TFLOPS rises with n (more parallelism + higher arithmetic
    intensity I = n/6 fp32) and plateaus near the hardware peak;
  - fp16/bf16 unlock tensor cores (T4: ~65 TFLOPS peak vs ~8.1 fp32);
  - odd sizes (n +/- 1) miss tensor-core tiling and fall off a cliff.

Run:  python experiments/ch01/matmul_tflops_sweep.py [--max-n 4096] [--save-plots]
Degrades to a small CPU sweep when no GPU is present (CI-safe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.profiling.timers import timeit
from llmdist.utils import env_check

# Vendor spec-sheet peaks (TFLOPS) for percent-of-peak columns. Not measurements.
KNOWN_PEAKS = {
    "Tesla T4": {"fp32": 8.1, "fp16": 65.0, "bf16": None},
    "A100": {"fp32": 19.5, "fp16": 312.0, "bf16": 312.0},
}


def peak_for(gpu_name: str, dtype_name: str) -> float | None:
    for key, peaks in KNOWN_PEAKS.items():
        if key.lower() in gpu_name.lower():
            return peaks.get(dtype_name)
    return None


def bench(n: int, dtype: torch.dtype, device: torch.device) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    stats = timeit(lambda: a @ b, iters=10, warmup=3, device=device)
    return 2 * n**3 / (stats["mean_ms"] * 1e-3) / 1e12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=4096)
    ap.add_argument("--save-plots", action="store_true")
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    on_gpu = env.num_gpus >= 1
    device = torch.device("cuda:0" if on_gpu else "cpu")
    gpu_name = env.gpu_names[0] if on_gpu else "CPU"

    if on_gpu:
        sizes = [n for n in [256, 512, 1024, 2048, 4096, 8192] if n <= args.max_n]
        dtypes = [("fp32", torch.float32), ("fp16", torch.float16)]
        if torch.cuda.is_bf16_supported():
            dtypes.append(("bf16", torch.bfloat16))
    else:
        print("[no GPU] running a reduced CPU sweep so the code path is exercised;")
        print("         TFLOPS will be ~2 orders of magnitude below any GPU.\n")
        sizes = [128, 256, 512]
        dtypes = [("fp32", torch.float32)]

    results: dict[str, list[float]] = {name: [] for name, _ in dtypes}
    hdr = f"{'n':>6} " + "".join(f"{name + ' TFLOPS':>14}{'%peak':>7}" for name, _ in dtypes)
    print(hdr)
    print("-" * len(hdr))
    for n in sizes:
        row = f"{n:>6} "
        for name, dt in dtypes:
            tf = bench(n, dt, device)
            results[name].append(tf)
            pk = peak_for(gpu_name, name)
            pct = f"{100 * tf / pk:6.1f}%" if pk else "     -"
            row += f"{tf:>14.2f}{pct:>7}"
        print(row)

    # Alignment cliff: tensor cores want M, N, K multiples of 8/16.
    if on_gpu:
        print("\nAlignment probe (fp16): n vs n-1 vs n+1")
        base = min(4096, args.max_n)
        for n in [base - 1, base, base + 1]:
            tf = bench(n, torch.float16, device)
            print(f"  n={n:>5}: {tf:8.2f} TFLOPS")
        print("Expect the multiple-of-8 size to win clearly: ragged shapes miss "
              "tensor-core tiling.")

    if args.save_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, vals in results.items():
            ax.plot(sizes, vals, marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("matrix size n")
        ax.set_ylabel("achieved TFLOPS")
        ax.set_title(f"matmul TFLOPS vs size ({gpu_name})")
        ax.legend()
        fig.tight_layout()
        path = out / "ch01_matmul_tflops.png"
        fig.savefig(path, dpi=120)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
