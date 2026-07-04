"""Memory-bound vs compute-bound ops: place real kernels on the roofline.

For each op we know FLOPs and compulsory bytes analytically, so we can report
achieved GFLOP/s AND achieved GB/s. The signature of a memory-bound op: it hits
a large fraction of peak *bandwidth* and a tiny fraction of peak *FLOPs*.
Matmul (intensity n/6 fp32) shows the reverse.

Run:  python experiments/ch01/roofline_ops.py [--n 2048] [--save-plots]
CPU-safe: runs a smaller sweep on CPU (CPUs have rooflines too).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import torch.nn.functional as F

from llmdist.profiling.timers import bandwidth_gbps, timeit
from llmdist.utils import env_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="square tensor side")
    ap.add_argument("--save-plots", action="store_true")
    args = ap.parse_args()

    env = env_check.detect()
    print(env.banner(), "\n")
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    n = args.n or (4096 if env.num_gpus >= 1 else 512)

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    s4 = 4  # fp32 bytes
    nn_ = n * n

    # op name -> (fn, flops, compulsory bytes)
    ln = torch.nn.LayerNorm(n).to(device)
    ops = [
        ("add (x+y)",       lambda: a + b,             1 * nn_,      3 * nn_ * s4),
        ("mul (x*y)",       lambda: a * b,             1 * nn_,      3 * nn_ * s4),
        ("gelu(x)",         lambda: F.gelu(a),         10 * nn_,     2 * nn_ * s4),
        ("softmax(x,-1)",   lambda: F.softmax(a, -1),  5 * nn_,      2 * nn_ * s4),
        ("layernorm(x)",    lambda: ln(a),             8 * nn_,      2 * nn_ * s4),
        ("matmul (a@b)",    lambda: a @ b,             2 * n**3,     3 * nn_ * s4),
    ]

    hdr = (f"{'op':<16} {'ms':>9} {'GFLOP/s':>10} {'GB/s':>8} "
           f"{'I (F/B)':>9} {'verdict':>14}")
    print(f"square size n={n} (each tensor {nn_ * s4 / 2**20:.0f} MiB fp32)\n")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, fn, flops, nbytes in ops:
        st = timeit(fn, iters=10, warmup=3, device=device)
        gflops = flops / (st["mean_ms"] * 1e-3) / 1e9
        gbs = bandwidth_gbps(nbytes, st["mean_ms"])
        intensity = flops / nbytes
        verdict = "compute-bound" if intensity > 25 else "memory-bound"
        rows.append((name, intensity, gflops))
        print(f"{name:<16} {st['mean_ms']:>9.3f} {gflops:>10.1f} {gbs:>8.1f} "
              f"{intensity:>9.2f} {verdict:>14}")

    print(
        "\nHow to read this:\n"
        "  - all elementwise/normalization ops should land near the SAME GB/s\n"
        "    (a big fraction of the machine's bandwidth): their time is bytes/BW.\n"
        "  - their GFLOP/s differ wildly and are all tiny vs the matmul: in the\n"
        "    memory-bound regime, arithmetic is free.\n"
        "  - verdict threshold shown is the T4 fp32 ridge (~25 FLOP/B); recompute\n"
        "    P/B for your own device."
    )

    if args.save_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        # empirical roofline: use best measured bandwidth & flops as pseudo-peaks
        best_bw_flops = max(r[2] / r[1] for r in rows)  # GFLOP/s per intensity ~ B
        best_p = max(r[2] for r in rows)
        xs = np.logspace(-2, 3, 100)
        roof = np.minimum(best_p, xs * best_bw_flops)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(xs, roof, "k-", lw=1, label="empirical roofline")
        for name, intensity, gflops in rows:
            ax.loglog([intensity], [gflops], "o")
            ax.annotate(name, (intensity, gflops), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("achieved GFLOP/s")
        ax.set_title("measured ops on the roofline")
        ax.legend()
        fig.tight_layout()
        path = out / "ch01_roofline.png"
        fig.savefig(path, dpi=120)
        print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
