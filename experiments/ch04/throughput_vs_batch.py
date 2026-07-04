"""Training throughput (tokens/s) vs batch size for MiniGPT.

Expected shape (derived, not benchmarked): tokens/s rises with batch size
while per-step fixed costs (kernel launches, Python dispatch) are amortized
and SM occupancy improves, then plateaus at the compute roof. The knee is
hardware-specific — this script finds yours.

Run:  python experiments/ch04/throughput_vs_batch.py [--save-plots]
CPU fallback: smaller batch range and fewer iters; the curve shape still shows
launch-overhead amortization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.timers import timeit
from llmdist.utils.env_check import detect


def print_table(rows: list[list[str]], header: list[str]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        t = Table(*header)
        for r in rows:
            t.add_row(*r)
        Console().print(t)
    except ImportError:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        print(fmt.format(*("-" * w for w in widths)))
        for r in rows:
            print(fmt.format(*r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=0, help="0 = auto (fewer on CPU)")
    args = ap.parse_args()

    env = detect()
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    print(env.banner())
    print(f"\ndevice: {device}\n")

    cfg = GPTConfig(vocab_size=8192, block_size=args.seq_len, n_layer=4,
                    n_head=4, d_model=256)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    T = cfg.block_size

    if device.type == "cuda":
        batches = [1, 2, 4, 8, 16, 32, 64]
        iters = args.iters or 10
    else:
        batches = [1, 2, 4, 8]
        iters = args.iters or 3

    flops_per_tok = model.flops_per_token()
    print(f"model: {model.num_params():,} params, "
          f"~{flops_per_tok / 1e6:.1f} MFLOPs/token (training, ~6N + attention)\n")

    rows, bs_list, tok_list = [], [], []
    for B in batches:
        x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

        def one_step() -> None:
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        try:
            t = timeit(one_step, iters=iters, warmup=2, device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"B={B}: OOM — stopping sweep (that IS a data point: ch05).")
            break
        tok_s = B * T / (t["mean_ms"] / 1e3)
        tflops = tok_s * flops_per_tok / 1e12
        rows.append([str(B), f"{t['mean_ms']:.1f}", f"{tok_s:,.0f}", f"{tflops:.3f}"])
        bs_list.append(B)
        tok_list.append(tok_s)

    print_table(rows, ["batch", "ms/step", "tokens/s", "achieved TFLOP/s"])
    if len(tok_list) >= 2:
        gain = tok_list[-1] / tok_list[0]
        print(f"\ntokens/s gain from B={bs_list[0]} to B={bs_list[-1]}: {gain:.1f}x")
        print("rising then flattening = per-step overhead amortized, then compute-bound.")

    if args.save_plots and bs_list:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bs_list, tok_list, "o-")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.set_ylabel("tokens/s")
        ax.set_title(f"MiniGPT training throughput vs batch ({device})")
        fig.tight_layout()
        fig.savefig(out / "ch04_throughput_vs_batch.png", dpi=120)
        print(f"saved {out / 'ch04_throughput_vs_batch.png'}")


if __name__ == "__main__":
    main()
