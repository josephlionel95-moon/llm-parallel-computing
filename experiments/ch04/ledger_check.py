"""Predicted vs measured static training memory for several MiniGPT sizes.

The ch04 ledger predicts steady-state static memory (params + grads + Adam
m,v) = 16 bytes/param in fp32. This script builds models of increasing width,
runs two steps, and compares the prediction with:
  - CUDA: measured allocated memory (activations excluded by measuring at the
    floor, right after zero_grad, plus grads re-materialized via one backward)
  - CPU:  exact byte accounting over live tensors.

Run:  python experiments/ch04/ledger_check.py [--save-plots]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.memory import mb, model_memory_ledger, snapshot
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


def measure_static_mb(cfg: GPTConfig, device: torch.device) -> tuple[int, float]:
    """Train 2 tiny steps, then report the static account (params+grads+opt)."""
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.block_size)), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.block_size)), device=device)
    for _ in range(2):
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        # NOTE: no zero_grad on the last iteration — we want grads alive so the
        # measured floor includes all three static accounts.
    n = model.num_params()

    if device.type == "cuda":
        del x, y, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        measured = snapshot(device).allocated_mb
    else:
        p = sum(q.numel() * q.element_size() for q in model.parameters())
        g = sum(q.grad.numel() * q.grad.element_size()
                for q in model.parameters() if q.grad is not None)
        s = sum(t.numel() * t.element_size() for st in opt.state.values()
                for t in st.values() if torch.is_tensor(t))
        measured = mb(p + g + s)

    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return n, measured


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512, 768])
    args = ap.parse_args()

    env = detect()
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    print(env.banner())
    print(f"\ndevice: {device}")
    print("prediction: 16 B/param = params(4) + grads(4) + Adam m(4) + v(4)\n")

    rows, preds, meas_list, labels = [], [], [], []
    for d in args.widths:
        cfg = GPTConfig(vocab_size=8192, block_size=128, n_layer=4,
                        n_head=max(4, d // 64), d_model=d)
        probe = MiniGPT(cfg)
        pred = model_memory_ledger(probe, optimizer_bytes_per_param=8, dtype_bytes=4)
        del probe
        n, measured = measure_static_mb(cfg, device)
        ratio = measured / pred["total_static_mb"] if pred["total_static_mb"] else 0.0
        rows.append([f"d={d}", f"{n:,}", f"{pred['total_static_mb']:.1f}",
                     f"{measured:.1f}", f"{ratio:.3f}"])
        preds.append(pred["total_static_mb"])
        meas_list.append(measured)
        labels.append(f"d={d}")

    print_table(rows, ["config", "params", "predicted MB", "measured MB", "ratio"])
    print("\nratio slightly above 1.0 is expected: non-parameter buffers (the causal")
    print("mask is block_size^2 bools per layer) and 512-byte allocator rounding.")

    if args.save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        xpos = range(len(labels))
        ax.bar([i - 0.2 for i in xpos], preds, width=0.4, label="predicted (16 B/param)")
        ax.bar([i + 0.2 for i in xpos], meas_list, width=0.4, label="measured")
        ax.set_xticks(list(xpos))
        ax.set_xticklabels(labels)
        ax.set_ylabel("MB")
        ax.set_title(f"Static training memory: ledger vs measurement ({device})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "ch04_ledger_check.png", dpi=120)
        print(f"saved {out / 'ch04_ledger_check.png'}")


if __name__ == "__main__":
    main()
