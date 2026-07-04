"""Memory timeline across the first training steps of MiniGPT.

Shows the four phases of each step and — the point of the experiment — the
optimizer-state jump: Adam's m and v (8 bytes/param, fp32) are allocated
lazily inside the FIRST optimizer.step(), raising the memory floor by exactly
N*8 bytes.

Run:  python experiments/ch04/memory_timeline.py [--steps 3] [--save-plots]
CPU fallback: no CUDA allocator stats, so we account bytes by walking the
live gradient/optimizer tensors — same ledger, exact numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.memory import mb, snapshot
from llmdist.utils.env_check import detect


def account_bytes(model: torch.nn.Module, opt: torch.optim.Optimizer) -> tuple[float, float, float]:
    """(params_mb, grads_mb, opt_state_mb) from live tensors — device-agnostic."""
    p = sum(q.numel() * q.element_size() for q in model.parameters())
    g = sum(q.grad.numel() * q.grad.element_size()
            for q in model.parameters() if q.grad is not None)
    s = sum(t.numel() * t.element_size() for st in opt.state.values()
            for t in st.values() if torch.is_tensor(t))
    return mb(p), mb(g), mb(s)


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
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-plots", action="store_true")
    args = ap.parse_args()

    env = detect()
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    print(env.banner())
    print(f"\ndevice: {device}\n")

    cfg = GPTConfig(vocab_size=8192, block_size=128, n_layer=4, n_head=4, d_model=256)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    N = model.num_params()
    print(f"MiniGPT: {N:,} params -> predicted optimizer-state allocation = "
          f"{mb(N * 8):.1f} MB (Adam m+v, fp32), appearing at the first step()\n")

    x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)

    rows: list[list[str]] = []
    series: list[tuple[str, float]] = []

    def record(step: int, phase: str) -> float:
        p, g, s = account_bytes(model, opt)
        total = snapshot(device).allocated_mb if device.type == "cuda" else p + g + s
        rows.append([str(step), phase, f"{p:.1f}", f"{g:.1f}", f"{s:.1f}", f"{total:.1f}"])
        series.append((f"s{step}:{phase}", total))
        return total

    opt_before_first_step = account_bytes(model, opt)[2]
    for step in range(args.steps):
        _, loss = model(x, y)
        record(step, "forward")
        loss.backward()
        record(step, "backward")
        opt.step()
        record(step, "step")
        opt.zero_grad(set_to_none=True)
        record(step, "zero_grad")
    opt_after = account_bytes(model, opt)[2]

    print_table(rows, ["step", "phase", "params MB", "grads MB", "opt MB",
                       "alloc MB" if device.type == "cuda" else "accounted MB"])

    print(f"\noptimizer state before first step(): {opt_before_first_step:.1f} MB")
    print(f"optimizer state after:               {opt_after:.1f} MB "
          f"(predicted {mb(N * 8):.1f} MB)")
    print("read the 'opt MB' column: 0.0 through step 0's forward/backward, then a")
    print("one-time jump inside step() — the classic 'OOM after the first step' cause.")

    if args.save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[2] / "visualizations"
        out.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot([v for _, v in series], "o-")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([k for k, _ in series], rotation=45, ha="right")
        ax.set_ylabel("MB")
        ax.set_title(f"Memory timeline, first {args.steps} steps ({device})")
        fig.tight_layout()
        fig.savefig(out / "ch04_memory_timeline.png", dpi=120)
        print(f"saved {out / 'ch04_memory_timeline.png'}")


if __name__ == "__main__":
    main()
