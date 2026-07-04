"""OOM-point finder: grow MiniGPT until it no longer trains on your GPU.

For each width d_model we (1) PREDICT static memory from the ch04 ledger,
(2) attempt one full training step (forward + backward + Adam step — the step
that lazily allocates m and v, where naive sizing dies), and (3) report where
the wall actually is versus where arithmetic said it would be.

CPU-safe: without CUDA we cannot hit a real wall, so the script runs in
prediction-only mode — it prints the ledger's estimated OOM point for common
GPU capacities and clearly labels everything as an estimate. No fake numbers.

Run:  python experiments/ch05/oom_point_finder.py [--seq-len 256] [--batch 8]
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.profiling.memory import model_memory_ledger, snapshot
from llmdist.utils.env_check import detect

WIDTHS = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
COMMON_GPUS = {"T4": 16.0, "A100-40GB": 40.0, "A100-80GB": 80.0, "H100-80GB": 80.0}


def make_cfg(d: int, seq_len: int) -> GPTConfig:
    return GPTConfig(vocab_size=8192, block_size=seq_len, n_layer=8,
                     n_head=max(4, d // 64), d_model=d)


def predict_gb(cfg: GPTConfig, batch: int) -> tuple[int, float, float]:
    """(n_params, static_gb, activation_hotspots_gb) — fp32, ch03/ch04 formulas."""
    model = MiniGPT(cfg)
    led = model_memory_ledger(model)  # 16 B/param
    n = led["num_params"]
    del model
    B, T, d, H, L = batch, cfg.block_size, cfg.d_model, cfg.n_head, cfg.n_layer
    # dominant saved tensors per block: 2x (B,H,T,T) attention + 4d MLP hidden
    # + a few (B,T,d) residual-stream tensors; a deliberate LOWER bound.
    act = L * (2 * B * H * T * T + B * T * 4 * d + 6 * B * T * d) * 4
    return n, led["total_static_mb"] / 1024, act / 2**30


def try_one_step(cfg: GPTConfig, batch: int, device: torch.device) -> float:
    """Run one full training step; return measured peak GB. Raises on OOM."""
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    _, loss = model(x, y)
    loss.backward()
    opt.step()                      # <- the lazy Adam allocation lives here
    torch.cuda.synchronize(device)
    peak = snapshot(device).max_allocated_mb / 1024
    del model, opt, x, y, loss
    gc.collect()
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    env = detect()
    print(env.banner())

    if env.num_gpus >= 1:
        device = torch.device("cuda:0")
        cap_gb = env.gpu_mem_gb[0]
        print(f"\nprobing {env.gpu_names[0]} ({cap_gb:.1f} GB), "
              f"B={args.batch}, T={args.seq_len}. Predictions printed BEFORE probing.\n")
        header = (f"{'d_model':>8}{'params':>14}{'pred static GB':>15}"
                  f"{'pred +acts GB':>14}{'measured peak':>15}  verdict")
        print(header)
        last_ok = None
        for d in WIDTHS:
            cfg = make_cfg(d, args.seq_len)
            n, static_gb, act_gb = predict_gb(cfg, args.batch)
            line = f"{d:>8}{n:>14,}{static_gb:>15.2f}{static_gb + act_gb:>14.2f}"
            try:
                peak = try_one_step(cfg, args.batch, device)
                print(line + f"{peak:>13.2f} GB  ok")
                last_ok = d
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                print(line + f"{'—':>15}  *** OOM — the wall ***")
                print(f"\nlast width that trained: d_model={last_ok}")
                print(f"ledger said the wall arrives when pred(+acts) approaches "
                      f"{cap_gb:.1f} GB; compare the columns above.")
                break
        else:
            print("\nno OOM within the sweep — extend WIDTHS or raise --batch/--seq-len.")
    else:
        print("\nCPU-only: PREDICTION MODE (no allocations are attempted; every")
        print("number below is the ch04 ledger + ch03 activation lower bound).\n")
        print(f"B={args.batch}, T={args.seq_len}, L=8, fp32 Adam\n")
        print(f"{'d_model':>8}{'params':>14}{'static GB':>11}{'+acts GB':>10}   "
              + "".join(f"{g:>11}" for g in COMMON_GPUS))
        est = {g: None for g in COMMON_GPUS}
        for d in WIDTHS:
            cfg = make_cfg(d, args.seq_len)
            n, static_gb, act_gb = predict_gb(cfg, args.batch)
            total = static_gb + act_gb
            row = f"{d:>8}{n:>14,}{static_gb:>11.2f}{total:>10.2f}   "
            for g, cap in COMMON_GPUS.items():
                fits = total < cap * 0.9
                row += f"{'fits' if fits else 'OOM?':>11}"
                if not fits and est[g] is None:
                    est[g] = d
            print(row)
        print("\nestimated OOM points (first width predicted NOT to fit, 10% headroom):")
        for g, d in est.items():
            print(f"  {g:<11}: d_model ~ {d if d else '> ' + str(WIDTHS[-1])}")
        print("\nthese are ESTIMATES from formulas — run this script on a real GPU")
        print("(Kaggle/Colab T4) to watch prediction meet the allocator.")


if __name__ == "__main__":
    main()
