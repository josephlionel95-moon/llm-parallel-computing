"""Export a torch.profiler chrome trace of steady-state MiniGPT training steps.

Uses schedule(wait=1, warmup=1, active=2) so step 0 — lazy Adam init, cuDNN
autotuning, allocator warmup — is excluded: we profile the steady state, not
the startup transient. Open the resulting JSON at chrome://tracing or
https://ui.perfetto.dev.

Run:  python experiments/ch04/profiler_trace.py [--out profiling/ch04_train_step.json]
CPU fallback: CPU-only activities; the trace still shows operator timing and
the autograd structure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from llmdist.models.mini_gpt import GPTConfig, MiniGPT
from llmdist.utils.env_check import detect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parents[2]
                                / "profiling" / "ch04_train_step.json"))
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    env = detect()
    device = torch.device("cuda:0" if env.num_gpus >= 1 else "cpu")
    print(env.banner())
    print(f"\ndevice: {device}\n")

    cfg = GPTConfig(vocab_size=8192, block_size=128, n_layer=4, n_head=4, d_model=256)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch, cfg.block_size), device=device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities,
                 schedule=schedule(wait=1, warmup=1, active=2),
                 profile_memory=True, record_shapes=True) as prof:
        for _ in range(4):
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            prof.step()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"\ntrace written to {out}")
    print("open it at chrome://tracing or https://ui.perfetto.dev")
    print("reading order: (1) find optimizer.step on the CPU row to bracket a step;")
    print("(2) check the GPU stream for gaps (launch-bound) vs dense kernels")
    print("(compute-bound); (3) matmuls (aten::mm/addmm/bmm) should dominate.")


if __name__ == "__main__":
    main()
