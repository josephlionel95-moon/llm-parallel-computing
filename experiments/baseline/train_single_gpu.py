"""Single-GPU (or CPU) MiniGPT training — THE reference baseline.

Every distributed strategy in this course is judged against this script:
same model, same synthetic corpus, same measurement points. Run it first,
then run the distributed trainers with the SAME --global-batch, and compare
with experiments/baseline/compare_runs.py.

    python experiments/baseline/train_single_gpu.py                  # defaults
    python experiments/baseline/train_single_gpu.py --global-batch 16 --steps 60
    python experiments/baseline/train_single_gpu.py --amp --save-plots

Fair-comparison rules baked in:
  * global batch is what matters: a 2-GPU DP run with per-rank batch 8 must be
    compared against this baseline at --global-batch 16 (weak scaling compares
    per-device batch instead — see ch25);
  * identical seed and data order -> the loss curve should closely track a DP
    run with the same global batch (ch08 explains the reduction-order caveats);
  * timing excludes warmup steps (allocator + cuDNN autotune noise);
  * peak memory is reset after warmup and read at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402
from llmdist.utils import metrics  # noqa: E402
from llmdist.workers.ch08 import make_batch, synthetic_tokens  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "experiments", "baseline", "results")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=60, help="measured optimizer steps")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warmup steps")
    ap.add_argument("--global-batch", type=int, default=16,
                    help="sequences per step; match the distributed run's world*bsz")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per step (global-batch must divide evenly)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (bf16 if supported, else fp16+GradScaler)")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--run-name", default="single_gpu")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    assert args.global_batch % args.grad_accum == 0, "grad-accum must divide global-batch"
    micro_bsz = args.global_batch // args.grad_accum

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig(vocab_size=256, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, d_model=args.d_model)

    # Same seed as the DP trainers -> identical initial weights (ch08 relies on this).
    torch.manual_seed(0)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP setup: bf16 needs no loss scaling (same exponent range as fp32, ch20);
    # fp16 does (gradients underflow below 2^-24, ch04).
    use_bf16 = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = args.amp and device.type == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    stream = synthetic_tokens()  # same deterministic corpus as ch08/ch09 trainers
    n_windows = len(stream) - cfg.block_size - 1
    n_params = model.num_params()

    print(f"device={device} amp={'bf16' if use_bf16 else 'fp16' if use_fp16 else 'off'} "
          f"params={n_params:,} global_batch={args.global_batch} "
          f"(micro={micro_bsz} x accum={args.grad_accum})")

    phase = {"forward": 0.0, "backward": 0.0, "comm": 0.0, "optimizer": 0.0}
    timing_on = False

    def _t(key: str, t_start: float) -> None:
        if timing_on:
            if device.type == "cuda":
                torch.cuda.synchronize()  # fair phase timing (ch01: async launches)
            phase[key] += time.perf_counter() - t_start

    def get_batch(step: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic shuffle keyed by step: same schedule a DP run sees as the
        # union of its rank shards, so data order matches at equal global batch.
        g = torch.Generator().manual_seed(step)
        idx = torch.randperm(n_windows, generator=g)[:args.global_batch]
        sl = idx[micro * micro_bsz:(micro + 1) * micro_bsz]
        x, y = make_batch(stream, sl, cfg.block_size)
        return x.to(device), y.to(device)

    def train_step(step: int) -> float:
        """One optimizer step = grad-accum micro-steps. Returns mean loss."""
        total = 0.0
        for m in range(args.grad_accum):
            x, y = get_batch(step, m)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_bf16 or use_fp16):
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            _t("forward", t0)
            t1 = time.perf_counter()
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            _t("backward", t1)
            total += loss.item() * args.grad_accum
        t2 = time.perf_counter()
        if use_fp16:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        _t("optimizer", t2)
        return total / args.grad_accum

    # ---- warmup (untimed): allocator growth, autotune, lazy Adam state ----
    for step in range(args.warmup):
        train_step(step)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ---- measured steps --------------------------------------------------
    timing_on = True
    losses: list[float] = []
    t_start = time.perf_counter()
    for step in range(args.warmup, args.warmup + args.steps):
        losses.append(train_step(step))
        if (step - args.warmup) % max(1, args.steps // 10) == 0:
            print(f"step {step - args.warmup:>4}  loss {losses[-1]:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    # ---- report + save record --------------------------------------------
    tokens = args.steps * args.global_batch * cfg.block_size
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20
               if device.type == "cuda" else 0.0)
    rec = metrics.RunRecord(
        run_name=args.run_name,
        strategy="single",
        world_size=1,
        device=device.type,
        backend="none",
        n_params=n_params,
        global_batch=args.global_batch,
        seq_len=cfg.block_size,
        steps=args.steps,
        elapsed_s=elapsed,
        tokens_per_s=tokens / elapsed,
        step_ms=1e3 * elapsed / args.steps,
        phase_ms={k: 1e3 * v / args.steps for k, v in phase.items()},
        peak_mem_mb=peak_mb,
        final_loss=metrics.final_loss(losses),
        losses=losses,
        extra={"amp": "bf16" if use_bf16 else "fp16" if use_fp16 else "off",
               "grad_accum": args.grad_accum},
    )
    path = metrics.save(rec, args.out_dir)

    print(f"\nthroughput : {rec.tokens_per_s:,.0f} tokens/s")
    print(f"step time  : {rec.step_ms:.1f} ms "
          f"(fwd {rec.phase_ms['forward']:.1f} / bwd {rec.phase_ms['backward']:.1f} "
          f"/ opt {rec.phase_ms['optimizer']:.1f} — comm 0.0, that's the point)")
    print(f"peak memory: {peak_mb:.1f} MB" if peak_mb else "peak memory: n/a (CPU)")
    print(f"final loss : {rec.final_loss:.4f}")
    print(f"record     : {path}")
    print("\nNext: run a distributed trainer at the same global batch, e.g.\n"
          f"  torchrun --nproc_per_node=2 experiments/ch08/manual_dp_train.py "
          f"--bsz {args.global_batch // 2} --steps {args.steps} --metrics\n"
          "then: python experiments/baseline/compare_runs.py")

    if args.save_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(losses)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.set_title(f"Single-device baseline (global batch {args.global_batch})")
        ax.grid(alpha=0.3)
        out = os.path.join(ROOT, "visualizations", "baseline_single_gpu_loss.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
