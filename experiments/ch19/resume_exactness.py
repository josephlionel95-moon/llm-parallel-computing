"""The bitwise resume test: kill training halfway, resume, demand equality.

    python experiments/ch19/resume_exactness.py
    python experiments/ch19/resume_exactness.py --steps 100 --kill-at 50

Run A trains `steps` steps uninterrupted. Run B trains to `kill-at`, is
"killed" (model, optimizer, scheduler, RNG all destroyed and rebuilt from
scratch), resumes from the checkpoint, and finishes. If — and only if — the
checkpoint captured ALL seven pieces of training state, the post-resume
losses equal run A's losses BITWISE (`==`, not allclose). Dropout is on so
the RNG stream matters; the scheduler decays so its state matters; data is
drawn from the global torch RNG so the data stream matters.

This script is the correctness oracle for src/llmdist/checkpoint.py. Break
something on purpose (--sabotage {optimizer,rng,scheduler}) to see the
loss-curve signature each omission produces.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from llmdist.checkpoint import (  # noqa: E402
    latest_checkpoint, load_checkpoint, save_checkpoint, training_state_dict)
from llmdist.models.mini_gpt import GPTConfig, MiniGPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--kill-at", type=int, default=30)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sabotage", choices=["optimizer", "rng", "scheduler"],
                    default=None, help="deliberately omit one piece of state")
    return ap.parse_args()


def build(seed: int | None = None
          ) -> tuple[MiniGPT, torch.optim.Adam, torch.optim.lr_scheduler.LambdaLR]:
    """Fresh model/optimizer/scheduler — called again after the 'kill' to
    prove that nothing survives in memory: resume must come from the file."""
    if seed is not None:
        torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2,
                    d_model=64, dropout=0.1)  # dropout ON: RNG must round-trip
    model = MiniGPT(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(  # warmup 10, then 1/sqrt decay
        opt, lambda s: min((s + 1) / 10, (10 / max(s, 10)) ** 0.5))
    return model, opt, sched


def train_step(model: MiniGPT, opt: torch.optim.Optimizer,
               sched: torch.optim.lr_scheduler.LRScheduler, bsz: int) -> float:
    """One step on synthetic data drawn from the GLOBAL torch RNG.

    Drawing data from the global stream is deliberate: it makes the data
    sequence part of the RNG state, so restoring RNG also restores the data
    position (piece #6) in this synthetic setting.
    """
    idx = torch.randint(0, model.cfg.vocab_size, (bsz, model.cfg.block_size + 1))
    x, y = idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
    _, loss = model(x, targets=y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    return loss.item()


def main() -> None:
    args = parse_args()
    torch.use_deterministic_algorithms(True)
    workdir = tempfile.mkdtemp(prefix="ch19_resume_")
    ckpt_dir = os.path.join(workdir, "ckpts")

    # ---- Run A: uninterrupted ------------------------------------------
    model, opt, sched = build(args.seed)
    losses_a = [train_step(model, opt, sched, args.bsz)
                for _ in range(args.steps)]

    # ---- Run B, phase 1: train to kill-at, checkpoint, "die" ------------
    model, opt, sched = build(args.seed)
    losses_b = [train_step(model, opt, sched, args.bsz)
                for _ in range(args.kill_at)]
    state = training_state_dict(model, opt, sched, step=args.kill_at)
    if args.sabotage == "optimizer":
        del state["optimizer"]
    elif args.sabotage == "rng":
        del state["rng"]
    elif args.sabotage == "scheduler":
        del state["scheduler"]
    save_checkpoint(os.path.join(ckpt_dir, f"step_{args.kill_at:08d}.pt"), state)
    del model, opt, sched, state                      # the "crash"

    # ---- Run B, phase 2: fresh process resumes from the file ------------
    model, opt, sched = build(seed=args.seed + 999)   # WRONG seed on purpose:
    # if resume is correct, everything below comes from the checkpoint and
    # this seed must not matter at all.
    ckpt = latest_checkpoint(ckpt_dir)
    assert ckpt is not None
    loaded = load_checkpoint(ckpt, model, opt, sched)
    start = loaded["step"]
    losses_b += [train_step(model, opt, sched, args.bsz)
                 for _ in range(args.steps - start)]

    # ---- Verdict: bitwise, not allclose ---------------------------------
    diffs = [abs(a - b) for a, b in zip(losses_a, losses_b)]
    exact = [a == b for a, b in zip(losses_a, losses_b)]
    print(f"{'step':>4}  {'run A':>10}  {'run B':>10}  {'bitwise':>7}")
    show = list(range(0, args.steps, max(1, args.steps // 12)))
    show += [args.kill_at - 1, args.kill_at, args.steps - 1]
    for i in sorted(set(s for s in show if 0 <= s < args.steps)):
        mark = "==" if exact[i] else "≠≠"
        resumed = "  <- resume" if i == args.kill_at else ""
        print(f"{i:>4}  {losses_a[i]:>10.6f}  {losses_b[i]:>10.6f}  {mark:>7}{resumed}")

    n_exact = sum(exact)
    print(f"\nbitwise-equal steps : {n_exact}/{args.steps}")
    print(f"max |Δloss|         : {max(diffs):.3e}")
    shutil.rmtree(workdir)
    if args.sabotage:
        assert n_exact < args.steps, "sabotage should have broken equality!"
        print(f"sabotage '{args.sabotage}' broke the resume, as predicted. "
              "Look at WHERE the curves split — that is the diagnostic signature.")
    else:
        assert n_exact == args.steps, "resume is NOT exact — a piece of state leaked!"
        print("resume is bitwise exact: the checkpoint captured the whole machine. ✓")


if __name__ == "__main__":
    main()
