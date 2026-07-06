"""Monte-Carlo check of Young's optimal checkpoint interval τ* = sqrt(2δM).

    python experiments/ch19/optimal_interval.py
    python experiments/ch19/optimal_interval.py --delta 30 --mtbf 14400 --save-plots

Simulates thousands of training days on flaky hardware: work proceeds,
a checkpoint (cost δ) is taken every τ seconds, failures arrive as a
Poisson process (MTBF M), each failure rolls progress back to the last
completed checkpoint and pays a restart cost R. For a sweep of τ we
measure the fraction of wall-clock time that produced *retained* useful
work, and compare the empirical best τ with Young's closed form.

No torch needed — this is pure failure arithmetic. Defaults model a free
Colab session: δ = 30 s (save + Drive sync), M = 4 h, R = 120 s (VM boot,
pip install, checkpoint download).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from llmdist.checkpoint import young_interval  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=30.0, help="checkpoint cost (s)")
    ap.add_argument("--mtbf", type=float, default=4 * 3600, help="mean time between failures (s)")
    ap.add_argument("--restart", type=float, default=120.0, help="restart cost (s)")
    ap.add_argument("--horizon", type=float, default=100 * 3600, help="simulated wall-clock (s)")
    ap.add_argument("--trials", type=int, default=40, help="runs per τ (variance!)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def simulate(tau: float, delta: float, mtbf: float, restart: float,
             horizon: float, rng: random.Random) -> float:
    """One simulated run; returns retained useful work / wall-clock.

    Event loop over checkpoint cycles. Within one cycle we do τ seconds of
    work then δ of saving; a Poisson failure (exponential inter-arrival)
    can strike anywhere; work since the last completed save is lost.
    """
    t = 0.0                 # wall clock
    useful = 0.0            # retained work (survived a completed checkpoint)
    pending = 0.0           # work done since last completed checkpoint
    next_fail = rng.expovariate(1.0 / mtbf)
    while t < horizon:
        cycle_end = t + (tau - pending) + delta   # finish work quantum + save
        if next_fail < cycle_end:                 # failure strikes this cycle
            work_done = max(0.0, min(next_fail - t, tau - pending))
            _ = work_done                         # ...and is lost (not retained)
            t = next_fail + restart
            pending = 0.0
            next_fail = t + rng.expovariate(1.0 / mtbf)
        else:                                     # cycle completes
            useful += tau
            pending = 0.0
            t = cycle_end
    return useful / horizon


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tau_star = young_interval(args.delta, args.mtbf)

    taus = sorted(set(
        [tau_star * f for f in (0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.5, 6.0, 12.0)]))
    print(f"δ = {args.delta:.0f} s   M = {args.mtbf / 3600:.1f} h   "
          f"R = {args.restart:.0f} s   Young τ* = {tau_star:.0f} s "
          f"({tau_star / 60:.1f} min)\n")
    hdr = f"{'τ (s)':>8} {'τ/τ*':>6} {'efficiency':>10} {'overhead':>9}"
    print(hdr + "\n" + "-" * len(hdr))

    results = []
    for tau in taus:
        eff = sum(simulate(tau, args.delta, args.mtbf, args.restart,
                           args.horizon, rng)
                  for _ in range(args.trials)) / args.trials
        results.append((tau, eff))
        print(f"{tau:>8.0f} {tau / tau_star:>6.2f} {eff:>10.4f} {1 - eff:>9.4f}")

    best_tau, best_eff = max(results, key=lambda r: r[1])
    _, eff_at_star = min(results, key=lambda r: abs(r[0] - tau_star))
    predicted_overhead = math.sqrt(2 * args.delta / args.mtbf)
    print(f"\nempirical best  : τ = {best_tau:.0f} s  (τ* = {tau_star:.0f} s)")
    print(f"overhead at τ*  : measured {1 - eff_at_star:.3f}"
          f"  vs theory √(2δ/M) = {predicted_overhead:.3f} (+R/M term)")
    print("U-curve: left wall = δ/τ checkpoint tax, right wall = τ/2M rework.")

    if args.save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs, ys = zip(*results)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([x / 60 for x in xs], [1 - y for y in ys], "o-", label="simulated overhead")
        ax.axvline(tau_star / 60, ls="--", c="tab:red", label=f"Young τ* = {tau_star / 60:.1f} min")
        ax.set_xscale("log")
        ax.set_xlabel("checkpoint interval τ (min)")
        ax.set_ylabel("overhead fraction (1 − efficiency)")
        ax.set_title(f"Young–Daly in practice: δ={args.delta:.0f}s, M={args.mtbf / 3600:.1f}h")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(ROOT, "visualizations", "ch19_optimal_interval.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
