"""Compare training runs: single GPU vs any distributed strategy.

Reads every RunRecord JSON in experiments/baseline/results/ and prints a
comparison table plus speedup / scaling-efficiency numbers against the
single-device baseline. Optionally plots throughput, memory, step-time
breakdown, and loss curves side by side.

    python experiments/baseline/compare_runs.py
    python experiments/baseline/compare_runs.py --save-plots
    python experiments/baseline/compare_runs.py --dir my_results/

Definitions (derived in ch05/ch25):
  speedup(S)            = tokens_per_s(S) / tokens_per_s(baseline)
  scaling efficiency    = speedup / world_size          (1.0 = perfect)
  comm overhead         = phase_ms["comm"] / step_ms    (0 for baseline)
The gap between efficiency and 1.0 is Amdahl's serial fraction — mostly the
communication slice you can see directly in the phase breakdown.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from llmdist.utils import metrics  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "experiments", "baseline", "results")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR, help="directory of RunRecord JSONs")
    ap.add_argument("--baseline", default=None,
                    help="run_name to use as baseline (default: first strategy=='single')")
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def pick_baseline(records: list[metrics.RunRecord],
                  name: str | None) -> metrics.RunRecord | None:
    if name is not None:
        for r in records:
            if r.run_name == name:
                return r
        sys.exit(f"baseline run '{name}' not found in records")
    singles = [r for r in records if r.strategy == "single"]
    return singles[0] if singles else None


def fmt_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cells, widths))


def main() -> None:
    args = parse_args()
    records = metrics.load_all(args.dir)
    if not records:
        sys.exit(f"no RunRecord JSONs in {args.dir} — run a trainer first, e.g.\n"
                 "  python experiments/baseline/train_single_gpu.py")

    base = pick_baseline(records, args.baseline)
    if base is None:
        print("warning: no single-device baseline found; speedups omitted.\n")

    # Sanity: comparisons are only fair at equal global batch and model size.
    if base is not None:
        for r in records:
            if (r.global_batch, r.n_params) != (base.global_batch, base.n_params):
                print(f"WARNING: '{r.run_name}' has global_batch={r.global_batch}, "
                      f"params={r.n_params:,} but baseline has "
                      f"{base.global_batch}/{base.n_params:,} — speedup not meaningful.")

    header = ["run", "strategy", "world", "device", "tok/s", "step ms",
              "comm %", "peak MB", "loss", "speedup", "efficiency"]
    rows: list[list[str]] = []
    for r in records:
        comm_pct = 100 * r.phase_ms.get("comm", 0.0) / r.step_ms if r.step_ms else 0.0
        speedup = r.tokens_per_s / base.tokens_per_s if base else float("nan")
        eff = speedup / r.world_size if base else float("nan")
        rows.append([r.run_name, r.strategy, str(r.world_size), r.device,
                     f"{r.tokens_per_s:,.0f}", f"{r.step_ms:.1f}",
                     f"{comm_pct:.1f}", f"{r.peak_mem_mb:.0f}",
                     f"{r.final_loss:.4f}",
                     f"{speedup:.2f}x" if base else "-",
                     f"{100 * eff:.0f}%" if base else "-"])

    try:
        from rich.console import Console
        from rich.table import Table
        table = Table(title=f"Training run comparison ({len(records)} runs)")
        for h in header:
            table.add_column(h)
        for row in rows:
            table.add_row(*row)
        Console().print(table)
    except ImportError:
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        print(fmt_row(header, widths))
        print(fmt_row(["-" * w for w in widths], widths))
        for row in rows:
            print(fmt_row(row, widths))

    if base is not None and len(records) > 1:
        print("\nReading the table:")
        print("  * speedup < world_size is EXPECTED: the comm % column shows where it went")
        print("  * peak MB identical across DP ranks = full replica each -> ZeRO/FSDP (ch15/16)")
        print("  * loss curves should nearly match at equal global batch (ch08 caveats)")

    if args.save_plots:
        import matplotlib.pyplot as plt
        names = [r.run_name for r in records]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].bar(names, [r.tokens_per_s for r in records], color="#4c72b0")
        axes[0].set_title("throughput (tokens/s)")
        axes[1].bar(names, [r.peak_mem_mb for r in records], color="#dd8452")
        axes[1].set_title("peak memory rank0 (MB)")
        bottom = [0.0] * len(records)
        for key, color in [("forward", "#4c72b0"), ("backward", "#55a868"),
                           ("comm", "#dd8452"), ("optimizer", "#c44e52")]:
            vals = [r.phase_ms.get(key, 0.0) for r in records]
            axes[2].bar(names, vals, bottom=bottom, label=key, color=color)
            bottom = [b + v for b, v in zip(bottom, vals)]
        axes[2].set_title("step time breakdown (ms)")
        axes[2].legend(fontsize=8)
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        out = os.path.join(ROOT, "visualizations", "baseline_comparison.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=120)
        print(f"\nsaved {out}")

        fig2, ax = plt.subplots(figsize=(7, 4))
        for r in records:
            if r.losses:
                ax.plot(r.losses, label=f"{r.run_name} (w={r.world_size})", alpha=0.8)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.set_title("loss curves (should overlap at equal global batch)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out2 = os.path.join(ROOT, "visualizations", "baseline_loss_curves.png")
        fig2.tight_layout()
        fig2.savefig(out2, dpi=120)
        print(f"saved {out2}")


if __name__ == "__main__":
    main()
