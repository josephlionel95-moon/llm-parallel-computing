"""A single JSON schema for every training run in the course.

Why: the whole point of the course is *comparison* — single GPU vs DP vs DDP
vs FSDP vs ... . Comparisons only work if every run records the same fields
the same way. Every trainer (baseline and distributed) writes one RunRecord;
`experiments/baseline/compare_runs.py` reads any number of them and produces
the comparison table, speedups, and scaling efficiencies.

Schema (all sizes in the units stated by the field name):
    run_name          e.g. "single_gpu", "manual_dp_w2", "ddp_w2"
    strategy          "single" | "manual_dp" | "ddp" | "fsdp" | ...
    world_size        number of processes
    device            "cuda" | "cpu"
    backend           "nccl" | "gloo" | "none"
    n_params          model parameter count
    global_batch      sequences per optimizer step, summed over ranks
    seq_len           tokens per sequence
    steps             optimizer steps measured
    elapsed_s         wall time for the measured steps
    tokens_per_s      global training throughput
    step_ms           mean ms per optimizer step
    phase_ms          {"forward":..,"backward":..,"comm":..,"optimizer":..} mean ms/step
    peak_mem_mb       max CUDA memory allocated on rank 0 (0.0 on CPU)
    final_loss        mean loss over the last 10% of steps
    losses            full loss curve (list of float)
    extra             free-form dict for strategy-specific fields
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunRecord:
    run_name: str
    strategy: str
    world_size: int
    device: str
    backend: str
    n_params: int
    global_batch: int
    seq_len: int
    steps: int
    elapsed_s: float
    tokens_per_s: float
    step_ms: float
    phase_ms: dict[str, float] = field(default_factory=dict)
    peak_mem_mb: float = 0.0
    final_loss: float = 0.0
    losses: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def save(record: RunRecord, out_dir: str) -> str:
    """Write `<out_dir>/<run_name>.json`; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record.run_name}.json")
    with open(path, "w") as f:
        json.dump(asdict(record), f, indent=2)
    return path


def load_all(dir_path: str) -> list[RunRecord]:
    """Load every *.json RunRecord in a directory, sorted by (strategy, world_size)."""
    records: list[RunRecord] = []
    if not os.path.isdir(dir_path):
        return records
    for name in sorted(os.listdir(dir_path)):
        if name.endswith(".json"):
            with open(os.path.join(dir_path, name)) as f:
                d = json.load(f)
            known = {k: d[k] for k in RunRecord.__dataclass_fields__ if k in d}
            records.append(RunRecord(**known))
    records.sort(key=lambda r: (r.strategy, r.world_size))
    return records


def final_loss(losses: list[float]) -> float:
    """Mean of the last 10% of the loss curve (>=1 point)."""
    if not losses:
        return float("nan")
    k = max(1, len(losses) // 10)
    return sum(losses[-k:]) / k
