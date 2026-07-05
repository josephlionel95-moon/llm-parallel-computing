# Baseline: single-GPU training + comparison harness

The reference point for every distributed strategy in this course. All trainers
write the same `RunRecord` JSON (see `src/llmdist/utils/metrics.py`) into
`experiments/baseline/results/`; `compare_runs.py` turns any set of records into
a comparison table with speedup and scaling efficiency.

## Workflow (Kaggle 2x T4 example)

```bash
# 1. The baseline: single GPU, global batch 16
python experiments/baseline/train_single_gpu.py --global-batch 16 --steps 60

# 2. Manual data parallelism: 2 GPUs, per-rank batch 8 = same global batch
torchrun --nproc_per_node=2 experiments/ch08/manual_dp_train.py \
    --bsz 8 --steps 60 --metrics

# 3. Compare (table + optional plots)
python experiments/baseline/compare_runs.py --save-plots
```

What to look for:

- **speedup < 2x** — the `comm %` column shows exactly where the missing time went
  (naive DP serializes the gradient all-reduce after backward; ch09's DDP overlaps it).
- **identical peak MB on every rank** — data parallelism replicates everything;
  the optimizer states dominate. ZeRO (ch15) and FSDP (ch16) attack this number.
- **near-identical loss curves** — at equal global batch, DP is mathematically the
  same update as the baseline (derived in ch08); differences are reduction order.

Both scripts run on CPU too (Gloo / no GPU): the numbers change, the lessons don't.
Flags: `--amp` (bf16/fp16), `--grad-accum N`, `--n-layer/--d-model/--block-size`
to scale the model, `--run-name` to keep multiple records.
