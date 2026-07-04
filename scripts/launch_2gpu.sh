#!/usr/bin/env bash
# Launch any experiment script on all visible GPUs (falls back to 2 CPU processes).
# Usage: bash scripts/launch_2gpu.sh experiments/ch08/manual_dp_train.py [args...]
set -e
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
NPROC=${NGPU:-0}; [ "$NPROC" -lt 1 ] && NPROC=2
echo "Launching $1 with --nproc_per_node=$NPROC"
torchrun --standalone --nproc_per_node=$NPROC "$@"
