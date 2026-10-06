#!/usr/bin/env bash
# Run inside the existing training container; output paths must be new.
set -euo pipefail
BASE=$1
NAME=$2
SEED=$3
ENVS=$4
MIN_DELAY=$5
ROOT="$BASE/training_runs/delay_followup_20260929"
export PYTHONPATH="$ROOT/source/src"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_MPS_PIPE_DIRECTORY=/tmp/hd1910-mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/hd1910-mps/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
set +e
"$BASE/.venv-container/bin/python" "$ROOT/source/scripts/iterate_hd1910.py" \
  --kind velocity --checkpoint "$ROOT/parent/model_799.pt" \
  --output "$ROOT/runs/$NAME" --seed "$SEED" --num-envs "$ENVS" \
  --iterations 1200 --action-rate-cost 5 --slew-demand-cost 20 \
  --yaw-square-weight 0.25 --motor-delay-min-steps "$MIN_DELAY"
code=$?
printf '%s\n' "$code" > "$ROOT/$NAME.exit"
exit "$code"
