#!/usr/bin/env bash
# Existing container only; a frozen source tree and a new output per experiment.
set -euo pipefail
BASE=$1
ROOT=$2
NAME=$3
SEED=$4
ENVS=$5
HOLD=$6
export PYTHONPATH="$ROOT/source/src"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [[ ${USE_EXISTING_MPS:-0} == 1 ]]; then
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/hd1910-mps/pipe
    export CUDA_MPS_LOG_DIRECTORY=/tmp/hd1910-mps/log
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
fi
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/$NAME.exit"' EXIT
"$BASE/.venv-container/bin/python" "$ROOT/source/scripts/iterate_hd1910.py" \
    --kind velocity --checkpoint "$ROOT/parent/model_1199.pt" \
    --output "$ROOT/runs/$NAME" --seed "$SEED" --num-envs "$ENVS" \
    --iterations 600 --action-rate-cost 5 --slew-demand-cost 20 \
    --yaw-square-weight 0.25 --motor-delay-min-steps 3 --delay-hold-steps "$HOLD"
