#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
TASK="${TASK:-Mjlab-SwayInPlace-Flat-MicroDuck}"
RUN_NAME=head_up_sway
OUTPUT=policies/sway_in_place/policy.onnx
if [[ "$TASK" == Mjlab-SwayHead-Flat-MicroDuck ]]; then
    RUN_NAME=wide_sway_head
    OUTPUT=policies/sway_head/policy.onnx
fi
case "${1:-smoke}" in
    smoke)
        exec .venv/bin/train "$TASK" --env.scene.num-envs 64 --agent.max-iterations 5 \
            --agent.run-name sway_smoke --agent.save-interval 5 --enable-nan-guard True ;;
    train)
        exec .venv/bin/train "$TASK" --env.scene.num-envs "${3:-1024}" \
            --agent.max-iterations "${2:-1500}" --agent.run-name "$RUN_NAME" \
            --agent.save-interval 100 --enable-nan-guard True ;;
    resume)
        [[ $# -ge 3 ]] || { printf '%s\n' 'resume RUN MODEL [iterations] [num_envs]'; exit 2; }
        exec .venv/bin/train "$TASK" --env.scene.num-envs "${5:-1024}" \
            --agent.resume True --agent.load-run "$2" --agent.load-checkpoint "$3" \
            --agent.max-iterations "${4:-500}" --agent.run-name "$RUN_NAME" \
            --agent.save-interval 100 --enable-nan-guard True ;;
    export)
        [[ $# -ge 2 ]] || { printf '%s\n' 'export CHECKPOINT [output.onnx]'; exit 2; }
        exec .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$2" \
            --onnx-file "${3:-$OUTPUT}" --num-envs 1 ;;
    *) printf '%s\n' 'Usage: train_sway.sh smoke | train [iterations] [num_envs] | resume RUN MODEL [iterations] [num_envs] | export CHECKPOINT [output.onnx]'; exit 2 ;;
esac
