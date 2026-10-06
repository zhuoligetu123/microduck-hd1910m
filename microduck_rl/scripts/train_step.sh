#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
TASK=Mjlab-StepInPlace-Flat-MicroDuck
case "${1:-smoke}" in
    smoke)
        exec .venv/bin/train "$TASK" --env.scene.num-envs 64 --agent.max-iterations 5 \
            --agent.run-name step_smoke --agent.save-interval 5 --enable-nan-guard True
        ;;
    train)
        exec .venv/bin/train "$TASK" --env.scene.num-envs "${3:-1024}" \
            --agent.max-iterations "${2:-3000}" --agent.run-name step_phase50hz \
            --agent.save-interval 100 --enable-nan-guard True
        ;;
    resume)
        [[ $# -ge 3 ]] || { printf '%s\n' 'Usage: train_step.sh resume RUN MODEL [additional_iterations] [num_envs]'; exit 2; }
        exec .venv/bin/train "$TASK" --env.scene.num-envs "${5:-1024}" \
            --agent.resume True --agent.load-run "$2" --agent.load-checkpoint "$3" \
            --agent.max-iterations "${4:-300}" --agent.run-name step_head_up \
            --agent.save-interval 100 --enable-nan-guard True
        ;;
    export)
        [[ $# == 2 ]] || { printf '%s\n' 'Usage: train_step.sh export /absolute/path/model_N.pt'; exit 2; }
        mkdir -p policies/step_in_place
        exec .venv/bin/python scripts/export.py "$TASK" --checkpoint-file "$2" \
            --onnx-file policies/step_in_place/policy.onnx --num-envs 1
        ;;
    *) printf '%s\n' 'Usage: train_step.sh smoke | train [iterations] [num_envs] | resume RUN MODEL [iterations] [num_envs] | export CHECKPOINT'; exit 2 ;;
esac
