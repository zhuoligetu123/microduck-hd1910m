#!/usr/bin/env bash
# Frozen source in an existing training container. Never connects to hardware.
set -euo pipefail
BASE=$1
ROOT=$2
NAME=$3
SEED=$4
ENVS=$5
ITERS=${6:-4000}
PY="$BASE/.venv-container/bin/python"
SOURCE="$ROOT/source"
OUT="$ROOT/runs/$NAME"
mkdir -p "$ROOT/runs"
mkdir "$OUT"
export PYTHONPATH="$SOURCE/src" MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MICRODUCK_BAM_PROFILE="$SOURCE/src/mjlab_microduck/actuator/radxa_1910_m6.json"
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/$NAME.exit"' EXIT
cd "$OUT"
printf 'smoke\n' > stage.txt
"$PY" "$SOURCE/scripts/train_hd1910_bam.py" --installation "$ROOT/installation.json" \
    --env.scene.num-envs 64 --agent.max-iterations 5 --agent.seed "$SEED" \
    --agent.run-name smoke --agent.logger tensorboard --agent.upload-model False > smoke.log 2>&1
printf 'training\n' > stage.txt
"$PY" "$SOURCE/scripts/train_hd1910_bam.py" --installation "$ROOT/installation.json" \
    --warm-start-checkpoint "$ROOT/parent/model_1199.pt" \
    --env.scene.num-envs "$ENVS" --agent.max-iterations "$ITERS" --agent.seed "$SEED" \
    --agent.run-name "$NAME" --agent.logger tensorboard --agent.upload-model False > train.log 2>&1
printf 'evaluation\n' > stage.txt
# A unique run-name and output directory prevent accidentally evaluating old models.
mapfile -t POLICIES < <(find "$OUT/logs/rsl_rl/microduck_hd1910_xgobam" -name "*_${NAME}.onnx")
test "${#POLICIES[@]}" -eq 1
POLICY=${POLICIES[0]}
CHECKPOINT="$(dirname "$POLICY")/model_$((ITERS-1)).pt"
"$PY" "$SOURCE/scripts/audit_bounded_policy.py" --policy "$POLICY" \
    --checkpoint "$CHECKPOINT" --report parity.json > parity.log 2>&1
for eval_seed in 42 7 123; do
    for engine in cpu warp; do
        SCRIPT=replay_hd1910.py
        [[ $engine == warp ]] && SCRIPT=replay_hd1910_warp.py
        for condition in nominal low_long high_short; do
            voltage=7.4; delay=4; tilt=0
            [[ $condition != low_long ]] || { delay=6; tilt=5; }
            [[ $condition != high_short ]] || { voltage=8.0; delay=3; tilt=5; }
            report="${engine}_${condition}_${eval_seed}"
            # Failed tracking is a result to preserve, not a reason to skip the matrix.
            set +e
            "$PY" "$SOURCE/scripts/$SCRIPT" --bam-reference --policy "$POLICY" \
                --extended --seconds 20 --seed "$eval_seed" --voltage "$voltage" \
                --delay-steps "$delay" --initial-tilt-deg "$tilt" \
                --report "$report.json" > "$report.log" 2>&1
            status=$?
            set -e
            printf '%s\n' "$status" > "$report.exit"
            test -s "$report.json" # Unexpected crash must not masquerade as evaluation.
        done
    done
done
printf 'evaluated_not_hardware_qualified\n' > stage.txt
