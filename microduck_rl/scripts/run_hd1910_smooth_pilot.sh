#!/usr/bin/env bash
# Isolated simulation pilot. Never connects to or deploys onto a robot.
set -euo pipefail
if [[ $# -lt 5 || $# -gt 7 ]]; then
    echo "Usage: PYTHON=/path/to/python $0 CHECKPOINT OUTPUT ACTION_RATE_COST SEED NUM_ENVS [SLEW_DEMAND_COST] [ITERATIONS]" >&2
    exit 2
fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv-container/bin/python"}
CHECKPOINT=$(realpath -- "$1")
OUTPUT=$(realpath -m -- "$2")
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$PYTHON" "$ROOT/scripts/iterate_hd1910.py" \
    --kind velocity --checkpoint "$CHECKPOINT" --output "$OUTPUT" \
    --action-rate-cost "$3" --seed "$4" --num-envs "$5" \
    --slew-demand-cost "${6:-0}" \
    --yaw-square-weight 0.25 --iterations "${7:-300}" --pilot

# Failed nominal screening is still useful: measure stress, but never promote it.
POLICY=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy"])' "$OUTPUT/manifest.json")
for engine in cpu warp; do
    SCRIPT=replay_hd1910.py
    [[ $engine != warp ]] || SCRIPT=replay_hd1910_warp.py
    "$PYTHON" "$ROOT/scripts/$SCRIPT" --policy "$POLICY" \
        --seconds 20 --extended --seed 42 --voltage 8.4 --delay-steps 6 \
        --initial-tilt-deg 5 --report "$OUTPUT/diagnostic_${engine}_v8.4_delay6.json" \
        > "$OUTPUT/diagnostic_${engine}_v8.4_delay6.log" 2>&1
done
echo "Pilot and diagnostic replay finished; NOT a hardware deployment approval."
