#!/usr/bin/env bash
# Serial, finite simulation experiments for the 8 GB training GPU.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: PYTHON=/path/to/python $0 CHECKPOINT NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT=$(realpath -- "$1")
OUTPUT=$(realpath -m -- "$2")
mkdir -- "$OUTPUT"
for cost in 1 5; do
    name="demand${cost}_seed42"
    if bash "$SCRIPTS/run_hd1910_smooth_pilot.sh" \
        "$CHECKPOINT" "$OUTPUT/$name" 5 42 1024 "$cost" \
        > "$OUTPUT/$name.launcher.log" 2>&1; then
        printf '0\n' > "$OUTPUT/$name.exit"
    else
        code=$?
        printf '%s\n' "$code" > "$OUTPUT/$name.exit"
        exit "$code"
    fi
done
echo "Both simulation pilots finished; inspect motion quality before qualification."
