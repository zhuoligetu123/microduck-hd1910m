#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BIN_DIR="${BIN_DIR:-$ROOT/out/arm64/bin}"
PARAMS="${PARAMS:-$ROOT/local/params.toml}"
export ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-$("$PYTHON" -c 'from pathlib import Path; import onnxruntime; print(next((Path(onnxruntime.__file__).parent/"capi").glob("libonnxruntime.so.*")))')}"
test -f "$PARAMS"
test -x "$BIN_DIR/robotd"
mkdir -p "$ROOT/local"
exec 9>"$ROOT/local/runtime.lock"
flock -n 9 || exit 1
export MICRODUCK_ROBOTD_SOCKET="$ROOT/local/robotd.sock"
export MICRODUCK_APP_BIND="${MICRODUCK_APP_BIND:-0.0.0.0:38880}"
export MICRODUCK_APP_ALLOW_OPEN_LAN=1
unset MICRODUCK_APP_TOKEN
pids=()
cleanup() { if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; wait || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT TERM
"$BIN_DIR/robotd" --params "$PARAMS" --socket "$MICRODUCK_ROBOTD_SOCKET" &
pids+=("$!")
"$BIN_DIR/microduck-app-server" &
pids+=("$!")
wait -n "${pids[@]}"
