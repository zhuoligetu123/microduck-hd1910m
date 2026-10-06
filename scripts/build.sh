#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
MODE="${1:-native}"
case "$MODE" in
  native) ARGS=(build --release --locked); SUB=release ;;
  arm64)
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
    export CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++
    ARGS=(build --release --locked --target aarch64-unknown-linux-gnu)
    SUB=aarch64-unknown-linux-gnu/release ;;
  *) printf 'Usage: bash scripts/build.sh [native|arm64]\n' >&2; exit 2 ;;
esac
(cd "$ROOT/microduck" && cargo "${ARGS[@]}" -p robotd -p robotctl)
(cd "$ROOT/microduck_app/backend" && cargo "${ARGS[@]}")
mkdir -p "$ROOT/out/$MODE/bin"
install -m755 "$ROOT/microduck/target/$SUB/robotd" "$ROOT/microduck/target/$SUB/robotctl" "$ROOT/out/$MODE/bin/"
install -m755 "$ROOT/microduck_app/backend/target/$SUB/microduck-app-server" "$ROOT/out/$MODE/bin/"
printf 'Built %s\n' "$ROOT/out/$MODE/bin"
