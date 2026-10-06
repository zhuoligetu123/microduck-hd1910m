#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
test -x "$ROOT/out/arm64/bin/robotd"
mkdir -p "$ROOT/release-assets"
tar -C "$ROOT" --exclude=__pycache__ --exclude=test_luwu_policy.py -czf \
  "$ROOT/release-assets/microduck-hd1910m-arm64.tar.gz" \
  out/arm64 scripts radxa README.md THIRD_PARTY.md microduck/LICENSE microduck_rl/LICENSE
echo "$ROOT/release-assets/microduck-hd1910m-arm64.tar.gz"
