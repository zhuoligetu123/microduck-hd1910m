#!/usr/bin/env bash
# Start one already-created training container after its GPU predecessor exits.
set -euo pipefail
PREVIOUS=${1:?previous container required}
NEXT=${2:?next container required}
test "$PREVIOUS" != "$NEXT"
test "$(docker inspect -f '{{.State.Status}}' "$NEXT")" = created
while test "$(docker inspect -f '{{.State.Running}}' "$PREVIOUS")" = true; do
    sleep 30
done
date -Is
docker start "$NEXT"
