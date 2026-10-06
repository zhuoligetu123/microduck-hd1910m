#!/usr/bin/env bash
set -euo pipefail

# Creates an isolated compute container only. No robot devices or host services.
ROOT=$(realpath "${1:?Usage: bash start_training_container.sh HUGGINGFACE_ROOT [NAME]}")
NAME=${2:-hd1910-training-20260928}
IMAGE=registry.cn-hangzhou.aliyuncs.com/carserver/vlabot:v20260827-ci08-mindon-robotics-amd64-cuda12.8-humble-fastdds2.6
test -f "$ROOT/microduck_rl/uv.lock"
test -x "$ROOT/uv"
if docker container inspect "$NAME" >/dev/null 2>&1; then
    printf 'Container already exists; inspect it instead of replacing: %s\n' "$NAME" >&2
    exit 1
fi
mkdir -p "$ROOT/.container-home" "$ROOT/.cache" "$ROOT/.python" "$ROOT/.tmp"
docker run -d --name "$NAME" --gpus device=0 \
    --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    --shm-size 4g --cpus "${TRAIN_CPUS:-8}" --memory "${TRAIN_MEMORY:-16g}" \
    --mount "type=bind,src=$ROOT,dst=$ROOT" \
    --workdir "$ROOT/microduck_rl" \
    --env "HOME=$ROOT/.container-home" --env "TMPDIR=$ROOT/.tmp" \
    --env "XDG_CACHE_HOME=$ROOT/.cache" --env "UV_CACHE_DIR=$ROOT/.cache/uv" \
    --env "UV_PYTHON_INSTALL_DIR=$ROOT/.python" \
    --env "UV_PROJECT_ENVIRONMENT=$ROOT/microduck_rl/.venv-container" \
    --env UV_HTTP_TIMEOUT=600 --env MUJOCO_GL=egl --env OMP_NUM_THREADS=4 \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --label microduck.training=true --entrypoint /bin/sleep "$IMAGE" infinity
printf 'Install: docker exec %q ../uv sync --locked\n' "$NAME"
printf 'Run: docker exec %q .venv-container/bin/python scripts/train_hd1910_suite.py --output reports/pilot_container\n' "$NAME"
