#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE_NAME=${IMAGE_NAME:-jamel:latest}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-/workspace/JAMEL}

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  --ipc=host \
  --network=host \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e MODELSCOPE_CACHE=/workspace/.cache/modelscope \
  -e JAMEL_ROOT="$CONTAINER_WORKDIR" \
  -e VERL_AGENT_ROOT="$CONTAINER_WORKDIR/third_party/verl-agent" \
  -e SCALEWOB_ROOT="$CONTAINER_WORKDIR/env/browser_env/scalewob-env" \
  -e PYTHONPATH="$CONTAINER_WORKDIR:$CONTAINER_WORKDIR/third_party/verl-agent" \
  -v "$ROOT:$CONTAINER_WORKDIR" \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/workspace/.cache/huggingface" \
  -v "${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}:/workspace/.cache/modelscope" \
  -w "$CONTAINER_WORKDIR" \
  "$IMAGE_NAME" \
  "$@"
