#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE_NAME=${IMAGE_NAME:-jamel:latest}
INSTALL_TRAIN_DEPS=${INSTALL_TRAIN_DEPS:-0}

docker build \
  --network=host \
  --build-arg HTTP_PROXY="${http_proxy:-}" \
  --build-arg HTTPS_PROXY="${https_proxy:-}" \
  --build-arg INSTALL_TRAIN_DEPS="$INSTALL_TRAIN_DEPS" \
  -t "$IMAGE_NAME" \
  -f "$ROOT/docker/jamel/Dockerfile" \
  "$ROOT"
