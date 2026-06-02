#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
INSTALL_NODE_TOOLS=${INSTALL_NODE_TOOLS:-1}

cd "$ROOT"

echo "== JAMEL environment setup =="
echo "  root:    $ROOT"
echo "  python:  $PYTHON_VERSION"
echo

if [[ -e .venv/bin/python3 ]]; then
    if ! .venv/bin/python3 - <<'PY' >/dev/null 2>&1
import sys
print(sys.executable)
PY
    then
        echo "Removing stale or inaccessible .venv before uv sync."
        rm -rf .venv
    fi
fi

uv sync --locked --python "$PYTHON_VERSION" --extra dev --extra train
uv run --no-sync playwright install chromium

if [[ "$INSTALL_NODE_TOOLS" == "1" ]]; then
    npm install -g monocart-coverage-reports monocart-locator
fi

uv run --no-sync python - <<'PY'
import sys
import numpy
import torch
import transformers

print("== Python runtime ==")
print("python:", sys.executable)
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY

NODE_PATH=$(npm root -g) node -e "require('monocart-coverage-reports'); console.log('monocart ok')"

echo
echo "Environment ready. Activate with:"
echo "  source $ROOT/.venv/bin/activate"
