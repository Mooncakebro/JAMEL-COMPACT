#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SCALEWOB_ROOT=${SCALEWOB_ROOT:-"$ROOT/env/browser_env/scalewob-env"}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python
  fi
fi

"$PYTHON_BIN" "$ROOT/scripts/serve_scalewob.py" \
  --root "$SCALEWOB_ROOT" \
  --host "$HOST" \
  --port "$PORT"
