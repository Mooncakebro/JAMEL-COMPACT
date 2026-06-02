#!/usr/bin/env bash
set -euo pipefail

JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$JAMEL_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$JAMEL_ROOT/.venv/bin/python"
  else
    PYTHON_BIN=python
  fi
fi

exec "$PYTHON_BIN" "$JAMEL_ROOT/scripts/download_scalewob_env.py" "$@"
