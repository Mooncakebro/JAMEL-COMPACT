#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
CHECKPOINT=${CHECKPOINT:-}
COMPRESSOR_MODEL=${COMPRESSOR_MODEL:-}
OUTPUT_MODEL_PATH=${OUTPUT_MODEL_PATH:-"$ROOT/outputs/jamel_model"}

usage() {
    cat <<'EOF'
Usage:
  CHECKPOINT=/path/to/jamel_actor_checkpoint \
  COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
  [OUTPUT_MODEL_PATH=/path/to/jamel_model] \
  bash shell/package_model.sh

The output model directory will contain:
  actor/
  compressor/
EOF
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: package_model.sh does not accept positional arguments: $*" >&2
            echo "Use environment variables instead. See: bash shell/package_model.sh --help" >&2
            exit 2
            ;;
    esac
fi

if [[ -z "$CHECKPOINT" || -z "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: CHECKPOINT and COMPRESSOR_MODEL are required." >&2
    usage >&2
    exit 2
fi

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: CHECKPOINT is not a directory: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -d "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: COMPRESSOR_MODEL is not a directory: $COMPRESSOR_MODEL" >&2
    exit 2
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" -m jamel.train.memory.package_model \
    --checkpoint "$CHECKPOINT" \
    --compressor-model "$COMPRESSOR_MODEL" \
    --output-model-path "$OUTPUT_MODEL_PATH"
