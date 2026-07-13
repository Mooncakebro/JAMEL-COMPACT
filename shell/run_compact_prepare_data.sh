#!/bin/bash
# run_compact_prepare_data.sh — Prepare SFT data for JAMEL-COMPACT
#
# Unlike original JAMEL, COMPACT does NOT need offline memory compression.
# This script just concatenates, shuffles, and splits trajectory parquet
# files into train/val.
#
# Supports three INPUT formats:
#   1. Single parquet file:
#      INPUT=/path/to/trajectory.parquet
#
#   2. Directory with app subdirectories (ExplorerSFT-ReAct layout):
#      INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision
#      → auto-discovers react-vision/*/trajectory.parquet (80 apps)
#
#   3. Root directory containing both react-text and react-vision:
#      INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data
#      VARIANT=react-vision   # filter to one variant
#
# Usage:
#   INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
#   OUTPUT_DIR=data/compact_sft_data \
#   VAL_RATIO=0.05 \
#   bash shell/run_compact_prepare_data.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PYTHONPATH="$JAMEL_ROOT:${PYTHONPATH:-}"

INPUT=${INPUT:-/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision}
OUTPUT_DIR=${OUTPUT_DIR:-data/compact_sft_data}
VAL_RATIO=${VAL_RATIO:-0.05}
VARIANT=${VARIANT:-}          # react-text or react-vision (empty = auto)
APPS=${APPS:-}                # comma-separated app filter (empty = all)

if [[ ! -e "$INPUT" ]]; then
    echo "ERROR: INPUT not found: $INPUT" >&2
    exit 2
fi

EXTRA_ARGS=""
if [[ -n "$VARIANT" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --variant $VARIANT"
fi
if [[ -n "$APPS" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --apps $APPS"
fi

echo "=== JAMEL-COMPACT Data Preparation ==="
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT_DIR"
echo "  Val ratio:  $VAL_RATIO"
echo "  Variant:    ${VARIANT:-auto}"
echo "  Apps filter: ${APPS:-all}"
echo ""

python -m jamel_compact.data_cli \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --val-ratio "$VAL_RATIO" \
    $EXTRA_ARGS