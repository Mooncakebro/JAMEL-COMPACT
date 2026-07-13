#!/bin/bash
# run_compact_prepare_data.sh — Prepare SFT data for JAMEL-COMPACT
#
# Unlike original JAMEL, COMPACT does NOT need offline memory compression.
# This script just splits augmented trajectory parquet into train/val.
#
# Usage:
#   INPUT=/path/to/augmented_trajectory.parquet \
#   OUTPUT_DIR=data/compact_sft_data \
#   VAL_RATIO=0.05 \
#   bash shell/run_compact_prepare_data.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PYTHONPATH="$JAMEL_ROOT:${PYTHONPATH:-}"

INPUT=${INPUT:-data/augmented_accepted_samples.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-data/compact_sft_data}
VAL_RATIO=${VAL_RATIO:-0.05}

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: INPUT parquet not found: $INPUT" >&2
    exit 2
fi

echo "=== JAMEL-COMPACT Data Preparation ==="
echo "  Input:      $INPUT"
echo "  Output:     $OUTPUT_DIR"
echo "  Val ratio:  $VAL_RATIO"
echo ""

python -c "
from jamel_compact.data import prepare_compact_dataset
import sys

input_file = sys.argv[1]
output_dir = sys.argv[2]
val_ratio = float(sys.argv[3])

train_path, val_path = prepare_compact_dataset(
    input_files=input_file,
    output_dir=output_dir,
    val_ratio=val_ratio,
)
print(f'Train: {train_path}')
print(f'Val:   {val_path}')
" "$INPUT" "$OUTPUT_DIR" "$VAL_RATIO"