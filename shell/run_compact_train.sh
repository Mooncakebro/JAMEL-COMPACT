#!/bin/bash
# run_compact_train.sh — Train JAMEL-COMPACT
#
# Usage:
#   TRAIN_FILE=data/compact_train.parquet \
#   VAL_FILE=data/compact_val.parquet \
#   BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
#   OUTPUT_DIR=outputs/compact_ckpt \
#   TB_LOG_DIR=outputs/compact_tb \
#   GPU_IDS=0 \
#   bash shell/run_compact_train.sh
#
# GPU selection:
#   GPU_IDS=0          # single GPU 0
#   GPU_IDS=0,1,2      # GPUs 0, 1, 2
#   GPU_IDS=""          # all available GPUs (default)
#
# TensorBoard:
#   tensorboard --logdir outputs/compact_tb

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PYTHONPATH="$JAMEL_ROOT:${PYTHONPATH:-}"

TRAIN_FILE=${TRAIN_FILE:-data/compact_train.parquet}
VAL_FILE=${VAL_FILE:-data/compact_val.parquet}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/compact_ckpt}
TB_LOG_DIR=${TB_LOG_DIR:-outputs/compact_tb}
MEM_DIM=${MEM_DIM:-512}
NUM_MEM=${NUM_MEM:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_EPOCHS=${MAX_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-5}
LOG_STEPS=${LOG_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
VAL_STEPS=${VAL_STEPS:-200}
GPU_IDS=${GPU_IDS:-}              # e.g. "0" or "0,1,2" or "" (all)

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: TRAIN_FILE not found: $TRAIN_FILE" >&2
    exit 2
fi
if [[ ! -f "$VAL_FILE" ]]; then
    echo "ERROR: VAL_FILE not found: $VAL_FILE" >&2
    exit 2
fi

# Build GPU args
GPU_ARG=""
if [[ -n "$GPU_IDS" ]]; then
    GPU_ARG="--gpu-ids $GPU_IDS"
fi

echo "=== JAMEL-COMPACT Training ==="
echo "  Base model:  $BASE_MODEL"
echo "  Train file:   $TRAIN_FILE"
echo "  Val file:     $VAL_FILE"
echo "  Output:       $OUTPUT_DIR"
echo "  TensorBoard:  $TB_LOG_DIR"
echo "  GPUs:         ${GPU_IDS:-all}"
echo "  Mem dim:      $MEM_DIM"
echo "  Num mem:      $NUM_MEM"
echo "  Max length:   $MAX_LENGTH"
echo "  Epochs:       $MAX_EPOCHS"
echo "  Batch:        $BATCH_SIZE × $GRAD_ACCUM (accum)"
echo "  LR:           $LR"
echo ""

python -m jamel_compact.train \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --tb-log-dir "$TB_LOG_DIR" \
    --mem-dim "$MEM_DIM" \
    --num-mem-tokens "$NUM_MEM" \
    --max-length "$MAX_LENGTH" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --log-steps "$LOG_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --val-steps "$VAL_STEPS" \
    $GPU_ARG \
    "$@"