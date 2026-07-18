#!/bin/bash
# run_baseline_train.sh — Train pure Qwen3-VL baseline (no side memory)
#
# Standard SFT of Qwen3-VL on the same data as JAMEL-COMPACT.
# No memory modules, no chunking — just plain next-token CE loss.
#
# Usage:
#   TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
#   VAL_FILE=data/compact_sft_data/compact_val.parquet \
#   BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
#   OUTPUT_DIR=outputs/baseline_ckpt \
#   TB_LOG_DIR=outputs/baseline_tb \
#   GPU_IDS=0,1,2,3 \
#   bash shell/run_baseline_train.sh
#
# TensorBoard:
#   tensorboard --logdir outputs/baseline_tb

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PYTHONPATH="$JAMEL_ROOT:${PYTHONPATH:-}"

TRAIN_FILE=${TRAIN_FILE:-data/compact_sft_data/compact_train.parquet}
VAL_FILE=${VAL_FILE:-data/compact_sft_data/compact_val.parquet}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/baseline_ckpt}
TB_LOG_DIR=${TB_LOG_DIR:-outputs/baseline_tb}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_EPOCHS=${MAX_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-5}
LOG_STEPS=${LOG_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
VAL_STEPS=${VAL_STEPS:-200}
GPU_IDS=${GPU_IDS:-}

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
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    GPU_ARG="--gpu-ids $GPU_IDS"
fi

echo "=== Baseline Qwen3-VL SFT Training ==="
echo "  Base model:  $BASE_MODEL"
echo "  Train file:  $TRAIN_FILE"
echo "  Val file:    $VAL_FILE"
echo "  Output:      $OUTPUT_DIR"
echo "  TensorBoard: $TB_LOG_DIR"
echo "  GPUs:        ${GPU_IDS:-all}"
echo "  Max length:  $MAX_LENGTH"
echo "  Epochs:      $MAX_EPOCHS"
echo "  Batch:       $BATCH_SIZE × $GRAD_ACCUM (accum)"
echo "  LR:          $LR"
echo ""

exec python -m jamel_compact.baseline_train \
    --train-file "$TRAIN_FILE" \
    --val-file "$VAL_FILE" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --tb-log-dir "$TB_LOG_DIR" \
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
