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
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-0}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
LORA_BIAS=${LORA_BIAS:-none}
LOG_STEPS=${LOG_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
VAL_STEPS=${VAL_STEPS:-200}
GPU_IDS=${GPU_IDS:-}
FSDP=${FSDP:-auto}                # auto, 1 = torchrun FULL_SHARD, 0 = DataParallel
NPROC_PER_NODE=${NPROC_PER_NODE:-}

if ! [[ "$LORA_RANK" =~ ^[0-9]+$ && "$LORA_ALPHA" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LORA_RANK and LORA_ALPHA must be non-negative integers." >&2
    exit 2
fi
if (( LORA_RANK > 0 && LORA_ALPHA == 0 )); then
    echo "ERROR: LORA_ALPHA must be > 0 when LORA_RANK > 0." >&2
    exit 2
fi
if (( LORA_RANK == 0 && LORA_ALPHA != 0 )); then
    echo "ERROR: LORA_ALPHA must be 0 when LORA_RANK=0." >&2
    exit 2
fi

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

if [[ -z "$NPROC_PER_NODE" ]]; then
    if [[ -n "$GPU_IDS" ]]; then
        IFS=',' read -r -a _selected_gpus <<< "$GPU_IDS"
        NPROC_PER_NODE=${#_selected_gpus[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    else
        NPROC_PER_NODE=1
    fi
fi

FSDP_ACTIVE=0
if [[ "$FSDP" == "1" ]]; then
    FSDP_ACTIVE=1
elif [[ "$FSDP" == "auto" && "$NPROC_PER_NODE" -gt 1 ]]; then
    FSDP_ACTIVE=1
fi
if [[ "$FSDP_ACTIVE" == "1" && "$NPROC_PER_NODE" -lt 2 ]]; then
    echo "ERROR: FSDP requires at least two selected GPUs." >&2
    exit 2
fi
if [[ "$FSDP_ACTIVE" == "1" && -n "$GPU_IDS" \
      && "$NPROC_PER_NODE" -ne "${#_selected_gpus[@]}" ]]; then
    echo "ERROR: NPROC_PER_NODE=$NPROC_PER_NODE must equal the " \
         "${#_selected_gpus[@]} entries in GPU_IDS=$GPU_IDS." >&2
    exit 2
fi
if [[ "$FSDP_ACTIVE" == "1" ]] && ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun was not found in PATH." >&2
    exit 2
fi

LAUNCH=(python -m jamel_compact.baseline_train)
FSDP_ARG=()
if [[ "$FSDP_ACTIVE" == "1" ]]; then
    LAUNCH=(
        torchrun
        --standalone
        --nproc-per-node "$NPROC_PER_NODE"
        --module jamel_compact.baseline_train
    )
    FSDP_ARG=(--fsdp)
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
echo "  LoRA:        rank=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  LoRA targets: $LORA_TARGET_MODULES"
echo "  FSDP:        $FSDP_ACTIVE ($NPROC_PER_NODE processes)"
echo ""

exec "${LAUNCH[@]}" \
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
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --lora-bias "$LORA_BIAS" \
    --log-steps "$LOG_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --val-steps "$VAL_STEPS" \
    "${FSDP_ARG[@]}" \
    $GPU_ARG \
    "$@"
