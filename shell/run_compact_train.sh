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
#   MODEL_PARALLEL=1   # shard a frozen 8B model across selected GPUs
#   FSDP=1             # full-base multi-GPU SFT with torchrun FULL_SHARD
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
MEM_DIM=${MEM_DIM:-8}
NUM_MEM=${NUM_MEM:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
MAX_EPOCHS=${MAX_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-2e-5}
MEMORY_LR=${MEMORY_LR:-5e-6}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-0}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
LORA_BIAS=${LORA_BIAS:-none}
LOG_STEPS=${LOG_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-500}
VAL_STEPS=${VAL_STEPS:-200}
LAMBDA_OBS=${LAMBDA_OBS:-0.01}
LAMBDA_NLL=${LAMBDA_NLL:-0.01}
GPU_IDS=${GPU_IDS:-}              # e.g. "0" or "0,1,2" or "" (all)
CHUNK_SIZE=${CHUNK_SIZE:-8}          # 1 = single-step, >1 = session-chunked (v2 default: 8)
COVERAGE_WEIGHT_ETA=${COVERAGE_WEIGHT_ETA:-0.0}  # F7: 0=off, >0 upweights high-novelty samples
FREEZE_BASE=${FREEZE_BASE:-1}       # 1 = memory + optional LoRA; 0 = dense full SFT
MODEL_PARALLEL=${MODEL_PARALLEL:-auto} # auto, 1 = model sharding, 0 = DataParallel
FSDP=${FSDP:-auto}                    # auto, 1 = torchrun FULL_SHARD, 0 = disabled
NPROC_PER_NODE=${NPROC_PER_NODE:-}

if [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] && (( CHUNK_SIZE > 1 )) && [[ "$BATCH_SIZE" != "1" ]]; then
    echo "WARNING: CHUNK_SIZE=$CHUNK_SIZE uses recurrent training and requires BATCH_SIZE=1." >&2
    echo "Overriding BATCH_SIZE=$BATCH_SIZE to 1; increase GRAD_ACCUM for a larger effective batch." >&2
    BATCH_SIZE=1
fi

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
if (( LORA_RANK > 0 )) && [[ "$FREEZE_BASE" != "1" ]]; then
    echo "ERROR: LoRA requires FREEZE_BASE=1; dense base weights stay frozen." >&2
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
# Set CUDA_VISIBLE_DEVICES in the shell BEFORE Python launches.
# Setting it inside Python (after import torch) is too late — CUDA context
# is already initialized on the default GPU.
if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    GPU_ARG="--gpu-ids $GPU_IDS"
else
    GPU_ARG=""
fi

if [[ "$FREEZE_BASE" == "1" ]]; then
    BASE_TRAINING_ARG=("--freeze-base")
else
    BASE_TRAINING_ARG=("--train-base")
fi

SELECTED_GPU_COUNT=""
if [[ -n "$GPU_IDS" ]]; then
    IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_IDS"
    SELECTED_GPU_COUNT=${#SELECTED_GPUS[@]}
fi

if [[ -z "$NPROC_PER_NODE" ]]; then
    if [[ -n "$SELECTED_GPU_COUNT" ]]; then
        NPROC_PER_NODE=$SELECTED_GPU_COUNT
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
    else
        NPROC_PER_NODE=1
    fi
fi

FSDP_ACTIVE=0
if [[ "$FSDP" == "1" ]]; then
    FSDP_ACTIVE=1
elif [[ "$FSDP" == "auto" && "$FREEZE_BASE" == "0" && "$NPROC_PER_NODE" -gt 1 ]]; then
    FSDP_ACTIVE=1
fi

if [[ "$FSDP_ACTIVE" == "1" && "$NPROC_PER_NODE" -lt 2 ]]; then
    echo "ERROR: FSDP requires at least two selected GPUs." >&2
    exit 2
fi
if [[ "$FSDP_ACTIVE" == "1" && -n "$SELECTED_GPU_COUNT" && "$NPROC_PER_NODE" -ne "$SELECTED_GPU_COUNT" ]]; then
    echo "ERROR: NPROC_PER_NODE=$NPROC_PER_NODE must equal the $SELECTED_GPU_COUNT entries in GPU_IDS=$GPU_IDS." >&2
    exit 2
fi
if [[ "$FSDP_ACTIVE" == "1" ]] && ! command -v torchrun >/dev/null 2>&1; then
    echo "ERROR: torchrun was not found in PATH. Install/use a PyTorch environment with distributed support." >&2
    exit 2
fi

MODEL_PARALLEL_ARG=()
if [[ "$MODEL_PARALLEL" == "1" ]]; then
    if [[ "$FREEZE_BASE" != "1" ]]; then
        echo "ERROR: MODEL_PARALLEL=1 uses frozen-base device-map sharding." >&2
        echo "Use FREEZE_BASE=0 FSDP=1 for full-base multi-GPU training." >&2
        exit 2
    fi
    if [[ "$FSDP_ACTIVE" == "1" ]]; then
        echo "ERROR: MODEL_PARALLEL=1 and FSDP=1 are mutually exclusive." >&2
        exit 2
    fi
    MODEL_PARALLEL_ARG=("--model-parallel")
elif [[ "$MODEL_PARALLEL" == "0" ]]; then
    MODEL_PARALLEL_ARG=("--data-parallel")
fi

FSDP_ARG=()
LAUNCH=(python -m jamel_compact.train)
if [[ "$FSDP_ACTIVE" == "1" ]]; then
    FSDP_ARG=("--fsdp")
    MODEL_PARALLEL_ARG=()
    LAUNCH=(
        torchrun
        --standalone
        --nproc-per-node "$NPROC_PER_NODE"
        --module jamel_compact.train
    )
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
echo "  Chunk size:   $CHUNK_SIZE"
echo "  Base LR:      $LR"
echo "  Memory LR:    $MEMORY_LR"
echo "  LoRA:         rank=$LORA_RANK alpha=$LORA_ALPHA dropout=$LORA_DROPOUT"
echo "  LoRA targets: $LORA_TARGET_MODULES"
echo "  Freeze base:  $FREEZE_BASE"
echo "  Model shard:  $MODEL_PARALLEL"
echo "  FSDP:         $FSDP_ACTIVE ($NPROC_PER_NODE processes)"
echo "  Val every:    $VAL_STEPS optimizer steps"
echo "  Lambda obs:   $LAMBDA_OBS"
echo "  Lambda NLL:   $LAMBDA_NLL"
echo "  Cov weight:   $COVERAGE_WEIGHT_ETA"
exec "${LAUNCH[@]}" \
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
    --memory-lr "$MEMORY_LR" \
    --lora-rank "$LORA_RANK" \
    --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" \
    --lora-target-modules "$LORA_TARGET_MODULES" \
    --lora-bias "$LORA_BIAS" \
    --log-steps "$LOG_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --val-steps "$VAL_STEPS" \
    --lambda-obs "$LAMBDA_OBS" \
    --lambda-nll "$LAMBDA_NLL" \
    --chunk-size "$CHUNK_SIZE" \
    --coverage-weight-eta "$COVERAGE_WEIGHT_ETA" \
    "${BASE_TRAINING_ARG[@]}" \
    "${MODEL_PARALLEL_ARG[@]}" \
    "${FSDP_ARG[@]}" \
    $GPU_ARG \
    "$@"
