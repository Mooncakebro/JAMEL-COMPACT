#!/bin/bash
# run_react_full_sft.sh — 全量 ReAct SFT（train86 apps trajectory.parquet）
#
# 共享底层启动器 run_qwen25vl_7b_sft.sh。规模相关参数可通过环境变量覆盖。
#
# 流程：
#   1. 把 data/react/react-{text,vision}/<APPS...>/trajectory.parquet
#      合并 → SFT parquet（所有 session 共用一个 train/val 切分）。
#   2. 启动 8×A800 FSDP2 SFT，使用 web_prompt（训推共享）。
#   3. 每个 epoch 结束保存 ckpt，最多保留 SAVE_TOTAL_LIMIT 个。
#
# 用法：
#   bash shell/run_react_full_sft.sh                    # 全部 app
#   APPS="youdao alipay amap" bash shell/run_react_full_sft.sh   # 子集
#
# 关键路径（可环境变量覆盖）：
#   REACT_ROOT      data/react
#   APPS            （留空 = REACT_ROOT 下所有目录）
#   DATASET_DIR     data/jamel_react_full_sft
#   OUTPUT_DIR      outputs/jamel_sft_ckpt_react_full
#   BASE_MODEL_PATH Qwen/Qwen2.5-VL-7B-Instruct
#   COMPRESSOR_MODEL local Qwen3-VL-2B-Instruct directory
#   OUTPUT_MODEL_PATH optional packaged JAMEL model path for evaluation
#
# 长期维护约定：
#   recipe（prompt 格式 / 图像尺寸 / prune 阈值 / memory 配置 / 优化器 / LR /
#   batch / max_length）由 run_qwen25vl_7b_sft.sh 统一维护。

set -euo pipefail

JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
VERL_AGENT_ROOT=${VERL_AGENT_ROOT:-$(cd "$JAMEL_ROOT/third_party/verl-agent" && pwd)}
export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

REACT_ROOT=${REACT_ROOT:-"$JAMEL_ROOT/data/react"}
DATASET_DIR=${DATASET_DIR:-"$JAMEL_ROOT/data/jamel_react_full_sft"}
OUTPUT_DIR=${OUTPUT_DIR:-"$JAMEL_ROOT/outputs/jamel_sft_ckpt_react_full"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}}
COMPRESSOR_MODEL=${COMPRESSOR_MODEL:-}
OUTPUT_MODEL_PATH=${OUTPUT_MODEL_PATH:-}

# Scale-specific defaults:
#   TOTAL_EPOCHS=3
#   SAVE_TOTAL_LIMIT=6
#   VAL_STEPS=200
#   VAL_RATIO=0.02
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-6}
VAL_STEPS=${VAL_STEPS:-200}
VAL_RATIO=${VAL_RATIO:-0.02}

LOG_DIR=${LOG_DIR:-"$JAMEL_ROOT/outputs/logs"}
PREP_LOG=${PREP_LOG:-"$LOG_DIR/jamel_react_full_sft_prepare.log"}
TRAIN_LOG=${TRAIN_LOG:-"$LOG_DIR/jamel_react_full_sft_train.log"}

if [[ -z "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: COMPRESSOR_MODEL is required for SFT data preparation." >&2
    exit 2
fi
if [[ ! -d "$COMPRESSOR_MODEL" ]]; then
    echo "ERROR: COMPRESSOR_MODEL must be a local directory: $COMPRESSOR_MODEL" >&2
    exit 2
fi

# Resolve app list → input parquet paths.
if [[ -z "${APPS:-}" ]]; then
    if [[ ! -d "$REACT_ROOT/react-vision" ]]; then
        echo "ERROR: REACT_ROOT/react-vision not found: $REACT_ROOT/react-vision" >&2
        echo "Place trajectories under data/react or set REACT_ROOT=..." >&2
        exit 2
    fi
    mapfile -t APP_LIST < <(ls -1 "$REACT_ROOT/react-vision/" | sort)
else
    read -r -a APP_LIST <<< "$APPS"
fi

INPUT_PARQUETS=()
MISSING_APPS=()
for app in "${APP_LIST[@]}"; do
    pq="$REACT_ROOT/react-text/$app/trajectory.parquet"
    if [[ -f "$pq" ]]; then
        INPUT_PARQUETS+=("$pq")
    else
        MISSING_APPS+=("text_$app")
    fi
done

for app in "${APP_LIST[@]}"; do
    pq="$REACT_ROOT/react-vision/$app/trajectory.parquet"
    if [[ -f "$pq" ]]; then
        INPUT_PARQUETS+=("$pq")
    else
        MISSING_APPS+=("vision_$app")
    fi
done

if [[ ${#MISSING_APPS[@]} -gt 0 ]]; then
    echo "WARN: ${#MISSING_APPS[@]} apps have no trajectory.parquet, skipping:" >&2
    printf '  - %s\n' "${MISSING_APPS[@]}" >&2
fi
if [[ ${#INPUT_PARQUETS[@]} -eq 0 ]]; then
    echo "ERROR: no trajectory.parquet found under $REACT_ROOT" >&2
    exit 2
fi

echo "=== Step 1/2: prepare SFT parquet ${#INPUT_PARQUETS[@]} apps, val_ratio=$VAL_RATIO ==="
mkdir -p "$DATASET_DIR" "$LOG_DIR"
python "$JAMEL_ROOT/jamel/train/memory/prepare_sft_dataset.py" \
    --input            "${INPUT_PARQUETS[@]}" \
    --output           "$DATASET_DIR" \
    --compressor-model "$COMPRESSOR_MODEL" \
    --max-memory-items 512 \
    --max-length       8192 \
    --val-ratio        "$VAL_RATIO" \
    --compression-batch-size 4 \
    2>&1 | tee "$PREP_LOG"

echo "=== Step 2/2: SFT training epochs $TOTAL_EPOCHS, web_prompt, 640x360 image ==="
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/training_samples"

TRAIN_FILE="$DATASET_DIR/jamel_memory_sft_train.parquet" \
VAL_FILE="$DATASET_DIR/jamel_memory_sft_val.parquet" \
OUTPUT_DIR="$OUTPUT_DIR" \
BASE_MODEL_PATH="$BASE_MODEL_PATH" \
COMPRESSOR_MODEL="$COMPRESSOR_MODEL" \
OUTPUT_MODEL_PATH="$OUTPUT_MODEL_PATH" \
TOTAL_EPOCHS="$TOTAL_EPOCHS" \
SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT" \
VAL_STEPS="$VAL_STEPS" \
SFT_TRAINING_SAMPLE_DIR="$OUTPUT_DIR/training_samples" \
EXPERIMENT_NAME=qwen25vl_7b_memory_aug_sft_react_full \
bash "$JAMEL_ROOT/shell/run_qwen25vl_7b_sft.sh" 2>&1 | tee "$TRAIN_LOG"

# echo "=== Done. Checkpoints in: $OUTPUT_DIR ==="
# ls -1 "$OUTPUT_DIR" | grep -E '^global_step_' || echo "no global_step_* dirs found — check $TRAIN_LOG"
