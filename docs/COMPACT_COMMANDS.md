# JAMEL-COMPACT: Full Commands (Data Prep → Training → Eval)

This document provides the complete commands to run the JAMEL-COMPACT pipeline end-to-end.

---

## Step 0: Environment Setup

```bash
cd ~/JAMEL-COMPACT

# Set paths
export JAMEL_ROOT=$PWD
export PYTHONPATH=$JAMEL_ROOT:$PYTHONPATH

# Install dependencies (if not already done)
uv sync --locked --python 3.10 --extra dev --extra train
uv run playwright install chromium

# Install system fonts (required for Chinese apps in ScaleWoB)
sudo apt-get update
sudo apt-get install -y fontconfig fonts-noto-cjk fonts-noto-color-emoji
fc-cache -fv

# Download ScaleWoB benchmark (required for evaluation)
python scripts/download_scalewob_env.py
```

---

## Step 1: Data Preparation

Unlike original JAMEL, COMPACT does **not** need offline memory compression. This step auto-discovers `trajectory.parquet` files from the ExplorerSFT-ReAct dataset, shuffles, and splits into train/val.

### 1a. Quick test (2-3 apps)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
OUTPUT_DIR=data/compact_sft_data_example \
APPS=weibo,alipay \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

### 1b. Full react-vision dataset (80 apps, ~12K rows)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
OUTPUT_DIR=data/compact_sft_data \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

> **Note**: The full 80-app dataset is ~2.3GB of screenshots. If memory is limited, use the `APPS` filter to process fewer apps at a time, then manually concatenate the output parquet files.

### 1c. react-text variant (text-only, no screenshots)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-text \
OUTPUT_DIR=data/compact_sft_data_text \
VARIANT=react-text \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

### 1d. Both variants combined (160 apps, ~24K rows)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data \
OUTPUT_DIR=data/compact_sft_data_all \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

**What happens:**
- Auto-discovers all `trajectory.parquet` files under app subdirectories
- Phase 1: reads metadata (no screenshots) from all files → shuffle → produce train/val index sets
- Phase 2: streams each file, filters rows by index, writes directly to train/val parquet using `pyarrow.ParquetWriter`
- **No data copies in memory** — avoids OOM even for large datasets

**Input dataset structure:**
```
/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/
├── data/
│   ├── react-text/         # 80 apps, text-only (still has screenshots)
│   │   ├── weibo/trajectory.parquet   (150 rows)
│   │   ├── alipay/trajectory.parquet  (150 rows)
│   │   └── ...
│   └── react-vision/       # 80 apps, with vision prompts
│       ├── weibo/trajectory.parquet   (150 rows)
│       ├── alipay/trajectory.parquet  (150 rows)
│       └── ...
└── metadata/
    ├── manifest.json
    └── sessions.csv
```

**Output columns (essential subset retained):**
```
action, before_observation_str, before_open_pages_urls, before_screenshot,
coverage_delta_score, prompt, response, reward, session_id, step_idx,
session_step_idx, start_url, target_app, think, parsed_content, ...
```

**Output:**
```
data/compact_sft_data/
├── compact_train.parquet
└── compact_val.parquet
```

---

## Step 2: Training

### 2a. Train with Qwen3-VL-2B (default)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt \
TB_LOG_DIR=outputs/compact_tb \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=2e-5 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2b. Train with Qwen3-VL-8B

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b \
TB_LOG_DIR=outputs/compact_tb_8b \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=32 \
LR=1e-5 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2c. Freeze base model (train only side memory)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_frozen \
TB_LOG_DIR=outputs/compact_tb_frozen \
FREEZE_BASE=1 \
bash shell/run_compact_train.sh --freeze-base
```

### 2d. Monitor training with TensorBoard

Open a separate terminal:

```bash
tensorboard --logdir outputs/compact_tb --port 6006
# Open http://localhost:6006 in browser
```

**Logged metrics:**
- `train/loss_total`, `train/loss_action`, `train/loss_mem_l2`, `train/loss_mem_entropy`, `train/loss_uncert`
- `memory/layer{l}_mem_mean`, `memory/layer{l}_mem_std`, `memory/layer{l}_conf_mean`, `memory/layer{l}_conf_std`
- `train/grad_norm`, `train/learning_rate`, `train/step_time_s`
- `val/loss_total`, `val/loss_action`, `val/loss_mem_l2`, `val/loss_uncert`

**Output:**
```
outputs/compact_ckpt/
├── global_step_500/
│   ├── base_model/  (or base_model_ref.txt if frozen)
│   ├── side_memory/
│   │   ├── side_memories.pt
│   │   └── action_embed.pt
│   ├── compact_config.json
│   └── tokenizer files...
├── global_step_1000/
├── best/
└── final/
```

---

## Step 3: Evaluation

### 3a. Evaluate on test10 apps (paper setting)

```bash
CHECKPOINT=outputs/compact_ckpt/final \
APPS_MODE=test10 \
SCALEWOB_ROOT=env/browser_env/scalewob-env \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval \
DEVICE=cuda \
TEMPERATURE=0.8 \
TOP_P=0.9 \
bash shell/run_compact_eval.sh
```

### 3b. Evaluate on train86 apps (sanity check)

```bash
CHECKPOINT=outputs/compact_ckpt/final \
APPS_MODE=train86 \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_train86 \
bash shell/run_compact_eval.sh
```

### 3c. Single-app debug

```bash
CHECKPOINT=outputs/compact_ckpt/final \
APPS=weibo \
MAX_STEPS=20 \
NUM_SESSIONS=1 \
EVAL_OUTPUT=outputs/compact_eval_debug \
bash shell/run_compact_eval.sh
```

### 3d. Custom app list

```bash
CHECKPOINT=outputs/compact_ckpt/final \
APPS="alibaba jd taobao" \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_custom \
bash shell/run_compact_eval.sh
```

**Output:**
```
outputs/compact_eval/
├── vipshop_session0.parquet
├── vipshop_session1.parquet
├── vipshop_session2.parquet
├── alibaba_session0.parquet
├── ...
└── eval_summary.json
```

---

## Full Pipeline (All-in-One)

```bash
# ── 1. Data prep (react-vision, 80 apps) ──
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
OUTPUT_DIR=data/compact_sft_data \
bash shell/run_compact_prepare_data.sh

# ── 2. Train ──
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt \
TB_LOG_DIR=outputs/compact_tb \
bash shell/run_compact_train.sh

# ── 3. Eval ──
CHECKPOINT=outputs/compact_ckpt/final \
APPS_MODE=test10 \
EVAL_OUTPUT=outputs/compact_eval \
bash shell/run_compact_eval.sh
```

---

## Environment Variables Reference

### Data Preparation (`run_compact_prepare_data.sh`)

| Variable | Default | Description |
|---|---|---|
| `INPUT` | `.../react-vision` | Path to parquet file, directory, or list |
| `OUTPUT_DIR` | `data/compact_sft_data` | Output directory for train/val parquet |
| `VAL_RATIO` | `0.05` | Fraction of data for validation |
| `VARIANT` | (empty) | Filter: `react-text` or `react-vision` |
| `APPS` | (empty) | Comma-separated app names to filter |

### Training (`run_compact_train.sh`)

| Variable | Default | Description |
|---|---|---|
| `TRAIN_FILE` | `data/compact_train.parquet` | Train parquet file |
| `VAL_FILE` | `data/compact_val.parquet` | Val parquet file |
| `BASE_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Pretrained base model name or path |
| `OUTPUT_DIR` | `outputs/compact_ckpt` | Checkpoint output directory |
| `TB_LOG_DIR` | `outputs/compact_tb` | TensorBoard log directory |
| `MEM_DIM` | `512` | Reduced memory dimension $d_{mem}$ |
| `NUM_MEM` | `16` | Memory tokens per layer $N_m$ |
| `MAX_LENGTH` | `8192` | Max token length |
| `MAX_EPOCHS` | `3` | Number of training epochs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `LR` | `2e-5` | Learning rate |
| `LOG_STEPS` | `10` | TensorBoard logging frequency |
| `SAVE_STEPS` | `500` | Checkpoint save frequency |
| `VAL_STEPS` | `200` | Validation frequency |

### Evaluation (`run_compact_eval.sh`)

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `outputs/compact_ckpt/final` | Model checkpoint directory |
| `APPS_MODE` | `test10` | App split: `test10`, `train86`, or `all` |
| `APPS` | (empty) | Explicit app list (overrides `APPS_MODE`) |
| `SCALEWOB_ROOT` | `env/browser_env/scalewob-env` | ScaleWoB static files directory |
| `MAX_STEPS` | `50` | Steps per session |
| `NUM_SESSIONS` | `3` | Sessions per app |
| `EVAL_OUTPUT` | `outputs/compact_eval` | Evaluation output directory |
| `DEVICE` | `cuda` | Device for inference |
| `TEMPERATURE` | `0.8` | Sampling temperature |
| `TOP_P` | `0.9` | Top-p sampling |

---

## Parameter Overhead Summary

| Base Model | $d$ | Layers | Per-layer | Total new | Base | Overhead |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 1536 | 28 | 8.41M | 237.9M | ~2.0B | **11.9%** |
| Qwen3-VL-8B | 4096 | 36 | 13.66M | 508.7M | ~8.0B | **6.4%** |

The overhead **decreases** for larger base models because the projection layers (which scale with $d \times d_{mem}$) become a smaller fraction of the total.