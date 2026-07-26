# JAMEL-COMPACT v2: Full Commands (Data Prep → Training → Eval)

This document provides the complete commands to run the JAMEL-COMPACT v2 pipeline end-to-end, including both the **compact** model (with learned Kalman side memory) and the **baseline** (pure Qwen3-VL SFT, no side memory).

> **v2 changes**: Learned Kalman filter (variance P replaces pinned confidence C), multi-token observation (k=4 latent queries), zero-init injection, session-chunked training (chunk_size=8), coverage-weighted SFT, memory-conditioned generation. Old v1 checkpoints are **not compatible** (`model_version: 2`). See [COMPACT_V2_METHOD.md](COMPACT_V2_METHOD.md) for the full math.

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

Unlike original JAMEL, COMPACT does **not** need offline memory compression. This step auto-discovers `trajectory.parquet` files from the ExplorerSFT-ReAct dataset, rebuilds canonical prompts (same `build_web_prompt()` as eval), shuffles, and splits into train/val.

> **Recommended**: Use **1d** (both variants, 24K rows) — this matches original JAMEL's training data size. Since prompts are rebuilt from atomic columns, both `react-text` and `react-vision` produce identical canonical prompts.

### 1a. Quick test (2-3 apps)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
OUTPUT_DIR=data/compact_sft_data_example \
APPS=weibo,alipay \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

### 1b. Full react-vision only (80 apps, 12,000 rows)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-vision \
OUTPUT_DIR=data/compact_sft_data \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

### 1c. react-text only (80 apps, 12,000 rows)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data/react-text \
OUTPUT_DIR=data/compact_sft_data_text \
VARIANT=react-text \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

### 1d. Both variants combined ★ (160 app-dirs, 24,000 rows — recommended)

```bash
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data \
OUTPUT_DIR=data/compact_sft_data_all \
VAL_RATIO=0.05 \
bash shell/run_compact_prepare_data.sh
```

> **Why both variants work**: After prompt rebuilding, `react-text` and `react-vision` produce identical canonical prompts because both are rebuilt from the same atomic columns (`before_observation_str`, `before_screenshot`, etc.). This matches original JAMEL's behavior. The upstream `prompt` column is ignored.

**What happens:**
- Auto-discovers all `trajectory.parquet` files under app subdirectories
- Phase 1: reads metadata (no screenshots) from all files → shuffle → produce train/val index sets
- Phase 2: streams each file, filters rows by index, rebuilds prompts via `build_web_prompt()` (canonical JAMEL format with `<image>` tag), strips `<think>` from responses, writes directly to train/val parquet using `pyarrow.ParquetWriter`
- **No data copies in memory** — avoids OOM even for large datasets
- **Training/eval prompt consistency** — prompts match the canonical format used at evaluation time

> **Note**: By default, `prompt` is rebuilt from atomic columns (`before_observation_str`, etc.) and `response` has `<think>` stripped — this matches what original JAMEL does. Splitting is session-level to preserve recurrent trajectories. Pass `--no-rebuild-prompts` to the CLI if you want to keep the upstream prompt/response as-is.

**Input dataset structure:**
```
/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/
├── data/
│   ├── react-text/         # 80 apps, 12,000 rows (text-only prompts, has screenshots)
│   │   ├── weibo/trajectory.parquet   (150 rows)
│   │   ├── alipay/trajectory.parquet  (150 rows)
│   │   └── ... (80 apps total)
│   └── react-vision/       # 80 apps, 12,000 rows (vision-augmented prompts, has screenshots)
│       ├── weibo/trajectory.parquet   (150 rows)
│       ├── alipay/trajectory.parquet  (150 rows)
│       └── ... (80 apps total)
└── metadata/
    ├── manifest.json
    └── sessions.csv

Total: 160 app-dirs, 24,000 rows (80 apps × 150 rows × 2 variants)
Each row = one step in a 150-step browser exploration session.
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

## Step 2: Training (Compact Model v2 — with learned Kalman side memory)

> **Data**: Re-run data preparation after this update. The parquet schema is unchanged, but train/validation splitting is now session-level and rebuilt prompts use continuous `session_step_idx`; older row-random splits break recurrent chunk continuity.

> **v2 defaults**: `CHUNK_SIZE=8` (session-chunked training), `COVERAGE_WEIGHT_ETA=0.0` (off by default). These are passed automatically by the shell script.

### 2a. Train v2 on GPUs 6,7 (★ recommended)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_v2 \
TB_LOG_DIR=outputs/compact_tb_v2 \
GPU_IDS=6,7 \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=2e-5 \
CHUNK_SIZE=8 \
COVERAGE_WEIGHT_ETA=0.0 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2b. Train with Qwen3-VL-2B (single GPU)

```bash
TRAIN_FILE=data/compact_sft_data_all/compact_train.parquet \
VAL_FILE=data/compact_sft_data_all/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt \
TB_LOG_DIR=outputs/compact_tb \
GPU_IDS=0 \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=2e-5 \
CHUNK_SIZE=8 \
COVERAGE_WEIGHT_ETA=0.0 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2c. Train with Qwen3-VL-8B (single GPU)

```bash
TRAIN_FILE=data/compact_sft_data_all/compact_train.parquet \
VAL_FILE=data/compact_sft_data_all/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b \
TB_LOG_DIR=outputs/compact_tb_8b \
GPU_IDS=0 \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=32 \
LR=1e-5 \
CHUNK_SIZE=8 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2d. Train with coverage-weighted SFT (F7: upweight high-novelty samples)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_cw \
TB_LOG_DIR=outputs/compact_tb_cw \
GPU_IDS=6,7 \
CHUNK_SIZE=8 \
COVERAGE_WEIGHT_ETA=1.0 \
bash shell/run_compact_train.sh
```

### 2b. Train with Qwen3-VL-8B (single GPU)

```bash
TRAIN_FILE=data/compact_sft_data_all/compact_train.parquet \
VAL_FILE=data/compact_sft_data_all/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b \
TB_LOG_DIR=outputs/compact_tb_8b \
GPU_IDS=0 \
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

### 2e. Train on multiple GPUs

```bash
# Use GPUs 0, 1, 2
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt \
TB_LOG_DIR=outputs/compact_tb \
GPU_IDS=0,1,2 \
CHUNK_SIZE=8 \
bash shell/run_compact_train.sh
```

### 2f. Freeze base model (train only side memory)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_frozen \
TB_LOG_DIR=outputs/compact_tb_frozen \
GPU_IDS=0 \
FREEZE_BASE=1 \
CHUNK_SIZE=8 \
bash shell/run_compact_train.sh --freeze-base
```

### 2g. Monitor training with TensorBoard

Open a separate terminal:

```bash
tensorboard --logdir outputs/compact_tb_v2 --port 6006
# Open http://localhost:6006 in browser
```

**Logged metrics (v2):**
- `train/loss_total`, `train/loss_action`, `train/loss_action_unweighted`, `train/loss_obs`, `train/loss_nll`, `train/loss_mem_l2`
- `memory/layer{l}_gate`, `memory/layer{l}_lambda_mean`, `memory/layer{l}_Q_mean`, `memory/layer{l}_R_mean`, `memory/layer{l}_init_mem_norm`
- `train/learning_rate`, `train/step_time_s`
- `val/loss_total`, `val/loss_action`, `val/loss_obs`, `val/loss_nll`

> **v2 loss keys**: `loss_obs` (observation prediction MSE), `loss_nll` (Gaussian NLL for Kalman variance calibration), `loss_action_unweighted` (unweighted CE for comparison against coverage-weighted `loss_action`).

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

## Step 2B: Training (Baseline — pure Qwen3-VL SFT, no side memory)

The baseline trains the same pretrained Qwen3-VL model with standard SFT (plain next-token cross-entropy loss) on the same data as JAMEL-COMPACT. No memory modules, no chunking — this measures how much JAMEL-COMPACT's side memory contributes above and beyond simple SFT.

### 2B-a. Train baseline with Qwen3-VL-2B (multi-GPU)

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=/data2/songyuebing/JAMEL-DeltaState/LLMs/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/baseline_ckpt \
TB_LOG_DIR=outputs/baseline_tb \
GPU_IDS=6,7 \
MAX_LENGTH=8192 \
MAX_EPOCHS=20 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=2e-5 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_baseline_train.sh
```

### 2B-b. Train baseline with Qwen3-VL-8B

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/baseline_ckpt_8b \
TB_LOG_DIR=outputs/baseline_tb_8b \
GPU_IDS=0 \
MAX_EPOCHS=3 \
GRAD_ACCUM=32 \
LR=1e-5 \
bash shell/run_baseline_train.sh
```

### 2B-c. Monitor baseline training with TensorBoard

```bash
tensorboard --logdir outputs/baseline_tb --port 6007
# Open http://localhost:6007 in browser
```

**Logged metrics:**
- `train/loss` — cross-entropy loss
- `train/lr` — learning rate
- `val/loss` — validation cross-entropy loss

**Output:**
```
outputs/baseline_ckpt/
├── global_step_500/       # mid-epoch checkpoint
├── best/                  # best validation loss checkpoint
├── epoch0/                # per-epoch checkpoint
├── epoch1/
└── final/                 # final model
```

> **Note**: Each checkpoint directory contains model weights + tokenizer + processor files (saved together so eval can load everything from one path).

---

## Step 3: Evaluation

### 3a. Evaluate compact model on test10 apps (paper setting)

```bash
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS_MODE=test10 \
SCALEWOB_ROOT=env/browser_env/scalewob-env \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_v2 \
DEVICE=cuda \
TEMPERATURE=0.8 \
TOP_P=0.9 \
bash shell/run_compact_eval.sh
```

### 3a-bis. Evaluate with frozen memory (F5 ablation: --freeze-memory-init)

```bash
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_freeze \
FREEZE_MEMORY_INIT=1 \
bash shell/run_compact_eval.sh
```

### 3b. Evaluate on train86 apps (sanity check)

```bash
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS_MODE=train86 \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_train86 \
bash shell/run_compact_eval.sh
```

### 3c. Single-app debug

```bash
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS=weibo \
MAX_STEPS=20 \
NUM_SESSIONS=1 \
EVAL_OUTPUT=outputs/compact_eval_debug \
bash shell/run_compact_eval.sh
```

### 3d. Custom app list

```bash
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS="alibaba jd taobao" \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/compact_eval_custom \
bash shell/run_compact_eval.sh
```

### 3e. Evaluate baseline model on test10 apps

```bash
CHECKPOINT=outputs/baseline_ckpt/final \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
EVAL_OUTPUT=outputs/baseline_eval \
DEVICE=cuda \
TEMPERATURE=0.8 \
TOP_P=0.9 \
bash shell/run_baseline_eval.sh
```

### 3f. Baseline single-app debug

```bash
CHECKPOINT=outputs/baseline_ckpt/final \
APPS=weibo \
MAX_STEPS=20 \
NUM_SESSIONS=1 \
EVAL_OUTPUT=outputs/baseline_eval_debug \
bash shell/run_baseline_eval.sh
```

> **Important:** `APPS` specifies literal app names (e.g. `weibo`, `alibaba`).
> `APPS_MODE` specifies a split (e.g. `test10`, `train86`, `all`).
> Do **not** use `APPS=test10` — that will try to load a non-existent app
> called "test10" and produce 0 reward. Use `APPS_MODE=test10` instead.

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

Each session directory also contains per-step screenshots:
```
outputs/compact_eval_debug/weibo/session0/
├── step_001_before.png
├── step_001_after.png
├── step_002_before.png
├── step_002_after.png
├── ...
├── step_050_after.png
├── coverage/
└── trajectory_weibo_*.parquet
```

### 3g. Convert eval screenshots to MP4 video

After evaluation, convert the per-step `before`/`after` screenshots into a video for visualization:

```bash
# Sequential before→after per step, 2 fps (default):
python scripts/snapshots_to_mp4.py outputs/compact_eval_debug/weibo/session0/

# Side-by-side before|after, 4 fps, custom output:
python scripts/snapshots_to_mp4.py outputs/compact_eval_debug/weibo/session0/ \
    --mode side_by_side --fps 4 -o weibo_eval.mp4

# Only "after" screenshots:
python scripts/snapshots_to_mp4.py outputs/compact_eval_debug/weibo/session0/ \
    --mode after_only --fps 2
```

**Modes:**

| Mode | Description |
|------|-------------|
| `sequential` (default) | Each step produces 2 frames: before, then after |
| `side_by_side` | Each step is 1 frame with before \| after horizontally |
| `before_only` | Only before screenshots |
| `after_only` | Only after screenshots |

**Requirements:** `pip install opencv-python numpy`

---

## Full Pipeline (All-in-One)

```bash
# ── 1. Data prep (both variants, 24K rows — same as original JAMEL) ──
INPUT=/home/spc/JAMEL-DeltaState/data/ExplorerSFT-ReAct_Dataset/data \
OUTPUT_DIR=data/compact_sft_data_all \
bash shell/run_compact_prepare_data.sh

# ── 2a. Train compact model v2 (with learned Kalman side memory) ──
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_v2 \
TB_LOG_DIR=outputs/compact_tb_v2 \
GPU_IDS=6,7 \
CHUNK_SIZE=8 \
bash shell/run_compact_train.sh

# ── 2b. Train baseline (pure Qwen3-VL SFT, no side memory) ──
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/baseline_ckpt \
TB_LOG_DIR=outputs/baseline_tb \
GPU_IDS=6,7 \
MAX_EPOCHS=20 \
bash shell/run_baseline_train.sh

# ── 3a. Eval compact model v2 ──
CHECKPOINT=outputs/compact_ckpt_v2/final \
APPS_MODE=test10 \
EVAL_OUTPUT=outputs/compact_eval_v2 \
bash shell/run_compact_eval.sh

# ── 3b. Eval baseline ──
CHECKPOINT=outputs/baseline_ckpt/final \
APPS_MODE=test10 \
EVAL_OUTPUT=outputs/baseline_eval \
bash shell/run_baseline_eval.sh

# ── 4. Visualize eval trajectory as video ──
python scripts/snapshots_to_mp4.py outputs/compact_eval/weibo/session0/ \
    --mode side_by_side --fps 4 -o weibo_compact.mp4
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
| `TRAIN_FILE` | `data/compact_train.parquet` | Train parquet file (set to `data/compact_sft_data_all/compact_train.parquet` for 24K rows) |
| `VAL_FILE` | `data/compact_val.parquet` | Val parquet file (set to `data/compact_sft_data_all/compact_val.parquet` for 24K rows) |
| `BASE_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Pretrained base model name or path |
| `OUTPUT_DIR` | `outputs/compact_ckpt` | Checkpoint output directory |
| `TB_LOG_DIR` | `outputs/compact_tb` | TensorBoard log directory |
| `GPU_IDS` | (empty = all) | Comma-separated GPU IDs (e.g. `0` or `0,1,2`) |
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
| `CHUNK_SIZE` | `8` | v2: session-chunked training (1 = single-step, >1 = multi-step with memory carry-forward) |
| `COVERAGE_WEIGHT_ETA` | `0.0` | v2 F7: 0 = off; >0 upweights high-novelty samples by `1 + eta * max(coverage_delta, 0)` |

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
| `GPU_IDS` | (empty = all) | GPU ID(s) for eval (e.g. `0` or `0,1`) |
| `FREEZE_MEMORY_INIT` | `0` | v2 F5 ablation: `1` = never write new memory back to session state (tests if memory influences outputs) |

### Baseline Training (`run_baseline_train.sh`)

| Variable | Default | Description |
|---|---|---|
| `TRAIN_FILE` | `data/compact_sft_data/compact_train.parquet` | Train parquet file |
| `VAL_FILE` | `data/compact_sft_data/compact_val.parquet` | Val parquet file |
| `BASE_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | Pretrained base model name or path |
| `OUTPUT_DIR` | `outputs/baseline_ckpt` | Checkpoint output directory |
| `TB_LOG_DIR` | `outputs/baseline_tb` | TensorBoard log directory |
| `GPU_IDS` | (empty = all) | Comma-separated GPU IDs (e.g. `6,7`) |
| `MAX_LENGTH` | `8192` | Max token length |
| `MAX_EPOCHS` | `2` | Number of training epochs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `LR` | `2e-5` | Learning rate |
| `LOG_STEPS` | `10` | TensorBoard logging frequency |
| `SAVE_STEPS` | `500` | Checkpoint save frequency |
| `VAL_STEPS` | `200` | Validation frequency |

### Baseline Evaluation (`run_baseline_eval.sh`)

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT` | `outputs/baseline_ckpt/final` | Model checkpoint directory |
| `APPS_MODE` | `test10` | App split: `test10`, `train86`, or `all` |
| `APPS` | (empty) | Explicit app list (overrides `APPS_MODE`) |
| `SCALEWOB_ROOT` | `env/browser_env/scalewob-env` | ScaleWoB static files directory |
| `MAX_STEPS` | `50` | Steps per session |
| `NUM_SESSIONS` | `3` | Sessions per app |
| `EVAL_OUTPUT` | `outputs/baseline_eval` | Evaluation output directory |
| `DEVICE` | `cuda` | Device for inference |
| `TEMPERATURE` | `0.8` | Sampling temperature |
| `TOP_P` | `0.9` | Top-p sampling |
| `GPU_IDS` | (empty = all) | GPU ID(s) for eval |

### Video Generation (`scripts/snapshots_to_mp4.py`)

| Argument | Default | Description |
|---|---|---|
| `session_dir` | (required) | Directory containing `step_XXX_before.png` / `step_XXX_after.png` |
| `-o` / `--output` | `<session_dir>/eval_video.mp4` | Output MP4 path |
| `--mode` | `sequential` | Frame arrangement: `sequential`, `side_by_side`, `before_only`, `after_only` |
| `--fps` | `2.0` | Frames per second |
| `--codec` | `mp4v` | FourCC codec |

---

## Parameter Overhead Summary

| Base Model | $d$ | Layers | Per-layer | Total new | Base | Overhead |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 1536 | 28 | 8.41M | 237.9M | ~2.0B | **11.9%** |
| Qwen3-VL-8B | 4096 | 36 | 13.66M | 508.7M | ~8.0B | **6.4%** |

The overhead **decreases** for larger base models because the projection layers (which scale with $d \times d_{mem}$) become a smaller fraction of the total.
