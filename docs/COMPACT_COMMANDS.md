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

> **v2 defaults**: `CHUNK_SIZE=8`, `FREEZE_BASE=1`, `MEMORY_LR=5e-6`, and
> `COVERAGE_WEIGHT_ETA=0.0`. The pretrained backbone is frozen unless
> full-model fine-tuning is explicitly enabled.

### Choosing a training mode

The exact shell variable is `FREEZE_BASE` (not `FREEZE_BASEMODEL`). The
following settings are the supported combinations:

| Goal | `FREEZE_BASE` | LoRA | `MODEL_PARALLEL` | `FSDP` | Notes |
|---|---:|---:|---|---|---|
| Train only COMPACT memory/action modules | `1` | disabled (`LORA_RANK=0`) | `auto` or `1` for large models | `0`/`auto` | Recommended low-cost mode |
| Train memory modules plus base LoRA adapters | `1` | enabled (`LORA_RANK>0`, `LORA_ALPHA>0`) | `auto` or `1` for large models | `0`/`auto` | Dense base remains frozen |
| Dense full SFT plus COMPACT modules | `0` | disabled (`LORA_RANK=0`) | `0`/ignored | `auto` or `1` on multiple GPUs | Use FSDP for 4B/8B models |
| Debug or small-model single-GPU training | `1` or `0` | optional | `0` | `0` | No sharding is needed |

Definitions:

- `FREEZE_BASE=1` freezes the pretrained Qwen weights. COMPACT side-memory
  and action modules still train; LoRA adapters also train if enabled.
- `FREEZE_BASE=0` trains the dense pretrained model and COMPACT modules. LoRA
  is rejected in this mode because LoRA and dense base SFT are alternatives.
- `LORA_RANK=0` disables LoRA. To enable it, set both `LORA_RANK>0` and
  `LORA_ALPHA>0`; keep `FREEZE_BASE=1`.
- `MODEL_PARALLEL=1` loads a frozen base model with a Transformers/Accelerate
  device map across GPUs inside one process. It is useful for large frozen
  models and cannot be combined with FSDP.
- `MODEL_PARALLEL=0` selects legacy batch-splitting DataParallel. It is only
  appropriate for small models or `CHUNK_SIZE=1`; chunked `BATCH_SIZE=1`
  cannot be usefully split and leaves only the first GPU doing the work.
- `FSDP=1` launches one torchrun process per selected GPU and shards dense
  parameters, gradients, and optimizer state. `FSDP=auto` enables this only
  for `FREEZE_BASE=0` with multiple GPUs. FSDP disables model parallelism.

Use `GRAD_ACCUM` to increase the effective batch without increasing the
per-GPU activation memory. When `CHUNK_SIZE>1`, the launcher forces
`BATCH_SIZE=1` because one batch is one recurrent session chunk.

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
MEMORY_LR=5e-6 \
FREEZE_BASE=1 \
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
MEMORY_LR=5e-6 \
FREEZE_BASE=1 \
CHUNK_SIZE=8 \
COVERAGE_WEIGHT_ETA=0.0 \
LOG_STEPS=10 \
SAVE_STEPS=500 \
VAL_STEPS=200 \
bash shell/run_compact_train.sh
```

### 2c. Train with Qwen3-VL-8B (model-sharded across GPUs)

```bash
TRAIN_FILE=data/compact_sft_data_all/compact_train.parquet \
VAL_FILE=data/compact_sft_data_all/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b \
TB_LOG_DIR=outputs/compact_tb_8b \
GPU_IDS=5,6,7 \
MODEL_PARALLEL=1 \
FREEZE_BASE=1 \
MEM_DIM=512 \
NUM_MEM=16 \
MAX_LENGTH=8192 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=32 \
LR=1e-5 \
MEMORY_LR=5e-6 \
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

### 2e. Full-SFT Qwen3-VL-8B across multiple GPUs with FSDP

```bash
TRAIN_FILE=data/compact_sft_data_all/compact_train.parquet \
VAL_FILE=data/compact_sft_data_all/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b_full_sft \
TB_LOG_DIR=outputs/compact_tb_8b_full_sft \
GPU_IDS=5,6,7 \
FREEZE_BASE=0 \
FSDP=1 \
NPROC_PER_NODE=3 \
MAX_LENGTH=4096 \
CHUNK_SIZE=8 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=1e-5 \
MEMORY_LR=5e-6 \
VAL_STEPS=100 \
SAVE_STEPS=100 \
bash shell/run_compact_train.sh
```

`FREEZE_BASE=0` with multiple selected GPUs automatically enables FSDP when
`FSDP=auto`. FSDP runs one process per GPU, shards parameters, gradients, and
optimizer state, and synchronizes validation/checkpointing across ranks.

### 2f. Train COMPACT with Qwen3-VL-4B using LoRA

`LORA_RANK=0` is the default and preserves the existing frozen-base or full-SFT
behavior. LoRA is enabled only when `LORA_RANK>0` and `LORA_ALPHA>0` are both
set. COMPACT LoRA also requires `FREEZE_BASE=1`: dense base weights stay frozen,
while LoRA adapters and the COMPACT side-memory modules are trained.

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_4b_lora \
TB_LOG_DIR=outputs/compact_tb_4b_lora \
GPU_IDS=5,6 \
FREEZE_BASE=1 \
MODEL_PARALLEL=auto \
FSDP=0 \
MAX_LENGTH=4096 \
CHUNK_SIZE=8 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=1e-4 \
MEMORY_LR=5e-6 \
LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
LORA_BIAS=none \
bash shell/run_compact_train.sh
```

### 2g. Train COMPACT with Qwen3-VL-8B using LoRA and model sharding

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_8b_lora \
TB_LOG_DIR=outputs/compact_tb_8b_lora \
GPU_IDS=5,6,7 \
FREEZE_BASE=1 \
MODEL_PARALLEL=1 \
FSDP=0 \
MAX_LENGTH=4096 \
CHUNK_SIZE=8 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=1e-4 \
MEMORY_LR=5e-6 \
LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
LORA_BIAS=none \
bash shell/run_compact_train.sh
```

### 2h. Opt into full-model fine-tuning

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/compact_ckpt_full \
TB_LOG_DIR=outputs/compact_tb_full \
GPU_IDS=0 \
FREEZE_BASE=0 \
LR=1e-5 \
MEMORY_LR=5e-6 \
CHUNK_SIZE=8 \
bash shell/run_compact_train.sh
```

### 2i. Monitor training with TensorBoard

Open a separate terminal:

```bash
tensorboard --logdir outputs/compact_tb_v2 --port 6006
# Open http://localhost:6006 in browser
```

**Logged metrics (v2):**
- `train/loss_total`, `train/loss_action`, `train/loss_action_unweighted`, `train/loss_obs`, `train/loss_nll`, `train/loss_mem_l2`
- `memory/layer{l}_gate`, `memory/layer{l}_lambda_mean`, `memory/layer{l}_Q_mean`, `memory/layer{l}_R_mean`, `memory/layer{l}_init_mem_norm`
- `train/memory_learning_rate`, optional `train/base_learning_rate`, `train/step_time_s`
- `val/loss_total`, `val/loss_action`, `val/loss_obs`, `val/loss_nll`

> **v2 loss keys**: `loss_obs` (observation prediction MSE), `loss_nll` (Gaussian NLL for Kalman variance calibration), `loss_action_unweighted` (unweighted CE for comparison against coverage-weighted `loss_action`).

**Output:**
```
outputs/compact_ckpt/
├── global_step_500/
│   ├── base_model/       (full SFT only)
│   ├── base_model_ref.txt (frozen base or LoRA)
│   ├── base_model_lora/  (LoRA adapter only, when enabled)
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

### 2B-c. Train baseline Qwen3-VL-4B or 8B using LoRA

The baseline has no side-memory modules. With LoRA enabled, only the adapter
parameters are optimized and each checkpoint stores the adapter plus tokenizer
and processor files; evaluation automatically reloads the referenced base model.

```bash
# Use Qwen/Qwen3-VL-8B-Instruct here for the 8B run.
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=outputs/baseline_ckpt_4b_lora \
TB_LOG_DIR=outputs/baseline_tb_4b_lora \
GPU_IDS=5 \
MAX_LENGTH=4096 \
MAX_EPOCHS=3 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=1e-4 \
LORA_RANK=16 \
LORA_ALPHA=32 \
LORA_DROPOUT=0.05 \
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
LORA_BIAS=none \
bash shell/run_baseline_train.sh
```

### 2B-d. Monitor baseline training with TensorBoard

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

Set `GPU_IDS` only once. Inside Python, `cuda:0` maps to the first physical
GPU in that list; for example, `GPU_IDS=1` means Python `cuda:0` is physical
GPU 1. If a shell command assigns `GPU_IDS` twice, the last assignment wins.

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

### 3g. Render complete VLM inputs and outputs for debugging

Each trajectory row stores the canonical user prompt (including the pruned
AXTree), raw response, parsed action, and reward. New evaluations also store the
exact text after `apply_chat_template()`, token/truncation metadata, chat roles,
the previous action, and COMPACT memory/variance/surprise summaries. Render one
complete, tall PNG per step:

```bash
python scripts/render_eval_debug.py \
    outputs/compact_eval_debug/weibo/session0
```

Create the PNGs plus a readable scrolling video:

```bash
python scripts/render_eval_debug.py \
    outputs/compact_eval_debug/weibo/session0 \
    --video outputs/compact_eval_debug/weibo/session0/vlm_debug.mp4 \
    --seconds-per-step 8
```

For future evaluations, enable screenshots so the debug view contains both the
VLM image input and the browser result after its action:

```bash
SAVE_SCREENSHOTS=1 \
SAVE_VLM_DEBUG=1 \
EVAL_OUTPUT=outputs/compact_eval_debug \
bash shell/run_compact_eval.sh
```

Older trajectories remain supported. They contain the canonical user prompt and
response, but not the exact rendered chat-template text, token counts, previous
action, or recurrent-state summaries. Runs without `SAVE_SCREENSHOTS=1` render a
placeholder because the original image cannot be reconstructed afterward.

### 3h. Convert eval screenshots to a simple MP4 video

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

# ── 2c. Full-SFT compact 8B with FSDP ──
TRAIN_FILE=data/compact_train.parquet \
VAL_FILE=data/compact_val.parquet \
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct \
OUTPUT_DIR=outputs/compact_8b_full_sft \
TB_LOG_DIR=outputs/compact_8b_full_sft_tb \
GPU_IDS=5,6,7 \
FREEZE_BASE=0 \
FSDP=1 \
NPROC_PER_NODE=3 \
MAX_LENGTH=4096 \
CHUNK_SIZE=8 \
BATCH_SIZE=1 \
GRAD_ACCUM=16 \
LR=1e-5 \
MEMORY_LR=5e-6 \
VAL_STEPS=100 \
SAVE_STEPS=100 \
bash shell/run_compact_train.sh


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
| `BATCH_SIZE` | `1` | Per-device batch size in single-step mode; fixed to `1` when `CHUNK_SIZE>1` because each batch is one recurrent session chunk |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps; use this to increase effective batch size in chunked mode |
| `LR` | `2e-5` | Dense base-model learning rate for full SFT, or LoRA-adapter learning rate when LoRA is enabled |
| `MEMORY_LR` | `5e-6` | Learning rate for side-memory and action modules |
| `FREEZE_BASE` | `1` | `1` passes `--freeze-base`; `0` passes `--train-base` for dense full SFT |
| `LORA_RANK` | `0` | LoRA rank; `0` disables LoRA and preserves existing behavior |
| `LORA_ALPHA` | `0` | LoRA scaling alpha; must be greater than `0` when `LORA_RANK>0` |
| `LORA_DROPOUT` | `0.0` | Dropout applied in LoRA branches |
| `LORA_TARGET_MODULES` | `q_proj,...,down_proj` | Comma-separated target suffixes, or `all-linear` |
| `LORA_BIAS` | `none` | PEFT bias mode: `none`, `all`, or `lora_only` |
| `MODEL_PARALLEL` | `auto` | `auto` shards frozen chunked B=1 runs; `1` forces model sharding; `0` forces legacy DataParallel |
| `FSDP` | `auto` | `auto` enables torchrun `FULL_SHARD` only for dense multi-GPU SFT (`FREEZE_BASE=0`); `1` forces it |
| `NPROC_PER_NODE` | selected GPU count | Number of FSDP worker processes; normally one per selected GPU |
| `LOG_STEPS` | `10` | TensorBoard logging frequency |
| `SAVE_STEPS` | `500` | Checkpoint save frequency |
| `VAL_STEPS` | `200` | Validation frequency |
| `CHUNK_SIZE` | `8` | Stateful TBPTT length: gradients connect within each ordered chunk; detached memory/variance/surprise carry into the next chunk of the same session |
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
| `GPU_IDS` | (empty = all) | GPU ID(s) for eval (e.g. `0` or `0,1`); the selected IDs are applied before PyTorch loads |
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
| `FSDP` | `auto` | `auto`/`1` uses torchrun FSDP FULL_SHARD on multiple GPUs; `0` keeps legacy DataParallel |
| `NPROC_PER_NODE` | selected GPU count | Number of FSDP processes; normally one per selected GPU |
| `MAX_LENGTH` | `8192` | Max token length |
| `MAX_EPOCHS` | `2` | Number of training epochs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `LR` | `2e-5` | Learning rate |
| `LORA_RANK` | `0` | LoRA rank; `0` keeps standard full SFT |
| `LORA_ALPHA` | `0` | LoRA scaling alpha; required when rank is enabled |
| `LORA_DROPOUT` | `0.0` | Dropout applied in LoRA branches |
| `LORA_TARGET_MODULES` | `q_proj,...,down_proj` | Comma-separated target suffixes, or `all-linear` |
| `LORA_BIAS` | `none` | PEFT bias mode: `none`, `all`, or `lora_only` |
| `LOG_STEPS` | `10` | TensorBoard logging frequency |
| `SAVE_STEPS` | `500` | Checkpoint save frequency |
| `VAL_STEPS` | `200` | Validation frequency |

For full SFT of a large baseline model, use FSDP rather than legacy
DataParallel. FSDP shards parameters, gradients, and optimizer state across
the selected GPUs:

```bash
TRAIN_FILE=data/compact_sft_data/compact_train.parquet \
VAL_FILE=data/compact_sft_data/compact_val.parquet \
BASE_MODEL=/path/to/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=outputs/baseline_4b \
TB_LOG_DIR=outputs/baseline_4b_tb \
GPU_IDS=0,1,2,3,4,5,6,7 \
FSDP=1 \
BATCH_SIZE=1 \
GRAD_ACCUM=2 \
MAX_LENGTH=8192 \
bash shell/run_baseline_train.sh
```

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
