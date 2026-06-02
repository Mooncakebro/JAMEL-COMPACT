# Training

JAMEL training has two stages:

1. Convert exploration trajectories into memory-augmented SFT parquet files.
2. Train `MemoryAugmentedCausalLM` with the bundled `third_party/verl-agent`
   FSDP SFT trainer.

## Key Code

```text
jamel/train/memory/prepare_sft_dataset.py
jamel/train/memory/jamel_sft_dataset.py
jamel/train/memory/modeling.py
jamel/train/memory/encoder.py
jamel/train/memory/web_prompt.py
shell/run_qwen25vl_7b_sft.sh
third_party/verl-agent/verl/trainer/fsdp_sft_trainer.py
```

## Hardware

The full 7B multimodal SFT recipe was designed for 8 x 80GB GPUs. Smaller
experiments can use fewer GPUs by changing the `torchrun --nproc_per_node` line
in `shell/run_qwen25vl_7b_sft.sh`, but batch size, gradient checkpointing, and
maximum length may need adjustment.

## Prepare Data

Input trajectories are parquet files with screenshots, observations, actions,
rewards, and coverage artifacts. The preparation step recomputes the canonical
JAMEL prompt and precomputes latent memory tokens from previous session steps.

```bash
cd JAMEL

uv run python jamel/train/memory/prepare_sft_dataset.py \
  --input /path/to/trajectory.parquet \
  --output data/jamel_sft_data \
  --compressor-model /path/to/Qwen3-VL-2B-Instruct \
  --max-memory-items 512 \
  --max-length 8192 \
  --val-ratio 0.02 \
  --compression-batch-size 4
```

Output files:

```text
data/jamel_sft_data/jamel_memory_sft_train.parquet
data/jamel_sft_data/jamel_memory_sft_val.parquet
```

## Run SFT

```bash
TRAIN_FILE=data/jamel_sft_data/jamel_memory_sft_train.parquet \
VAL_FILE=data/jamel_sft_data/jamel_memory_sft_val.parquet \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=outputs/jamel_sft_ckpt \
TOTAL_EPOCHS=3 \
VAL_STEPS=200 \
bash shell/run_qwen25vl_7b_sft.sh
```

Important defaults:

```text
memory_max_items = 512
memory_hidden_size = 2048
max_length = 8192
model input image = 640 x 360
browser viewport = 1280 x 720
response format = <action>...</action>
```

The training script sets `PYTHONPATH` to include both this repository and
`third_party/verl-agent`.

## Small Smoke Training

For a single-app smoke run, use:

```bash
INPUT_PARQUET=/path/to/react-vision/youdao/trajectory.parquet \
DATASET_DIR=data/jamel_youdao_sft \
OUTPUT_DIR=outputs/jamel_youdao_ckpt \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
bash shell/run_youdao_sft.sh
```
