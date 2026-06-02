# Training

Start from the environment setup in the README. Training uses the same Python
environment as evaluation, plus `third_party/verl-agent/requirements.txt`.

## 1. Prepare SFT data

```bash
cd JAMEL

python jamel/train/memory/prepare_sft_dataset.py \
  --input /path/to/trajectory.parquet \
  --output data/jamel_sft_data \
  --compressor-model /path/to/Qwen3-VL-2B-Instruct \
  --max-memory-items 512 \
  --max-length 8192 \
  --val-ratio 0.02 \
  --compression-batch-size 4
```

This writes:

```text
data/jamel_sft_data/jamel_memory_sft_train.parquet
data/jamel_sft_data/jamel_memory_sft_val.parquet
```

## 2. Train and package the model

```bash
TRAIN_FILE=data/jamel_sft_data/jamel_memory_sft_train.parquet \
VAL_FILE=data/jamel_sft_data/jamel_memory_sft_val.parquet \
BASE_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
OUTPUT_DIR=outputs/jamel_sft_ckpt \
OUTPUT_MODEL_PATH=outputs/jamel_model \
NPROC_PER_NODE=8 \
TOTAL_EPOCHS=1 \
VAL_STEPS=200 \
bash shell/run_qwen25vl_7b_sft.sh
```

`OUTPUT_DIR` stores actor checkpoints such as `global_step_*`.
`NPROC_PER_NODE` is the number of training GPUs used by `torchrun`.
`OUTPUT_MODEL_PATH` stores the final JAMEL model used by evaluation:

```text
outputs/jamel_model/
  actor/
  compressor/
  model.json
```

Use `OUTPUT_MODEL_PATH` as `MODEL_PATH` in [EVALUATION.md](EVALUATION.md).
