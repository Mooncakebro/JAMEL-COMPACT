# JAMEL: Joint Agent Memory and Exploration Learning via Novelty Signals

[![Paper](https://img.shields.io/badge/arXiv-2606.01528-red)](https://arxiv.org/abs/2606.01528)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20TBD-FFD21E)](https://huggingface.co)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data%20TBD-FFD21E)](https://huggingface.co)

![arch](docs/arch.png)

This repository implements the core ideas of [Joint Agent Memory and Exploration Learning via Novelty Signals](https://arxiv.org/abs/2606.01528). In open-ended environments, agents often struggle to explore effectively, especially when external task rewards are sparse or unavailable. A central reason is that effective exploration requires memory. Without remembering what has already been tried, an agent can easily repeat exhausted behaviors instead of discovering new states.

The paper proposes **JAMEL** — **Joint Agent Memory and Exploration Learning** — a framework that trains agent memory and exploration policy together. Its main insight is that memory and exploration form a mutually reinforcing loop: memory helps the agent avoid repeated behaviors and identify unexplored actions, while novelty-driven exploration provides supervision for what the memory should encode. Instead of relying on manually annotated memory labels, JAMEL uses persistent novelty signals, such as JavaScript code coverage in GUI environments, as an intrinsic reward for discovering new behavior. Experiments on the [ScaleWoB](https://github.com/ScaleWoB/ScaleWoB) GUI benchmark show that JAMEL generalizes to unseen applications.

## Quick Start

Create and synchronize the local Python environment with `uv`:

```bash
cd JAMEL

bash shell/setup_env.sh
source .venv/bin/activate
```

Install the Node.js coverage tools used by browser evaluation:

```bash
npm install -g monocart-coverage-reports monocart-locator
```

Set local paths:

```bash
export JAMEL_ROOT=$PWD
export VERL_AGENT_ROOT=$PWD/third_party/verl-agent
export SCALEWOB_ROOT=$PWD/env/browser_env/scalewob-env
export PYTHONPATH=$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH:-}
```

Download and serve the browser environment:

```bash
bash shell/download_scalewob_env.sh --mode all
bash shell/serve_scalewob.sh
# Open http://127.0.0.1:8000/expedia/index.html
```

## Evaluation

Run JAMEL evaluation on `test10`:

```bash
CHECKPOINT=/path/to/jamel_checkpoint \
JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
NUM_GPUS=4 WORKERS_PER_GPU=1 \
EVAL_OUTPUT=outputs/eval_test10 \
bash shell/run_eval.sh
```

### App Split

The release uses the paper split:

- `train86`: 86 training apps.
- `test10`: `vipshop alibaba expedia taobao pinduoduo dongchedi youku keep meituan temu`.

See [docs/APP_SPLIT.md](docs/APP_SPLIT.md) and
[configs/benchmark_apps.json](configs/benchmark_apps.json).

## TRAINING

Prepare SFT data and train:

```bash
uv run python jamel/train/memory/prepare_sft_dataset.py \
  --input /path/to/trajectory.parquet \
  --output data/jamel_sft_data \
  --compressor-model /path/to/Qwen3-VL-2B-Instruct \
  --max-memory-items 512 \
  --max-length 8192 \
  --val-ratio 0.02

TRAIN_FILE=data/jamel_sft_data/jamel_memory_sft_train.parquet \
VAL_FILE=data/jamel_sft_data/jamel_memory_sft_val.parquet \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=outputs/jamel_sft_ckpt \
bash shell/run_qwen25vl_7b_sft.sh
```

## Documentation

- [Setup](docs/SETUP.md)
- [Environment Layout](docs/ENVIRONMENTS.md)
- [Browser Environment](docs/ENVIRONMENT.md)
- [Training](docs/TRAINING.md)
- [Evaluation](docs/EVALUATION.md)
- [Models and Data](docs/MODELS.md)
- [App Split](docs/APP_SPLIT.md)
