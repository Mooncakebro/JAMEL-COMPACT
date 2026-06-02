# Setup

JAMEL can be configured directly on a host machine. Docker is useful as a
reference environment, but the release should be understandable without requiring
users to build an image.

## Prerequisites

Host requirements:

```text
Python >= 3.10
uv
Node.js >= 20 with npm
CUDA-capable NVIDIA driver, if running model inference/training on GPU
Enough disk for model caches, ScaleWoB static files, datasets, and outputs
```

## uv Python Environment

Install `uv` with your package manager or the official installer, then create
and synchronize the project environment:

```bash
cd JAMEL

bash shell/setup_env.sh
source .venv/bin/activate
```

If Python 3.10 is not available locally, install it with:

```bash
uv python install 3.10
```

`shell/setup_env.sh` is a thin wrapper around the checked-in dependency files:
`pyproject.toml` declares dependencies and `uv.lock` pins the resolved versions.
It runs `uv sync --locked --python 3.10 --extra dev --extra train`, creates
`.venv`, installs JAMEL in editable mode, installs Playwright Chromium, installs
the Node coverage tools, and prints a runtime check.

If you prefer to run the steps manually:

```bash
uv sync --locked --python 3.10 --extra dev --extra train
uv run playwright install chromium
npm install -g monocart-coverage-reports monocart-locator
```

On Linux, Playwright may also need system browser libraries. If Chromium fails
to launch, run:

```bash
uv run playwright install --with-deps chromium
```

or install the missing libraries with your system package manager.

For full `verl-agent` training dependencies, install:

```bash
uv pip install -r third_party/verl-agent/requirements.txt
```

If `flash-attn` fails to build, install a wheel matching your CUDA/PyTorch stack
or continue with the common runtime dependencies for evaluation and smoke tests.

The checked-in `uv.lock` uses PyTorch CUDA 12.8 wheels on Linux/Windows. This
matches systems whose CUDA driver API reports `12080`, for example NVIDIA
driver 570.x. If an existing `.venv` was created before this lockfile change,
rebuild it with:

```bash
rm -rf .venv
uv sync --locked --python 3.10 --extra dev --extra train
source .venv/bin/activate
```

For GPU training on a different driver stack, use
[docker/jamel/Dockerfile](../docker/jamel/Dockerfile) as the reference for the
CUDA/PyTorch combination. To replace the PyTorch wheels in the synchronized
`.venv`, activate it and select the matching PyTorch backend. For example, for
CUDA 12.8:

```bash
source .venv/bin/activate
uv pip install --reinstall torch torchvision --torch-backend cu128
```

After replacing synchronized wheels manually, avoid running `uv sync` again
unless you intend to restore the locked dependency set. Use `uv run --no-sync`
for ad hoc Python commands that should preserve the manual override.

## Node Tools

JAMEL uses Monocart tooling for JavaScript coverage artifacts:

```bash
npm install -g monocart-coverage-reports monocart-locator
```

If global npm installs are not allowed on your machine, configure an npm prefix
or install these tools in a local Node environment and make sure the executables
are on `PATH`.

## Environment Variables

Set these from the repository root:

```bash
export JAMEL_ROOT=$PWD
export VERL_AGENT_ROOT=$PWD/third_party/verl-agent
export SCALEWOB_ROOT=$PWD/env/browser_env/scalewob-env
export PYTHONPATH=$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH:-}
```

Offline model-cache flags are optional:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
```

## Browser Environment

Download or mirror ScaleWoB into `env/browser_env/scalewob-env`:

```bash
bash shell/download_scalewob_env.sh --mode all
```

If the endpoint requires login, use Playwright storage state or cookies as
shown in [ENVIRONMENT.md](ENVIRONMENT.md).

Serve the static apps:

```bash
bash shell/serve_scalewob.sh
# open http://127.0.0.1:8000/weibo/index.html
```

## Smoke Checks

Run:

```bash
uv run pytest -q tests/test_baseline_eval.py tests/test_weak_model_sft_augment.py tests/test_weak_model_sft_labeling.py
uv run python scripts/print_app_split.py test10
uv run playwright --version
NODE_PATH=$(npm root -g) node -e "require('monocart-coverage-reports'); console.log('monocart ok')"
```

Expected split sizes:

```text
test10 = 10 apps
train86 = 86 apps
all = 96 apps
```

## Model Paths

Weights are not included in the repo. Put downloaded models in a local cache or
any host path and pass explicit paths:

```bash
export CHECKPOINT=/path/to/jamel-checkpoint
export JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint
export COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
```

`JAMEL_BASE_MODEL` is optional for freshly exported checkpoints. Set it when a
checkpoint's `memory_augment_config.json` records a stale absolute path from a
different machine.

Evaluation example:

```bash
CHECKPOINT=/path/to/jamel-checkpoint \
JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
APPS_MODE=test10 \
NUM_GPUS=1 WORKERS_PER_GPU=1 \
EVAL_OUTPUT=outputs/eval_test10 \
bash shell/run_eval.sh
```

SFT example:

```bash
TRAIN_FILE=data/jamel_sft_data/jamel_memory_sft_train.parquet \
VAL_FILE=data/jamel_sft_data/jamel_memory_sft_val.parquet \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=outputs/jamel_sft_ckpt \
bash shell/run_qwen25vl_7b_sft.sh
```

## Docker Reference

The Docker setup is not required for the quick start. If you want a reproducible
environment recipe, use [docker/jamel/Dockerfile](../docker/jamel/Dockerfile) as
the reference for exact system packages and `uv`/Node installation order.

Convenience wrappers are provided:

```bash
bash docker/jamel/build.sh
bash docker/jamel/run.sh
```

Set `INSTALL_TRAIN_DEPS=1` when building if you want the image to install the
full `third_party/verl-agent/requirements.txt` stack.

## Notes

- Keep generated `outputs/`, downloaded weights, and parquet datasets outside
  source control.
- Full training still depends on GPU memory and exact model checkpoints.
- If local dependency resolution differs from Docker, compare your environment
  against `docker/jamel/Dockerfile`.
