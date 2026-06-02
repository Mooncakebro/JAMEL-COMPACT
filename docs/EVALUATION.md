# Evaluation

Start from the environment setup in the README. Evaluation needs a ScaleWoB
directory at `$SCALEWOB_ROOT` and a JAMEL model path set through `MODEL_PATH`.
`MODEL_PATH` should point to a directory containing `actor/` and `compressor/`.

Chinese apps require system CJK fonts. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y fontconfig fonts-noto-cjk fonts-noto-color-emoji
fc-cache -fv
```

## test10

```bash
MODEL_PATH=/path/to/jamel_model \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=1 \
NUM_GPUS=4 \
WORKERS_PER_GPU=1 \
EVAL_OUTPUT=outputs/eval_test10 \
bash shell/run_eval.sh
```

`APPS_MODE=test10` is the paper evaluation setting and is also the default.
The `train86` and `test10` app split is defined in `configs/benchmark_apps.json`.
Use `APPS=...` for a custom app list; `APPS_MODE` only accepts `test10`,
`train86`, or `all`.

## train86

```bash
MODEL_PATH=/path/to/jamel_model \
APPS_MODE=train86 \
MAX_STEPS=50 \
NUM_SESSIONS=1 \
EVAL_OUTPUT=outputs/eval_train86 \
bash shell/run_eval.sh
```

Use this for training-set checks, not as the held-out metric.

## Single app debug

```bash
MODEL_PATH=/path/to/jamel_model \
APPS=weibo \
NUM_GPUS=1 \
WORKERS_PER_GPU=1 \
MAX_STEPS=20 \
EVAL_OUTPUT=outputs/eval_weibo_debug \
bash shell/run_eval.sh
```

## Outputs

`EVAL_OUTPUT` contains:

```text
trajectory.parquet
summary.json
coverage/coverage_*.json
reward_curve.*
worker_*.log
```
