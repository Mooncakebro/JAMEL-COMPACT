# Evaluation

JAMEL evaluation runs continuous browser sessions on ScaleWoB apps and reports
cumulative coverage reward. The paper metric uses `test10` with 50 steps per app.
Download the ScaleWoB static files first with
`bash shell/download_scalewob_env.sh --mode all`, or set `SCALEWOB_ROOT` to an
existing mirror.

## Key Code

```text
jamel/utils/eval/eval_memory_aug_episode.py
jamel/cli/baseline_eval.py
shell/run_eval.sh
```

## JAMEL on test10

```bash
cd JAMEL

CHECKPOINT=/path/to/jamel_checkpoint \
JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
APPS_MODE=test10 \
MAX_STEPS=50 \
NUM_SESSIONS=3 \
NUM_GPUS=4 \
WORKERS_PER_GPU=1 \
EVAL_OUTPUT=outputs/eval_test10 \
bash shell/run_eval.sh
```

`APPS_MODE=test10` is the default.
`JAMEL_BASE_MODEL` is only needed when the checkpoint metadata points to a base
model path that does not exist on the current machine.

## JAMEL on train86

```bash
CHECKPOINT=/path/to/jamel_checkpoint \
JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
APPS_MODE=train86 \
MAX_STEPS=50 \
NUM_SESSIONS=1 \
EVAL_OUTPUT=outputs/eval_train86 \
bash shell/run_eval.sh
```

Use train86 runs for debugging or training-set sanity checks, not as the
held-out test metric.

## Single-App Debug

```bash
CHECKPOINT=/path/to/jamel_checkpoint \
JAMEL_BASE_MODEL=/path/to/base-model-used-by-checkpoint \
COMPRESSOR_MODEL=/path/to/Qwen3-VL-2B-Instruct \
APPS=weibo \
NUM_GPUS=1 \
WORKERS_PER_GPU=1 \
MAX_STEPS=20 \
EVAL_OUTPUT=outputs/eval_weibo_debug \
bash shell/run_eval.sh
```

## Important Settings

```text
temperature = 0.8
top_p = 0.9
memory_max_items = 512
viewport = 1280 x 720
model image = 640 x 360
```

Sampling is part of the intended eval recipe; greedy decoding tends to repeat
actions in loops.

## Outputs

Per app/session output includes:

```text
trajectory.parquet
summary.json
coverage/coverage_*.json
reward_curve.*
worker_*.log
```

The trajectory parquet stores prompt, response, parsed action, screenshots,
AXTree/DOM observations, reward, coverage delta, coverage checksum, and session
metadata.

## Baselines

OpenAI-compatible baseline evaluation is exposed through:

```bash
jamel baseline-eval --help
```

The helper scripts `shell/run_gemini31_react_10apps_50.sh` and
`shell/run_qwen3vl8b_react_10apps_50.sh` run ReAct-style baselines on `test10`.
