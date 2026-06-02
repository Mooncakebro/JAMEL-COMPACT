#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${QWEN3VL8B_MODEL_PATH:-"$ROOT/models/Qwen3-VL-8B-Instruct"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3-vl-8b-instruct}
PORT=${VLLM_PORT:-4010}
API_KEY=${VLLM_API_KEY:-local-qwen3vl8b}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image": 1}'
