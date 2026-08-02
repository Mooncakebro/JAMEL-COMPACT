#!/bin/bash
# run_baseline_eval.sh — Evaluate pure Qwen3-VL SFT baseline on ScaleWoB
#
# Usage:
#   CHECKPOINT=outputs/baseline_ckpt/final \
#   APPS_MODE=test10 \
#   MAX_STEPS=50 \
#   NUM_SESSIONS=3 \
#   EVAL_OUTPUT=outputs/baseline_eval \
#   bash shell/run_baseline_eval.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JAMEL_ROOT=${JAMEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PYTHONPATH="$JAMEL_ROOT:${PYTHONPATH:-}"

CHECKPOINT=${CHECKPOINT:-outputs/baseline_ckpt/final}
APPS_MODE=${APPS_MODE:-test10}
APPS=${APPS:-}
SCALEWOB_ROOT=${SCALEWOB_ROOT:-$JAMEL_ROOT/env/browser_env/scalewob-env}
MAX_STEPS=${MAX_STEPS:-50}
NUM_SESSIONS=${NUM_SESSIONS:-3}
EVAL_OUTPUT=${EVAL_OUTPUT:-outputs/baseline_eval}
DEVICE=${DEVICE:-cuda}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.9}
GPU_IDS=${GPU_IDS:-}              # e.g. "0" or "1" (empty = all)
BROWSER_TIMEOUT_MS=${BROWSER_TIMEOUT_MS:-30000}
RESET_RETRIES=${RESET_RETRIES:-3}
MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
SAVE_SCREENSHOTS=${SAVE_SCREENSHOTS:-0}
PORT=${PORT:-0}                  # 0 = automatically reserve a free port
RESUME=${RESUME:-1}              # 1 = skip completed app/session entries

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
if [[ ! -d "$SCALEWOB_ROOT" ]]; then
    echo "ERROR: ScaleWoB root not found: $SCALEWOB_ROOT" >&2
    echo "Run: python scripts/download_scalewob_env.py" >&2
    exit 2
fi

EXTRA_ARGS=()
if [[ -n "$APPS" ]]; then
    EXTRA_ARGS+=(--apps "$APPS")
else
    EXTRA_ARGS+=(--apps-mode "$APPS_MODE")
fi
OPTIONAL_ARGS=()
if [[ "$SAVE_SCREENSHOTS" == "1" ]]; then
    OPTIONAL_ARGS+=(--save-screenshots)
fi
if [[ "$RESUME" != "1" ]]; then
    OPTIONAL_ARGS+=(--no-resume)
fi

echo "=== Baseline Qwen3-VL SFT Evaluation ==="
echo "  Checkpoint:   $CHECKPOINT"
echo "  Apps mode:    $APPS_MODE"
echo "  ScaleWoB:     $SCALEWOB_ROOT"
echo "  Max steps:    $MAX_STEPS"
echo "  Sessions:     $NUM_SESSIONS"
echo "  Output:       $EVAL_OUTPUT"
echo "  GPU:          ${GPU_IDS:-all}"
echo "  Reset retries: $RESET_RETRIES"
echo "  Browser ms:   $BROWSER_TIMEOUT_MS"
echo "  Input tokens: $MAX_INPUT_TOKENS"
echo "  New tokens:   $MAX_NEW_TOKENS"
echo "  Port:         $([[ "$PORT" == "0" ]] && echo auto || echo "$PORT")"
echo "  Resume:       $RESUME"
echo ""

# Set CUDA_VISIBLE_DEVICES in the shell BEFORE Python launches.
if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "  CUDA mapping: Python cuda:0 -> physical GPU ${GPU_IDS%%,*}"
fi

python -m jamel_compact.baseline_eval \
    --checkpoint "$CHECKPOINT" \
    "${EXTRA_ARGS[@]}" \
    --scalewob-root "$SCALEWOB_ROOT" \
    --max-steps "$MAX_STEPS" \
    --num-sessions "$NUM_SESSIONS" \
    --eval-output "$EVAL_OUTPUT" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --gpu-ids "$GPU_IDS" \
    --port "$PORT" \
    --browser-timeout-ms "$BROWSER_TIMEOUT_MS" \
    --reset-retries "$RESET_RETRIES" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${OPTIONAL_ARGS[@]}" \
    "$@"
