#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M)}
RUN_ROOT=${RUN_ROOT:-"$ROOT/outputs/baseline_gui_eval/gemini31_flash_lite_react_text_vision_10apps_50_tokenbudget_parallel8_${RUN_STAMP}"}
MODEL_NAME=${REACT_MODEL:-gemini-3.1-flash-lite-preview}
BASE_URL=${REACT_BASE_URL:-http://localhost:4000/v1/chat/completions}
TEXT_KEY=${REACT_TEXT_API_KEY:?Set REACT_TEXT_API_KEY for react-text}
VISION_KEY=${REACT_VISION_API_KEY:?Set REACT_VISION_API_KEY for react-vision}
TOKENIZER_NAME=${REACT_HISTORY_TOKENIZER_NAME:-Qwen/Qwen3-235B-A22B}
MAX_STEPS=${MAX_STEPS:-50}
SEED=${SEED:-7}
WORKERS=${WORKERS:-8}

mkdir -p "$RUN_ROOT/logs"
: > "$RUN_ROOT/logs/pids.txt"

APPS=(
  vipshop
  alibaba
  expedia
  taobao
  pinduoduo
  dongchedi
  youku
  keep
  meituan
  temu
)

if [[ "$WORKERS" -lt 1 ]]; then
  echo "WORKERS must be >= 1" >&2
  exit 2
fi

write_metadata() {
  local apps_json
  apps_json=$(printf '"%s",' "${APPS[@]}")
  apps_json="[${apps_json%,}]"
  cat > "$RUN_ROOT/run_metadata.json" <<JSON
{
  "run_root": "$RUN_ROOT",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "model": "$MODEL_NAME",
  "base_url": "$BASE_URL",
  "agents": ["react-text", "react-vision"],
  "apps": $apps_json,
  "max_steps": $MAX_STEPS,
  "seed": $SEED,
  "workers_per_agent": $WORKERS,
  "history_budget_mode": "token",
  "history_tokenizer_name": "$TOKENIZER_NAME",
  "model_context_tokens": 1000000,
  "context_margin_tokens": 8192,
  "model_max_tokens": 768,
  "model_temperature": 0.2,
  "model_reasoning_effort": "disable",
  "decision_timeout": 600,
  "env_step_timeout": 180,
  "notes": "No API keys are stored in this file. Keys are read from REACT_TEXT_API_KEY and REACT_VISION_API_KEY."
}
JSON
}

run_agent() {
  local agent=$1
  local key=$2
  local port_base=$3
  local run_id="gemini31_flash_lite_${agent}_10apps_50_tokenbudget_parallel8_${RUN_STAMP}"

  for ((worker=0; worker<WORKERS; worker++)); do
    shard=()
    for ((idx=worker; idx<${#APPS[@]}; idx+=WORKERS)); do
      shard+=("${APPS[$idx]}")
    done
    if [[ ${#shard[@]} -eq 0 ]]; then
      continue
    fi
    app_csv=$(IFS=,; echo "${shard[*]}")
    worker_name=$(printf "worker_%02d" "$worker")
    worker_dir="$RUN_ROOT/$agent/$worker_name"
    log_file="$RUN_ROOT/logs/${agent}_${worker_name}.log"
    mkdir -p "$worker_dir"
    echo "[$(date -u +%H:%M:%S)] launch $agent $worker_name apps=$app_csv port=$((port_base + worker))" | tee -a "$RUN_ROOT/logs/launch.log"
    (
      cd "$ROOT"
      export PYTHONUNBUFFERED=1
      export BASELINE_MODEL_API_KEY="$key"
      export BASELINE_MODEL_BASE_URL="$BASE_URL"
      export BASELINE_MODEL_NAME="$MODEL_NAME"
      "$PYTHON_BIN" -m jamel.cli.main baseline-eval \
        --apps "$app_csv" \
        --agents "$agent" \
        --max-steps "$MAX_STEPS" \
        --seed "$SEED" \
        --run-id "$run_id" \
        --output-dir "$worker_dir" \
        --port "$((port_base + worker))" \
        --policy model \
        --model-temperature 0.2 \
        --model-max-tokens 768 \
        --model-timeout 120 \
        --model-retries 3 \
        --model-retry-backoff 10 \
        --model-context-tokens 1000000 \
        --context-margin-tokens 8192 \
        --history-budget-mode token \
        --history-tokenizer-name "$TOKENIZER_NAME" \
        --history-observation-char-budget 0 \
        --decision-timeout 600 \
        --env-step-timeout 180 \
        --model-reasoning-effort disable
    ) > "$log_file" 2>&1 &
    echo "$! $agent $worker_name $app_csv" >> "$RUN_ROOT/logs/pids.txt"
  done
}

write_metadata
run_agent react-text "$TEXT_KEY" 8920
run_agent react-vision "$VISION_KEY" 8940

status=0
while read -r pid agent worker apps; do
  if ! wait "$pid"; then
    echo "[$(date -u +%H:%M:%S)] FAILED $agent $worker pid=$pid apps=$apps" | tee -a "$RUN_ROOT/logs/launch.log"
    status=1
  else
    echo "[$(date -u +%H:%M:%S)] DONE $agent $worker pid=$pid apps=$apps" | tee -a "$RUN_ROOT/logs/launch.log"
  fi
done < "$RUN_ROOT/logs/pids.txt"

exit "$status"
