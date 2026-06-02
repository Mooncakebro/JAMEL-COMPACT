#!/usr/bin/env bash
set -euo pipefail

ROOT=${JAMEL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%d_%H%M)}
MODEL_PATH=${QWEN3VL8B_MODEL_PATH:-"$ROOT/models/Qwen3-VL-8B-Instruct"}
MODEL_NAME=${REACT_MODEL:-qwen3-vl-8b-instruct}
BASE_URL=${REACT_BASE_URL:-http://127.0.0.1:4010/v1/chat/completions}
API_KEY=${REACT_API_KEY:-local-qwen3vl8b}
TOKENIZER_NAME=${REACT_HISTORY_TOKENIZER_NAME:-$MODEL_PATH}
MODEL_CONTEXT_TOKENS=${MODEL_CONTEXT_TOKENS:-262144}
CONTEXT_MARGIN_TOKENS=${CONTEXT_MARGIN_TOKENS:-8192}
MODEL_MAX_TOKENS=${MODEL_MAX_TOKENS:-768}
MODEL_TEMPERATURE=${MODEL_TEMPERATURE:-0.2}
MAX_STEPS=${MAX_STEPS:-50}
SEED=${SEED:-7}
WORKERS=${WORKERS:-8}
RUN_AGENTS_SEQUENTIALLY=${RUN_AGENTS_SEQUENTIALLY:-1}
MODEL_TIMEOUT=${MODEL_TIMEOUT:-600}
DECISION_TIMEOUT=${DECISION_TIMEOUT:-900}
ENV_STEP_TIMEOUT=${ENV_STEP_TIMEOUT:-180}
RUN_ROOT=${RUN_ROOT:-"$ROOT/outputs/baseline_gui_eval/qwen3vl8b_react_text_vision_10apps_50_tokenbudget_ctx${MODEL_CONTEXT_TOKENS}_parallel${WORKERS}_${RUN_STAMP}"}

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
  "model_path": "$MODEL_PATH",
  "base_url": "$BASE_URL",
  "agents": ["react-text", "react-vision"],
  "apps": $apps_json,
  "max_steps": $MAX_STEPS,
  "seed": $SEED,
  "workers_per_agent": $WORKERS,
  "run_agents_sequentially": $RUN_AGENTS_SEQUENTIALLY,
  "history_budget_mode": "token",
  "history_tokenizer_name": "$TOKENIZER_NAME",
  "model_context_tokens": $MODEL_CONTEXT_TOKENS,
  "context_margin_tokens": $CONTEXT_MARGIN_TOKENS,
  "model_max_tokens": $MODEL_MAX_TOKENS,
  "model_temperature": $MODEL_TEMPERATURE,
  "decision_timeout": $DECISION_TIMEOUT,
  "env_step_timeout": $ENV_STEP_TIMEOUT,
  "notes": "Local Qwen3-VL-8B run. API key is a local placeholder and is not stored here."
}
JSON
}

run_agent() {
  local agent=$1
  local port_base=$2
  local run_id="qwen3vl8b_${agent}_10apps_50_tokenbudget_ctx${MODEL_CONTEXT_TOKENS}_parallel${WORKERS}_${RUN_STAMP}"

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
      export BASELINE_MODEL_API_KEY="$API_KEY"
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
        --model-temperature "$MODEL_TEMPERATURE" \
        --model-max-tokens "$MODEL_MAX_TOKENS" \
        --model-timeout "$MODEL_TIMEOUT" \
        --model-retries 3 \
        --model-retry-backoff 10 \
        --model-context-tokens "$MODEL_CONTEXT_TOKENS" \
        --context-margin-tokens "$CONTEXT_MARGIN_TOKENS" \
        --history-budget-mode token \
        --history-tokenizer-name "$TOKENIZER_NAME" \
        --history-observation-char-budget 0 \
        --decision-timeout "$DECISION_TIMEOUT" \
        --env-step-timeout "$ENV_STEP_TIMEOUT"
    ) > "$log_file" 2>&1 &
    echo "$! $agent $worker_name $app_csv" >> "$RUN_ROOT/logs/pids.txt"
  done
}

wait_for_pids() {
  local status=0
  while read -r pid agent worker apps; do
    if ! wait "$pid"; then
      echo "[$(date -u +%H:%M:%S)] FAILED $agent $worker pid=$pid apps=$apps" | tee -a "$RUN_ROOT/logs/launch.log"
      status=1
    else
      echo "[$(date -u +%H:%M:%S)] DONE $agent $worker pid=$pid apps=$apps" | tee -a "$RUN_ROOT/logs/launch.log"
    fi
  done < "$RUN_ROOT/logs/pids.txt"
  : > "$RUN_ROOT/logs/pids.txt"
  return "$status"
}

write_metadata

status=0
if [[ "$RUN_AGENTS_SEQUENTIALLY" == "1" ]]; then
  run_agent react-text 9020
  wait_for_pids || status=1
  run_agent react-vision 9040
  wait_for_pids || status=1
else
  run_agent react-text 9020
  run_agent react-vision 9040
  wait_for_pids || status=1
fi

exit "$status"
