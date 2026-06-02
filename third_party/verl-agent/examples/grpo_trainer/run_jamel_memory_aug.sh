# Example:
# JAMEL_PROFILE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
#   COMPRESSOR_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct \
#   bash examples/grpo_trainer/run_jamel_memory_aug.sh
set -x
. env.sh
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd) || {
    echo "JAMEL repository root not found above third_party/verl-agent" >&2
    exit 1
}

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

MEMORY_MAX_ITEMS=${MEMORY_MAX_ITEMS:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_jamel_memory_aug_$(date +%m%d%H%M%S)}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"tensorboard_log/${EXPERIMENT_NAME}"}
COVERAGE_DIR=${COVERAGE_DIR:-$VERL_AGENT_ROOT/outputs/$EXPERIMENT_NAME/jamel}
TARGET_URLS=${TARGET_URLS:-'["http://127.0.0.1:8000/weibo/index.html","http://127.0.0.1:8000/agoda/index.html","http://127.0.0.1:8000/airbnb/index.html","http://127.0.0.1:8000/douban/index.html","http://127.0.0.1:8000/bilibili/index.html","http://127.0.0.1:8000/wikipedia/index.html"]'}

MEMORY_EXTRA_OVERRIDES=()
if [[ -n "$MEMORY_MAX_ITEMS" ]]; then
    MEMORY_EXTRA_OVERRIDES+=(+actor_rollout_ref.model.memory_augment.max_memory_items="$MEMORY_MAX_ITEMS")
fi

# proxy_open
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size "${TRAIN_DATA_SIZE:-8}" \
#     --val_data_size "${VAL_DATA_SIZE:-8}"
# proxy_close

unset MASTER_PORT
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-2}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}" \
    actor_rollout_ref.model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    actor_rollout_ref.model.custom_cls.name="MemoryAugmentedCausalLM" \
    actor_rollout_ref.model.memory_augment.memory_hidden_size="${MEMORY_HIDDEN_SIZE:-2048}" \
    actor_rollout_ref.model.memory_augment.enable_online_builder=True \
    actor_rollout_ref.model.memory_augment.compressor_model_name="${COMPRESSOR_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}" \
    actor_rollout_ref.model.memory_augment.compressor_device_map="${COMPRESSOR_DEVICE_MAP:-local_cuda}" \
    actor_rollout_ref.model.memory_augment.compressor_torch_dtype="${COMPRESSOR_TORCH_DTYPE:-auto}" \
    actor_rollout_ref.model.memory_augment.history_window="${MEMORY_HISTORY_WINDOW:-100000}" \
    "${MEMORY_EXTRA_OVERRIDES[@]}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE:-8}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.prompt_length="${ROLLOUT_PROMPT_LENGTH:-${MAX_PROMPT_LENGTH:-8192}}" \
    actor_rollout_ref.rollout.response_length="${ROLLOUT_RESPONSE_LENGTH:-${MAX_RESPONSE_LENGTH:-1024}}" \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE:-8}" \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=JAMEL \
    env.seed=0 \
    env.max_steps="${ENV_MAX_STEPS:-6}" \
    env.history_length="${ENV_HISTORY_LENGTH:-0}" \
    env.rollout.n="${GROUP_SIZE:-8}" \
    env.resources_per_worker.num_cpus=0.1 \
    env.jamel.target_urls="$TARGET_URLS" \
    env.jamel.coverage_dir="$COVERAGE_DIR" \
    env.jamel.jamel_root="$JAMEL_ROOT" \
    env.jamel.record_coverage="${JAMEL_RECORD_COVERAGE:-True}" \
    env.jamel.headless="${JAMEL_HEADLESS:-True}" \
    env.jamel.timeout="${JAMEL_TIMEOUT:-600000}" \
    env.jamel.worker_timeout="${JAMEL_WORKER_TIMEOUT:-${JAMEL_TIMEOUT:-600000}}" \
    env.jamel.trace_mode="${JAMEL_TRACE_MODE:-both}" \
    env.jamel.trace_freq="${JAMEL_TRACE_FREQ:-1}" \
    env.jamel.max_traces_per_dump="${JAMEL_MAX_TRACES_PER_DUMP:-2}" \
    env.jamel.trace_image="${JAMEL_TRACE_IMAGE:-True}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_jamel' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE:-2}" \
    trainer.nnodes=1 \
    trainer.save_freq="${TRAINER_SAVE_FREQ:-1}" \
    trainer.max_actor_ckpt_to_keep="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}" \
    trainer.max_critic_ckpt_to_keep="${TRAINER_MAX_CRITIC_CKPT_TO_KEEP:-2}" \
    trainer.test_freq="${TRAINER_TEST_FREQ:-5}" \
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS:-100}" \
    trainer.val_before_train=False "$@"
