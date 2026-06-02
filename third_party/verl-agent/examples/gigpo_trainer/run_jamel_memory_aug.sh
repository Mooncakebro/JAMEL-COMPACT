set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd) || {
    echo "JAMEL repository root not found above third_party/verl-agent" >&2
    exit 1
}

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
COMPRESSOR_MODEL_PATH=${COMPRESSOR_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
MEMORY_HIDDEN_SIZE=${MEMORY_HIDDEN_SIZE:-2048}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-8}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
GROUP_SIZE=${GROUP_SIZE:-4}
GIGPO_MODE=${GIGPO_MODE:-mean_norm}
TARGET_URL=${TARGET_URL:-http://localhost:8000/weibo/}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gigpo_jamel_memory_aug_$(date +%m%d%H%M%S)}
COVERAGE_DIR=${COVERAGE_DIR:-$VERL_AGENT_ROOT/outputs/$EXPERIMENT_NAME/jamel}
JAMEL_TRACE_MODE=${JAMEL_TRACE_MODE:-both}
JAMEL_TRACE_FREQ=${JAMEL_TRACE_FREQ:-1}
JAMEL_MAX_TRACES_PER_DUMP=${JAMEL_MAX_TRACES_PER_DUMP:-2}
JAMEL_TRACE_IMAGE=${JAMEL_TRACE_IMAGE:-True}
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}
TRAINER_MAX_CRITIC_CKPT_TO_KEEP=${TRAINER_MAX_CRITIC_CKPT_TO_KEEP:-2}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    actor_rollout_ref.model.custom_cls.name="MemoryAugmentedCausalLM" \
    actor_rollout_ref.model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    actor_rollout_ref.model.memory_augment.enable_online_builder=True \
    actor_rollout_ref.model.memory_augment.compressor_model_name="$COMPRESSOR_MODEL_PATH" \
    actor_rollout_ref.model.memory_augment.history_window=4 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="$GIGPO_MODE" \
    env.env_name=JAMEL \
    env.seed=0 \
    env.max_steps=6 \
    env.history_length=2 \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=1 \
    env.jamel.target_urls="[\"$TARGET_URL\"]" \
    env.jamel.coverage_dir="$COVERAGE_DIR" \
    env.jamel.jamel_root="$JAMEL_ROOT" \
    env.jamel.record_coverage=True \
    env.jamel.headless=True \
    env.jamel.trace_mode="$JAMEL_TRACE_MODE" \
    env.jamel.trace_freq="$JAMEL_TRACE_FREQ" \
    env.jamel.max_traces_per_dump="$JAMEL_MAX_TRACES_PER_DUMP" \
    env.jamel.trace_image="$JAMEL_TRACE_IMAGE" \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_jamel' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep="$TRAINER_MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.max_critic_ckpt_to_keep="$TRAINER_MAX_CRITIC_CKPT_TO_KEEP" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False "$@"
