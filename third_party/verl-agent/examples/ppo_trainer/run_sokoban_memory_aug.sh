set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd)

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
COMPRESSOR_MODEL_PATH=${COMPRESSOR_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
MEMORY_HIDDEN_SIZE=${MEMORY_HIDDEN_SIZE:-2048}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-32}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-64}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_memory_aug_$(date +%m%d%H%M%S)}

python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    actor_rollout_ref.model.custom_cls.name="MemoryAugmentedCausalLM" \
    actor_rollout_ref.model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    actor_rollout_ref.model.memory_augment.enable_online_builder=True \
    actor_rollout_ref.model.memory_augment.compressor_model_name="$COMPRESSOR_MODEL_PATH" \
    actor_rollout_ref.model.memory_augment.history_window=4 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    critic.model.custom_cls.name="MemoryAugmentedValueModel" \
    critic.model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Sokoban \
    env.history_length=3 \
    env.seed=0 \
    env.max_steps=10 \
    env.rollout.n=1 \
    env.sokoban.mode='rgb_array' \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_sokoban' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False "$@"
