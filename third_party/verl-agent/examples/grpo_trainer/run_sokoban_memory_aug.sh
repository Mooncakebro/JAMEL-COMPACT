set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd)

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

ENGINE=${ENGINE:-hf}


num_cpus_per_env_worker=0.05 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

# Use a Qwen3-VL actor backbone together with compressed memory prefix.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
COMPRESSOR_MODEL_PATH=${COMPRESSOR_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
COMPRESSOR_DEVICE_MAP=${COMPRESSOR_DEVICE_MAP:-local_cuda}
COMPRESSOR_TORCH_DTYPE=${COMPRESSOR_TORCH_DTYPE:-auto}
MEMORY_HIDDEN_SIZE=${MEMORY_HIDDEN_SIZE:-2048}
MEMORY_MAX_ITEMS=${MEMORY_MAX_ITEMS:-6}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-4}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
GROUP_SIZE=${GROUP_SIZE:-2}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-8}
ACTOR_PPO_MICRO_BATCH_SIZE=${ACTOR_PPO_MICRO_BATCH_SIZE:-1}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE:-1}
REF_LOG_PROB_MICRO_BATCH_SIZE=${REF_LOG_PROB_MICRO_BATCH_SIZE:-8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_memory_aug_$(date +%m%d%H%M%S)}

python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size="$TRAIN_DATA_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.image_key=images \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    actor_rollout_ref.model.custom_cls.name="MemoryAugmentedCausalLM" \
    actor_rollout_ref.model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    actor_rollout_ref.model.memory_augment.enable_online_builder=True \
    actor_rollout_ref.model.memory_augment.compressor_model_name="$COMPRESSOR_MODEL_PATH" \
    actor_rollout_ref.model.memory_augment.compressor_device_map="$COMPRESSOR_DEVICE_MAP" \
    actor_rollout_ref.model.memory_augment.compressor_torch_dtype="$COMPRESSOR_TORCH_DTYPE" \
    actor_rollout_ref.model.memory_augment.history_window=1000 \
    +actor_rollout_ref.model.memory_augment.max_memory_items="$MEMORY_MAX_ITEMS" \
    actor_rollout_ref.model.memory_augment.cache_history_memory=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ACTOR_PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Sokoban \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$GROUP_SIZE \
    env.sokoban.mode='rgb_array' \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_sokoban' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
