set -x
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd) || {
    echo "JAMEL repository root not found above third_party/verl-agent" >&2
    exit 1
}

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=1

train_data_size=8
val_data_size=8
group_size=4
target_url=${TARGET_URL:-http://localhost:8000/weibo/}
experiment_name=${EXPERIMENT_NAME:-grpo_qwen2.5_1.5b}
coverage_dir=${COVERAGE_DIR:-$(pwd)/outputs/$experiment_name/jamel}
jamel_trace_mode=${JAMEL_TRACE_MODE:-both}
jamel_trace_freq=${JAMEL_TRACE_FREQ:-1}
jamel_max_traces_per_dump=${JAMEL_MAX_TRACES_PER_DUMP:-2}
jamel_trace_image=${JAMEL_TRACE_IMAGE:-True}
trainer_max_actor_ckpt_to_keep=${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}
trainer_max_critic_ckpt_to_keep=${TRAINER_MAX_CRITIC_CKPT_TO_KEEP:-2}

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=JAMEL \
    env.seed=0 \
    env.max_steps=6 \
    env.history_length=2 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.jamel.target_urls="[\"$target_url\"]" \
    env.jamel.coverage_dir=$coverage_dir \
    env.jamel.jamel_root="$JAMEL_ROOT" \
    env.jamel.record_coverage=True \
    env.jamel.headless=True \
    env.jamel.trace_mode=$jamel_trace_mode \
    env.jamel.trace_freq=$jamel_trace_freq \
    env.jamel.max_traces_per_dump=$jamel_max_traces_per_dump \
    env.jamel.trace_image=$jamel_trace_image \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_jamel' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$trainer_max_actor_ckpt_to_keep \
    trainer.max_critic_ckpt_to_keep=$trainer_max_critic_ckpt_to_keep \
    trainer.test_freq=5 \
    trainer.total_epochs=50 \
    trainer.val_before_train=True $@
