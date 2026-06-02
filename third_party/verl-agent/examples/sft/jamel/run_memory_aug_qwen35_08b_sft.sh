set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_AGENT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
JAMEL_ROOT=$(cd -- "$VERL_AGENT_ROOT/../.." && pwd)

export PYTHONPATH="$JAMEL_ROOT:$VERL_AGENT_ROOT:${PYTHONPATH}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-VL-0.8B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-"$JAMEL_ROOT/data/jamel_sft_data/jamel_memory_sft_train.parquet"}
VAL_FILE=${VAL_FILE:-"$JAMEL_ROOT/data/jamel_sft_data/jamel_memory_sft_val.parquet"}
MEMORY_HIDDEN_SIZE=${MEMORY_HIDDEN_SIZE:-auto}
MEMORY_MAX_ITEMS=${MEMORY_MAX_ITEMS:-4}

python3 -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.max_length=2048 \
    data.truncation=right \
    data.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/sft_dataset.py" \
    data.custom_cls.name="MemoryTokenSFTDataset" \
    data.memory_max_items="$MEMORY_MAX_ITEMS" \
    model.partial_pretrain="$MODEL_PATH" \
    model.custom_cls.path="file://$JAMEL_ROOT/jamel/train/memory/modeling.py" \
    model.custom_cls.name="MemoryAugmentedCausalLM" \
    model.memory_augment.memory_hidden_size="$MEMORY_HIDDEN_SIZE" \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp2 \
    model.lora_rank=0 \
    use_remove_padding=False \
    trainer.default_local_dir="$JAMEL_ROOT/outputs/jamel_memory_qwen35_08b_sft" \
    trainer.project_name='jamel-memory-sft' \
    trainer.experiment_name='memory_aug_qwen35_08b_sft' \
    trainer.total_epochs=1 \
    trainer.logger=['console'] "$@"
