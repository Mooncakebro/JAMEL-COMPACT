from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Callable, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments as HFTrainingArguments,
)

from jamel.train.agent.format_funcs.context_memory import (
    format_web_explorer_example_w_context_memory,
)


@dataclass
class ModelArguments:
    model_id_or_path: str


@dataclass
class DataArguments:
    data_path: str
    dataset_num_proc: int = 1


@dataclass
class TrainingArguments(HFTrainingArguments):
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = None
    max_seq_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    loss_fn_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional python path to a custom loss function, e.g. module.sub:loss_fn"
        },
    )


class SFTDataCollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict]) -> dict:
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        for f in features:
            f.pop("labels", None)
        batch = super().__call__(features)

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, label in enumerate(labels):
            padded_labels[i, : label.shape[0]] = label
        batch["labels"] = padded_labels
        return batch


class SFTTrainer(Trainer):
    def __init__(self, *args, loss_fn: Optional[Callable[..., Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        if self._loss_fn is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)
        return self._loss_fn(model=model, inputs=inputs, return_outputs=return_outputs)


def _resolve_loss_fn(loss_fn_path: Optional[str]) -> Optional[Callable[..., Any]]:
    if not loss_fn_path:
        return None
    if ":" not in loss_fn_path:
        raise ValueError("loss_fn_path must be in the form 'module.sub:callable_name'")
    module_name, fn_name = loss_fn_path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Resolved loss function is not callable: {loss_fn_path}")
    return fn


def _find_all_linear_names(model: torch.nn.Module) -> list[str]:
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            linear_names.add(parts[-1])
    return sorted(linear_names)


def _apply_lora(model: torch.nn.Module, training_args: TrainingArguments) -> torch.nn.Module:
    if not training_args.lora_rank or not training_args.lora_alpha:
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    if training_args.lora_target_modules:
        target_modules = [m.strip() for m in training_args.lora_target_modules.split(",") if m.strip()]
    else:
        target_modules = _find_all_linear_names(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def _tokenize_example(example: dict, tokenizer, max_seq_length: int) -> dict:
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        output = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
        )
        if isinstance(output, dict):
            input_ids = output["input_ids"]
            assistant_mask = output.get("assistant_tokens_mask")
        else:
            input_ids = output
            assistant_mask = None
    else:
        text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        input_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        assistant_mask = None

    input_ids = input_ids[:max_seq_length]
    attention_mask = [1] * len(input_ids)

    if assistant_mask is not None:
        assistant_mask = assistant_mask[:max_seq_length]
        labels = [tid if mask == 1 else -100 for tid, mask in zip(input_ids, assistant_mask)]
    else:
        labels = input_ids[:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id_or_path,
        trust_remote_code=True,
    )

    model = _apply_lora(model, training_args)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = load_dataset(data_args.data_path, split="train", num_proc=data_args.dataset_num_proc)
    train_dataset = train_dataset.map(
        format_web_explorer_example_w_context_memory,
        num_proc=data_args.dataset_num_proc,
        load_from_cache_file=False,
    )
    train_dataset = train_dataset.map(
        lambda ex: _tokenize_example(ex, tokenizer, training_args.max_seq_length),
        remove_columns=train_dataset.column_names,
        num_proc=data_args.dataset_num_proc,
        load_from_cache_file=False,
    )

    loss_fn = _resolve_loss_fn(training_args.loss_fn_path)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTDataCollator(tokenizer=tokenizer, padding=True),
        tokenizer=tokenizer,
        loss_fn=loss_fn,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
