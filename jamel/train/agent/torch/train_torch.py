from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

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
    train_batch_size: int = 4
    eval_batch_size: int = 4


@dataclass
class TrainingArguments:
    # Output and logging
    output_dir: str = field(default="./output")
    logging_dir: Optional[str] = field(default=None)
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3

    # Training hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Sequence length
    max_seq_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    # LoRA configuration
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = None

    # DeepSpeed configuration
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file (JSON) or 'auto' for automatic configuration"}
    )
    deepspeed_config: Optional[dict] = field(
        default=None,
        metadata={"help": "DeepSpeed config as dict (alternative to deepspeed file path)"}
    )

    # Custom loss function
    loss_fn_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional python path to a custom loss function, e.g. module.sub:loss_fn"
        },
    )

    # Distributed training
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})
    world_size: int = field(default=1, metadata={"help": "Number of processes for distributed training"})

    # Optimization
    fp16: bool = field(default=False, metadata={"help": "Use FP16 mixed precision training"})
    bf16: bool = field(default=False, metadata={"help": "Use BF16 mixed precision training"})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Enable gradient checkpointing"})


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_seq_length: int, num_proc: int = 1):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Load and process dataset
        train_dataset = load_dataset(data_path, split="train", num_proc=num_proc)
        train_dataset = train_dataset.map(
            format_web_explorer_example_w_context_memory,
            num_proc=num_proc,
            load_from_cache_file=False,
        )

        # Tokenize
        self.data = train_dataset.map(
            lambda ex: self._tokenize_example(ex, tokenizer, max_seq_length),
            remove_columns=train_dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
        )

    def _tokenize_example(self, example: dict, tokenizer, max_seq_length: int) -> dict:
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function to pad sequences in a batch"""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for ids, mask, lbls in zip(input_ids, attention_mask, labels):
        seq_len = len(ids)
        padding_len = max_len - seq_len

        # Right padding
        padded_input_ids.append(torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([mask, torch.zeros(padding_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((padding_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
    }


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


def _find_all_linear_names(model: nn.Module) -> list[str]:
    linear_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.split(".")
            linear_names.add(parts[-1])
    return sorted(linear_names)


def _apply_lora(model: nn.Module, training_args: TrainingArguments) -> nn.Module:
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


def _get_deepspeed_config(training_args: TrainingArguments) -> dict:
    """Get or create DeepSpeed configuration"""
    if training_args.deepspeed_config:
        return training_args.deepspeed_config

    if training_args.deepspeed:
        if training_args.deepspeed == "auto":
            # Auto-generate DeepSpeed config
            config = {
                "train_batch_size": training_args.train_batch_size * training_args.world_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": training_args.learning_rate,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": training_args.weight_decay,
                    },
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "total_num_steps": -1,  # Will be set during training
                        "warmup_min_lr": 0,
                        "warmup_max_lr": training_args.learning_rate,
                        "warmup_num_steps": training_args.warmup_steps,
                    },
                },
                "fp16": {"enabled": training_args.fp16},
                "bf16": {"enabled": training_args.bf16},
                "gradient_clipping": training_args.max_grad_norm,
            }
            return config
        else:
            # Load from file
            with open(training_args.deepspeed, "r") as f:
                return json.load(f)

    return None


class SFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        tokenizer,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.global_step = 0

        # Setup distributed training
        self._setup_distributed()

        # Setup DeepSpeed if enabled
        self.deepspeed_engine = None
        self.optimizer = None
        self.scheduler = None

        if args.deepspeed:
            self._setup_deepspeed()
        else:
            self._setup_standard_training()

    def _setup_distributed(self):
        """Setup distributed training if needed"""
        if self.args.local_rank != -1:
            torch.distributed.init_process_group(backend="nccl")
            self.args.world_size = torch.distributed.get_world_size()
            torch.cuda.set_device(self.args.local_rank)

    def _setup_deepspeed(self):
        """Setup DeepSpeed engine"""
        import deepspeed

        ds_config = _get_deepspeed_config(self.args)

        # Update total steps in scheduler
        if "scheduler" in ds_config:
            num_training_steps = len(self.train_dataset) * self.args.num_train_epochs
            effective_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
            num_training_steps = num_training_steps // (effective_batch_size * self.args.world_size)
            ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps

        # Create DeepSpeed engine
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
        )

        self.deepspeed_engine = model_engine
        self.optimizer = optimizer

    def _setup_standard_training(self):
        """Setup standard PyTorch training without DeepSpeed"""
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        # Setup learning rate scheduler
        num_training_steps = len(self.train_dataset) * self.args.num_train_epochs
        effective_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_training_steps = num_training_steps // effective_batch_size

        warmup_steps = self.args.warmup_steps
        if self.args.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.0,
            total_iters=warmup_steps,
        )

        # Setup mixed precision
        self.scaler = None
        if self.args.fp16 or self.args.bf16:
            dtype = torch.bfloat16 if self.args.bf16 else torch.float16
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.args.fp16)

    def train(self):
        """Main training loop"""
        # Create data loader
        sampler = None
        if self.args.local_rank != -1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
                shuffle=True,
            )

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # Training loop
        self.model.train()
        if self.deepspeed_engine:
            model = self.deepspeed_engine
        else:
            model = self.model

        device = next(model.parameters()).device

        for epoch in range(self.args.num_train_epochs):
            if sampler:
                sampler.set_epoch(epoch)

            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                if self.loss_fn:
                    loss = self.loss_fn(model=model, inputs=batch, return_outputs=False)
                else:
                    outputs = model(**batch)
                    loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps

                # Backward pass
                if self.deepspeed_engine:
                    model.backward(loss)
                else:
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Optimizer step
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.deepspeed_engine:
                        model.step()
                    else:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm
                            )
                            self.optimizer.step()

                        if self.scheduler:
                            self.scheduler.step()

                        self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        print(f"Epoch {epoch}, Step {self.global_step}, Loss: {loss.item():.4f}")

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()

        # Save final model
        self._save_checkpoint(final=True)

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        if self.args.local_rank in [-1, 0]:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if final:
                save_dir = output_dir / "final"
            else:
                save_dir = output_dir / f"checkpoint-{self.global_step}"

            save_dir.mkdir(parents=True, exist_ok=True)

            if self.deepspeed_engine:
                # DeepSpeed saves with its own format
                self.deepspeed_engine.save_checkpoint(str(save_dir))
            else:
                # Save model and tokenizer
                self.model.save_pretrained(str(save_dir))
                self.tokenizer.save_pretrained(str(save_dir))

                # Save optimizer and scheduler
                torch.save(
                    {
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "global_step": self.global_step,
                    },
                    save_dir / "optimizer.pt",
                )


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments) -> None:
    """Main training function with same API as HuggingFace version"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id_or_path,
        trust_remote_code=True,
    )

    # Apply LoRA if configured
    model = _apply_lora(model, training_args)

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Create dataset
    train_dataset = SFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        num_proc=data_args.dataset_num_proc,
    )

    # Resolve custom loss function
    loss_fn = _resolve_loss_fn(training_args.loss_fn_path)

    # Update training args with batch size from data args
    training_args.train_batch_size = data_args.train_batch_size

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train(model_args, data_args, training_args)
