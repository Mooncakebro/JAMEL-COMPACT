from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler

from jamel.log import log_utils
from jamel.train.agent.format_funcs.context_memory import (
    format_web_explorer_example_w_context_memory,
)
from jamel.train.agent.torch.train_torch import _apply_lora

logger = log_utils.get_logger(__name__)


@dataclass
class ModelArguments:
    model_id_or_path: str
    trust_remote_code: bool = True


@dataclass
class DataArguments:
    data_path: str
    dataset_num_proc: int = 1
    train_batch_size: int = 1
    reward_column: str = "reward"
    return_column: str = "return"
    format_messages: bool = True


@dataclass
class TrainingArguments:
    output_dir: str = field(default="./output_rl")
    logging_dir: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3

    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42

    max_seq_length: int = field(
        default=32768,
        metadata={
            "help": "Maximum sequence length. Long examples keep the right-most tokens."
        },
    )

    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = None

    local_rank: int = field(default=-1, metadata={"help": "Local rank for DDP training"})
    world_size: int = field(default=1, metadata={"help": "World size for DDP training"})
    fp16: bool = field(default=False, metadata={"help": "Use FP16 mixed precision"})
    bf16: bool = field(default=False, metadata={"help": "Use BF16 mixed precision"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Enable gradient checkpointing"}
    )

    rl_algorithm: str = field(
        default="ppo", metadata={"help": "Supported values: ppo, pg"}
    )
    gamma: float = 1.0
    reward_clip_min: Optional[float] = None
    reward_clip_max: Optional[float] = None
    normalize_advantages: bool = True
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    reference_kl_coef: float = 0.0
    sequence_logprob_reduction: str = field(
        default="mean", metadata={"help": "Supported values: mean, sum"}
    )
    dataloader_num_workers: int = 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compute_discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        returns[idx] = running
    return returns


def _load_single_parquet(
    file_path: Path,
    split: str,
    num_proc: int,
    reward_column: str,
    return_column: str,
    gamma: float,
    with_trajectory_return: bool,
):
    dataset = load_dataset(
        "parquet",
        data_files={split: str(file_path.resolve())},
        split=split,
        num_proc=num_proc,
    )
    if return_column not in dataset.column_names:
        if reward_column not in dataset.column_names:
            raise KeyError(
                f"Missing reward column `{reward_column}` in dataset: {file_path}"
            )
        if with_trajectory_return and "step" in dataset.column_names:
            dataset = dataset.sort("step")
        rewards = [float(reward or 0.0) for reward in dataset[reward_column]]
        returns = (
            _compute_discounted_returns(rewards, gamma)
            if with_trajectory_return
            else rewards
        )
        dataset = dataset.add_column(return_column, returns)
    if "__trajectory_id" not in dataset.column_names:
        dataset = dataset.add_column("__trajectory_id", [file_path.stem] * len(dataset))
    return dataset


def load_agent_dataset(
    data_path: str,
    split: str,
    num_proc: int,
    reward_column: str,
    return_column: str,
    gamma: float,
):
    path = Path(data_path)
    if path.is_file() and path.suffix == ".parquet":
        return _load_single_parquet(
            file_path=path,
            split=split,
            num_proc=num_proc,
            reward_column=reward_column,
            return_column=return_column,
            gamma=gamma,
            with_trajectory_return=False,
        )

    if path.is_dir():
        parquet_files = sorted(p for p in path.glob("*.parquet") if p.is_file())
        if parquet_files:
            datasets = [
                _load_single_parquet(
                    file_path=parquet_file,
                    split=split,
                    num_proc=num_proc,
                    reward_column=reward_column,
                    return_column=return_column,
                    gamma=gamma,
                    with_trajectory_return=True,
                )
                for parquet_file in parquet_files
            ]
            return concatenate_datasets(datasets)

    dataset = load_dataset(str(path.resolve()), split=split, num_proc=num_proc)
    if return_column not in dataset.column_names:
        if reward_column not in dataset.column_names:
            raise KeyError(
                f"Missing reward column `{reward_column}` in dataset: {data_path}"
            )
        rewards = [float(reward or 0.0) for reward in dataset[reward_column]]
        dataset = dataset.add_column(return_column, rewards)
    return dataset


def _ensure_messages(dataset, num_proc: int, format_messages: bool):
    if "messages" in dataset.column_names or not format_messages:
        return dataset
    return dataset.map(
        format_web_explorer_example_w_context_memory,
        num_proc=num_proc,
        load_from_cache_file=False,
    )


def _truncate(values: list[int], max_seq_length: int) -> list[int]:
    if len(values) <= max_seq_length:
        return values
    return values[-max_seq_length:]


def _tokenize_messages(messages, tokenizer, max_seq_length: int):
    if not messages:
        raise ValueError("Example is missing `messages`.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("RL training expects the final message to be from the assistant.")

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

        if assistant_mask is None:
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True
            )
            start = min(len(prompt_ids), len(input_ids))
            assistant_mask = [0] * start + [1] * (len(input_ids) - start)
    else:
        prompt_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages[:-1]
        )
        full_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        prompt_with_generation = (
            prompt_text + ("\n" if prompt_text else "") + "assistant:"
        )
        input_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer(prompt_with_generation, add_special_tokens=True)[
            "input_ids"
        ]
        start = min(len(prompt_ids), len(input_ids))
        assistant_mask = [0] * start + [1] * (len(input_ids) - start)

    input_ids = _truncate(list(input_ids), max_seq_length=max_seq_length)
    assistant_mask = _truncate(
        [int(mask) for mask in assistant_mask], max_seq_length=max_seq_length
    )
    attention_mask = [1] * len(input_ids)
    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(input_ids, assistant_mask)
    ]
    return input_ids, attention_mask, labels, assistant_mask


def _tokenize_example(
    example: dict,
    tokenizer,
    max_seq_length: int,
    reward_column: str,
    return_column: str,
):
    input_ids, attention_mask, labels, assistant_mask = _tokenize_messages(
        example["messages"], tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    reward = float(example.get(reward_column, 0.0) or 0.0)
    returns = float(example.get(return_column, reward) or reward)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "reward": reward,
        "returns": returns,
    }


class RLDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)

        def _pad(values: list[int], pad_value: int):
            return values + [pad_value] * (max_len - len(values))

        batch = {
            "input_ids": torch.tensor(
                [_pad(feature["input_ids"], pad_token_id) for feature in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [_pad(feature["attention_mask"], 0) for feature in features],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [_pad(feature["labels"], -100) for feature in features],
                dtype=torch.long,
            ),
            "assistant_mask": torch.tensor(
                [_pad(feature["assistant_mask"], 0) for feature in features],
                dtype=torch.long,
            ),
            "reward": torch.tensor(
                [feature["reward"] for feature in features], dtype=torch.float32
            ),
            "returns": torch.tensor(
                [feature["returns"] for feature in features], dtype=torch.float32
            ),
        }
        return batch


def _masked_reduce(values: torch.Tensor, mask: torch.Tensor, reduction: str) -> torch.Tensor:
    mask = mask.to(values.dtype)
    total = (values * mask).sum(dim=1)
    if reduction == "sum":
        return total
    if reduction == "mean":
        denom = mask.sum(dim=1).clamp(min=1.0)
        return total / denom
    raise ValueError(f"Unsupported reduction: {reduction}")


def _get_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr_name in ("hidden_size", "n_embd", "d_model", "model_dim"):
        hidden_size = getattr(config, attr_name, None)
        if hidden_size is not None:
            return int(hidden_size)
    raise ValueError("Failed to infer hidden size from model config.")


class ActorCriticModel(nn.Module):
    def __init__(self, policy_model: nn.Module):
        super().__init__()
        self.policy_model = policy_model
        self.value_head = nn.Linear(_get_hidden_size(policy_model), 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        values = self.value_head(hidden_states).squeeze(-1)
        return outputs.logits, values

    def get_policy_model(self) -> nn.Module:
        return self.policy_model


class OfflineRLTrainer:
    def __init__(
        self,
        actor_critic: ActorCriticModel,
        tokenizer,
        train_dataset,
        args: TrainingArguments,
        data_args: DataArguments,
        reference_model: Optional[nn.Module] = None,
    ):
        self.args = args
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.reference_model = reference_model
        self.global_step = 0
        self.backward_step = 0

        self._setup_distributed()
        self.device = self._get_device()

        actor_critic = actor_critic.to(self.device)
        if self.args.local_rank != -1:
            device_ids = [self.args.local_rank] if self.device.type == "cuda" else None
            actor_critic = DDP(actor_critic, device_ids=device_ids)
        self.actor_critic = actor_critic

        if self.reference_model is not None:
            self.reference_model = self.reference_model.to(self.device)
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad = False

        self.autocast_dtype = None
        if self.device.type == "cuda":
            if self.args.bf16:
                self.autocast_dtype = torch.bfloat16
            elif self.args.fp16:
                self.autocast_dtype = torch.float16

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.args.fp16 and self.device.type == "cuda"
        )

        self.optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.train_dataloader = self._build_dataloader()
        total_optimizer_steps = self._get_total_optimizer_steps()
        warmup_steps = self.args.warmup_steps
        if self.args.warmup_ratio > 0:
            warmup_steps = int(total_optimizer_steps * self.args.warmup_ratio)
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

    def _setup_distributed(self) -> None:
        if self.args.local_rank == -1 or dist.is_initialized():
            return
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        self.args.world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)

    def _get_device(self) -> torch.device:
        if self.args.local_rank != -1 and torch.cuda.is_available():
            return torch.device(f"cuda:{self.args.local_rank}")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_dataloader(self) -> DataLoader:
        sampler = None
        if self.args.local_rank != -1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
                shuffle=True,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_args.train_batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=RLDataCollator(self.tokenizer),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.device.type == "cuda",
        )

    def _get_total_optimizer_steps(self) -> int:
        updates_per_epoch = len(self.train_dataloader) * self.args.ppo_epochs
        updates_per_epoch = math.ceil(
            max(1, updates_per_epoch) / max(1, self.args.gradient_accumulation_steps)
        )
        return max(1, updates_per_epoch * self.args.num_train_epochs)

    def _unwrap_model(self) -> ActorCriticModel:
        return self.actor_critic.module if isinstance(self.actor_critic, DDP) else self.actor_critic

    def _move_batch_to_device(self, batch: dict) -> dict:
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda")
            for key, value in batch.items()
        }

    def _forward_actor_critic(self, batch: dict):
        if self.autocast_dtype is None:
            return self.actor_critic(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            return self.actor_critic(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

    def _forward_reference(self, batch: dict):
        if self.reference_model is None:
            return None
        if self.autocast_dtype is None:
            return self.reference_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            return self.reference_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits

    def _compute_sequence_statistics(
        self, logits: torch.Tensor, values: Optional[torch.Tensor], batch: dict
    ) -> dict:
        shift_logits = logits[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        shift_mask = (shift_labels != -100).to(logits.dtype)
        safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)

        token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_log_probs = token_log_probs * shift_mask

        token_entropy = -(
            F.softmax(shift_logits, dim=-1) * F.log_softmax(shift_logits, dim=-1)
        ).sum(dim=-1)
        token_entropy = token_entropy * shift_mask

        sequence_logprob = _masked_reduce(
            token_log_probs, shift_mask, self.args.sequence_logprob_reduction
        )
        sequence_entropy = _masked_reduce(token_entropy, shift_mask, "mean")

        sequence_value = None
        if values is not None:
            shift_values = values[:, :-1]
            sequence_value = _masked_reduce(shift_values, shift_mask, "mean")

        return {
            "sequence_logprob": sequence_logprob,
            "sequence_entropy": sequence_entropy,
            "sequence_value": sequence_value,
            "token_count": shift_mask.sum(dim=1),
        }

    def _prepare_rollout_stats(self, batch: dict) -> dict:
        with torch.no_grad():
            logits, values = self._forward_actor_critic(batch)
            stats = self._compute_sequence_statistics(logits=logits, values=values, batch=batch)
            old_logprob = stats["sequence_logprob"]
            old_value = stats["sequence_value"]
            if old_value is None:
                old_value = torch.zeros_like(batch["returns"])

            reference_logprob = None
            if self.reference_model is not None:
                reference_logits = self._forward_reference(batch)
                reference_stats = self._compute_sequence_statistics(
                    logits=reference_logits, values=None, batch=batch
                )
                reference_logprob = reference_stats["sequence_logprob"]

        returns = batch["returns"]
        if self.args.reward_clip_min is not None or self.args.reward_clip_max is not None:
            clip_min = (
                self.args.reward_clip_min
                if self.args.reward_clip_min is not None
                else torch.finfo(returns.dtype).min
            )
            clip_max = (
                self.args.reward_clip_max
                if self.args.reward_clip_max is not None
                else torch.finfo(returns.dtype).max
            )
            returns = returns.clamp(min=clip_min, max=clip_max)

        advantages = returns - old_value
        if self.args.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        rollout = {
            "old_logprob": old_logprob.detach(),
            "old_value": old_value.detach(),
            "returns": returns.detach(),
            "advantages": advantages.detach(),
            "reference_logprob": None if reference_logprob is None else reference_logprob.detach(),
        }
        return rollout

    def _compute_loss(self, batch: dict, rollout: dict) -> tuple[torch.Tensor, dict]:
        logits, values = self._forward_actor_critic(batch)
        stats = self._compute_sequence_statistics(logits=logits, values=values, batch=batch)
        new_logprob = stats["sequence_logprob"]
        new_value = stats["sequence_value"]
        entropy = stats["sequence_entropy"].mean()

        ratio = torch.exp(new_logprob - rollout["old_logprob"])
        advantages = rollout["advantages"]

        if self.args.rl_algorithm == "ppo":
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio, 1.0 - self.args.clip_range, 1.0 + self.args.clip_range
            ) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.args.clip_range).float().mean().item()
            )
        elif self.args.rl_algorithm == "pg":
            policy_loss = -(new_logprob * advantages).mean()
            clip_fraction = 0.0
        else:
            raise ValueError(
                f"Unsupported rl_algorithm={self.args.rl_algorithm}. Use `ppo` or `pg`."
            )

        value_pred_clipped = rollout["old_value"] + (
            new_value - rollout["old_value"]
        ).clamp(-self.args.value_clip_range, self.args.value_clip_range)
        value_loss_unclipped = (new_value - rollout["returns"]) ** 2
        value_loss_clipped = (value_pred_clipped - rollout["returns"]) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        reference_kl = torch.tensor(0.0, device=self.device)
        if rollout["reference_logprob"] is not None and self.args.reference_kl_coef > 0:
            reference_kl = (new_logprob - rollout["reference_logprob"]).mean()

        loss = (
            policy_loss
            + self.args.value_coef * value_loss
            - self.args.entropy_coef * entropy
            + self.args.reference_kl_coef * reference_kl
        )

        with torch.no_grad():
            approx_kl = (rollout["old_logprob"] - new_logprob).mean().item()
            value_mean = new_value.mean().item()

        metrics = {
            "loss": loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "entropy": entropy.detach().item(),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "ratio_mean": ratio.detach().mean().item(),
            "advantage_mean": advantages.detach().mean().item(),
            "return_mean": rollout["returns"].detach().mean().item(),
            "reward_mean": batch["reward"].detach().mean().item(),
            "value_mean": value_mean,
            "reference_kl": reference_kl.detach().item(),
        }
        return loss, metrics

    def _backward(self, loss: torch.Tensor) -> None:
        scaled_loss = loss / max(1, self.args.gradient_accumulation_steps)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _optimizer_step(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _log_metrics(self, epoch: int, ppo_epoch: int, metrics: dict) -> None:
        logger.info(
            "offline_rl_step",
            epoch=epoch,
            ppo_epoch=ppo_epoch,
            global_step=self.global_step,
            **metrics,
        )

    def _prune_checkpoints(self) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return
        output_dir = Path(self.args.output_dir)
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=lambda item: int(item.name.split("-")[-1]),
        )
        while len(checkpoints) > self.args.save_total_limit:
            oldest = checkpoints.pop(0)
            for child in oldest.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(oldest.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            oldest.rmdir()

    def _save_checkpoint(self, final: bool = False) -> None:
        if self.args.local_rank not in (-1, 0):
            return
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_dir = output_dir / ("final" if final else f"checkpoint-{self.global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self._unwrap_model()
        policy_model = unwrapped.get_policy_model()
        policy_model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        torch.save(unwrapped.value_head.state_dict(), save_dir / "value_head.pt")
        with open(save_dir / "trainer_state.json", "w", encoding="utf-8") as file_obj:
            json.dump(
                {
                    "global_step": self.global_step,
                    "training_args": asdict(self.args),
                    "data_args": asdict(self.data_args),
                },
                file_obj,
                ensure_ascii=False,
                indent=2,
            )
        if not final:
            self._prune_checkpoints()

    def train(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.actor_critic.train()

        for epoch in range(self.args.num_train_epochs):
            sampler = getattr(self.train_dataloader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            for batch in self.train_dataloader:
                batch = self._move_batch_to_device(batch)
                rollout = self._prepare_rollout_stats(batch)

                for ppo_epoch in range(self.args.ppo_epochs):
                    loss, metrics = self._compute_loss(batch=batch, rollout=rollout)
                    self._backward(loss)
                    self.backward_step += 1

                    if (
                        self.backward_step % max(1, self.args.gradient_accumulation_steps)
                        == 0
                    ):
                        self._optimizer_step()

                        if self.global_step % self.args.logging_steps == 0:
                            self._log_metrics(epoch=epoch, ppo_epoch=ppo_epoch, metrics=metrics)
                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            self._save_checkpoint(final=False)

        if self.backward_step % max(1, self.args.gradient_accumulation_steps) != 0:
            self._optimizer_step()
        self._save_checkpoint(final=True)


def train(
    model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments
) -> None:
    _seed_everything(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id_or_path, trust_remote_code=model_args.trust_remote_code
    )
    policy_model = _apply_lora(policy_model, training_args)
    if training_args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False

    actor_critic = ActorCriticModel(policy_model)
    value_head_path = Path(model_args.model_id_or_path) / "value_head.pt"
    if value_head_path.exists():
        actor_critic.value_head.load_state_dict(
            torch.load(value_head_path, map_location="cpu")
        )

    reference_model = None
    if training_args.reference_kl_coef > 0:
        reference_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_id_or_path, trust_remote_code=model_args.trust_remote_code
        )
        reference_model.eval()

    dataset = load_agent_dataset(
        data_path=data_args.data_path,
        split="train",
        num_proc=data_args.dataset_num_proc,
        reward_column=data_args.reward_column,
        return_column=data_args.return_column,
        gamma=training_args.gamma,
    )
    dataset = _ensure_messages(
        dataset,
        num_proc=data_args.dataset_num_proc,
        format_messages=data_args.format_messages,
    )
    dataset = dataset.map(
        lambda example: _tokenize_example(
            example,
            tokenizer=tokenizer,
            max_seq_length=training_args.max_seq_length,
            reward_column=data_args.reward_column,
            return_column=data_args.return_column,
        ),
        remove_columns=dataset.column_names,
        num_proc=data_args.dataset_num_proc,
        load_from_cache_file=False,
    )
    dataset = dataset.filter(
        lambda example: any(mask == 1 for mask in example["assistant_mask"]),
        num_proc=data_args.dataset_num_proc,
        load_from_cache_file=False,
    )

    trainer = OfflineRLTrainer(
        actor_critic=actor_critic,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_args=data_args,
        reference_model=reference_model,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parsed = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    model_args, data_args, training_args, remaining = parsed
    if remaining:
        logger.warning("ignoring_extra_args", args=remaining)
    train(model_args, data_args, training_args)
