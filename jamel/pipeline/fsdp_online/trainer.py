from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)

from jamel.log import log_utils
from .rollout import RolloutStepRecord

logger = log_utils.get_logger(__name__)


@dataclass
class PolicyGradientStats:
    loss: float
    mean_step_reward: float
    total_step_reward: float
    num_steps: int
    num_chunks: int
    num_completion_tokens: int


@dataclass
class ChunkSample:
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    reward: float
    temperature: float
    top_p: float


def _flatten_chunk_samples(
    records: List[RolloutStepRecord],
    default_temperature: float,
    default_top_p: float,
) -> List[ChunkSample]:
    chunk_samples: List[ChunkSample] = []
    for record in records:
        rollout_chunks = (record.rollout_data or {}).get("chunks", [])
        for chunk in rollout_chunks:
            completion_token_ids = list(chunk.get("completion_token_ids", []))
            if not completion_token_ids:
                continue
            chunk_samples.append(
                ChunkSample(
                    prompt_token_ids=list(chunk.get("prompt_token_ids", [])),
                    completion_token_ids=completion_token_ids,
                    reward=float(record.reward),
                    temperature=float(chunk.get("temperature", default_temperature)),
                    top_p=float(chunk.get("top_p", default_top_p)),
                )
            )
    return chunk_samples


def _iter_micro_batches(
    samples: List[ChunkSample],
    micro_batch_size: int,
) -> Iterable[List[ChunkSample]]:
    for start_idx in range(0, len(samples), max(1, micro_batch_size)):
        yield samples[start_idx : start_idx + max(1, micro_batch_size)]


def _collate_micro_batch(
    samples: List[ChunkSample],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_tensors = []
    label_tensors = []
    reward_tensors = []
    temperature_tensors = []
    top_p_tensors = []
    max_len = 0

    for sample in samples:
        input_ids = sample.prompt_token_ids + sample.completion_token_ids
        labels = [-100] * len(sample.prompt_token_ids) + sample.completion_token_ids
        input_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        label_tensors.append(torch.tensor(labels, dtype=torch.long))
        reward_tensors.append(sample.reward)
        temperature_tensors.append(sample.temperature)
        top_p_tensors.append(sample.top_p)
        max_len = max(max_len, len(input_ids))

    padded_inputs = []
    padded_labels = []
    padded_attention = []
    for input_ids, labels in zip(input_tensors, label_tensors):
        padding_len = max_len - len(input_ids)
        padded_inputs.append(
            torch.cat([input_ids, torch.full((padding_len,), pad_token_id, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([labels, torch.full((padding_len,), -100, dtype=torch.long)])
        )
        padded_attention.append(
            torch.cat(
                [
                    torch.ones(len(input_ids), dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long),
                ]
            )
        )

    return {
        "input_ids": torch.stack(padded_inputs).to(device),
        "labels": torch.stack(padded_labels).to(device),
        "attention_mask": torch.stack(padded_attention).to(device),
        "reward": torch.tensor(reward_tensors, dtype=torch.float32, device=device),
        "temperature": torch.tensor(
            temperature_tensors,
            dtype=torch.float32,
            device=device,
        ),
        "top_p": torch.tensor(top_p_tensors, dtype=torch.float32, device=device),
    }


def _sample_log_probs_from_rollout_policy(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return F.log_softmax(logits, dim=-1)

    scaled_logits = logits / max(temperature, 1e-6)
    if 0 < top_p < 1.0:
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        truncated_probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        return torch.log(truncated_probs.clamp_min(1e-12))

    return F.log_softmax(scaled_logits, dim=-1)


def _compute_sequence_logprob(
    model,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    active_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~active_mask, 0)

    token_logprobs = torch.zeros(
        shift_labels.shape,
        dtype=shift_logits.dtype,
        device=shift_logits.device,
    )
    for sample_idx in range(shift_logits.shape[0]):
        sample_log_probs = _sample_log_probs_from_rollout_policy(
            shift_logits[sample_idx],
            temperature=float(batch["temperature"][sample_idx].item()),
            top_p=float(batch["top_p"][sample_idx].item()),
        )
        token_logprobs[sample_idx] = sample_log_probs.gather(
            -1,
            safe_labels[sample_idx].unsqueeze(-1),
        ).squeeze(-1).to(shift_logits.dtype)

    token_logprobs = token_logprobs * active_mask.to(token_logprobs.dtype)
    sequence_logprob = token_logprobs.sum(dim=1)
    completion_token_count = active_mask.sum(dim=1)
    return sequence_logprob, completion_token_count


def run_policy_gradient_update(
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    rollout_records: List[RolloutStepRecord],
    device: torch.device,
    micro_batch_size: int,
    max_grad_norm: float,
    pg_epochs: int,
    rollout_temperature: float,
    rollout_top_p: float,
) -> PolicyGradientStats:
    num_steps = len(rollout_records)
    chunk_samples = _flatten_chunk_samples(
        rollout_records,
        default_temperature=rollout_temperature,
        default_top_p=rollout_top_p,
    )
    if num_steps == 0 or not chunk_samples:
        return PolicyGradientStats(
            loss=0.0,
            mean_step_reward=0.0,
            total_step_reward=0.0,
            num_steps=num_steps,
            num_chunks=0,
            num_completion_tokens=0,
        )

    optimizer.zero_grad(set_to_none=True)
    model.train()

    total_loss_value = 0.0
    total_completion_tokens = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for _ in range(max(1, pg_epochs)):
        for micro_batch in _iter_micro_batches(chunk_samples, micro_batch_size):
            batch = _collate_micro_batch(
                micro_batch,
                pad_token_id=pad_token_id,
                device=device,
            )
            sequence_logprob, completion_token_count = _compute_sequence_logprob(model, batch)
            loss = -(sequence_logprob * batch["reward"]).sum() / max(1, num_steps)
            loss.backward()
            total_loss_value += float(loss.detach().item())
            total_completion_tokens += int(completion_token_count.sum().item())

    if hasattr(model, "clip_grad_norm_"):
        model.clip_grad_norm_(max_grad_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    total_step_reward = float(sum(record.reward for record in rollout_records))
    mean_step_reward = total_step_reward / max(1, num_steps)
    return PolicyGradientStats(
        loss=total_loss_value,
        mean_step_reward=mean_step_reward,
        total_step_reward=total_step_reward,
        num_steps=num_steps,
        num_chunks=len(chunk_samples),
        num_completion_tokens=total_completion_tokens,
    )


def aggregate_pg_stats(stats: PolicyGradientStats, device: torch.device) -> PolicyGradientStats:
    if not dist.is_available() or not dist.is_initialized():
        return stats

    tensor = torch.tensor(
        [
            stats.loss,
            stats.total_step_reward,
            float(stats.num_steps),
            float(stats.num_chunks),
            float(stats.num_completion_tokens),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    loss, total_step_reward, num_steps, num_chunks, num_completion_tokens = tensor.tolist()
    mean_step_reward = total_step_reward / max(1.0, num_steps)
    return PolicyGradientStats(
        loss=float(loss),
        mean_step_reward=float(mean_step_reward),
        total_step_reward=float(total_step_reward),
        num_steps=int(num_steps),
        num_chunks=int(num_chunks),
        num_completion_tokens=int(num_completion_tokens),
    )


def save_fsdp_checkpoint(
    model,
    tokenizer,
    output_dir: str | Path,
    rank: int,
    extra_state: dict[str, Any] | None = None,
) -> Path | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, FullyShardedDataParallel):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = model.state_dict()
        if rank != 0:
            return None
        model.module.save_pretrained(str(output_dir), state_dict=state_dict)
    else:
        if rank != 0:
            return None
        model.save_pretrained(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))
    if extra_state is not None:
        with open(output_dir / "trainer_state.json", "w", encoding="utf-8") as file_obj:
            json.dump(extra_state, file_obj, ensure_ascii=False, indent=2)
    logger.info("saved_fsdp_online_checkpoint", output_dir=str(output_dir))
    return output_dir
