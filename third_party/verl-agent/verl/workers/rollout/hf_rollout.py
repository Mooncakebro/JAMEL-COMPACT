# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.torch_functional import get_response_mask
from verl.utils.device import get_torch_device

from .base import BaseRollout

__all__ = ["HFRollout"]


def _profile_enabled() -> bool:
    return os.environ.get("JAMEL_PROFILE", "0") == "1"


def _profile_log(event: str, *, start_time: float | None = None, **kwargs) -> None:
    if not _profile_enabled():
        return
    payload = {key: value for key, value in kwargs.items()}
    if start_time is not None:
        payload["elapsed_s"] = round(time.perf_counter() - start_time, 6)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    extra = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[PROFILE {timestamp}] {event} {extra}".rstrip(), flush=True)


def _debug_generation_enabled() -> bool:
    return os.environ.get("JAMEL_DEBUG_GENERATION", "0") == "1"


def _debug_log(message: str) -> None:
    if _debug_generation_enabled():
        print(f"[DEBUG_GENERATION] {message}", flush=True)


def _repeat_position_ids(position_ids: torch.Tensor, repeats: int) -> torch.Tensor:
    if position_ids.ndim != 3 or repeats == 1:
        return position_ids.repeat_interleave(repeats, dim=0)
    if position_ids.shape[0] <= 8:
        return position_ids.repeat_interleave(repeats, dim=1)
    return position_ids.repeat_interleave(repeats, dim=0)


def _append_response_position_ids(position_ids: torch.Tensor, response_length: int) -> torch.Tensor:
    delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype)
    if position_ids.ndim == 2:
        batch_size = position_ids.shape[0]
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        return torch.cat([position_ids, response_position_ids], dim=-1)
    if position_ids.ndim != 3:
        raise ValueError(f"Unsupported position_ids rank: {position_ids.ndim}")

    if position_ids.shape[0] <= 8:
        num_channels, batch_size, _ = position_ids.shape
        delta_position_id = delta_position_id.view(1, 1, -1).expand(num_channels, batch_size, -1)
    else:
        batch_size, num_channels, _ = position_ids.shape
        delta_position_id = delta_position_id.view(1, 1, -1).expand(batch_size, num_channels, -1)
    response_position_ids = position_ids[..., -1:] + delta_position_id
    return torch.cat([position_ids, response_position_ids], dim=-1)


def _stack_multi_modal_inputs(prompts: DataProto) -> dict[str, torch.Tensor]:
    multi_modal_inputs = prompts.non_tensor_batch.get("multi_modal_inputs", None)
    if multi_modal_inputs is None:
        return {}
    if len(multi_modal_inputs) == 0:
        return {}

    stacked_inputs: dict[str, torch.Tensor] = {}
    sample0 = multi_modal_inputs[0]
    if not isinstance(sample0, dict):
        raise TypeError(
            f"`multi_modal_inputs` must contain per-sample dicts, got {type(sample0)!r}."
        )

    for key in sample0.keys():
        values = [sample[key] for sample in multi_modal_inputs]
        if any(value is None for value in values):
            continue
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(
                f"`multi_modal_inputs[{key}]` must be tensors, got {[type(value).__name__ for value in values]!r}."
            )
        stacked_inputs[key] = torch.cat(values, dim=0)
    return stacked_inputs


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, memory_config=None):
        super().__init__()
        self.config = config
        self.module = module
        self.memory_builder = None
        if memory_config is not None and memory_config.get("enable_online_builder", False):
            from jamel.train.memory import OnlineHistoryMemoryBuilder

            self.memory_builder = OnlineHistoryMemoryBuilder(
                compressor_model_name=memory_config.compressor_model_name,
                memory_hidden_size=memory_config.memory_hidden_size,
                history_window=memory_config.get("history_window", 4),
                max_memory_items=memory_config.get("max_memory_items", None),
                torch_dtype=memory_config.get("compressor_torch_dtype", "auto"),
                device_map=memory_config.get("compressor_device_map", "auto"),
                cache_history_memory=memory_config.get("cache_history_memory", True),
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        total_start = time.perf_counter()
        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": self.config.n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        memory_tokens = prompts.batch.get("memory_tokens", None)
        memory_attention_mask = prompts.batch.get("memory_attention_mask", None)
        multi_modal_inputs = _stack_multi_modal_inputs(prompts)
        if memory_tokens is None and self.memory_builder is not None:
            history_records = prompts.non_tensor_batch.get("memory_history_records", None)
            if history_records is not None:
                history_records = [list(records) if isinstance(records, (list, tuple, np.ndarray)) else [] for records in history_records]
                memory_build_start = time.perf_counter()
                memory_tokens, memory_attention_mask = self.memory_builder.build_memory_inputs(
                    batch_size=idx.shape[0],
                    history_records=history_records,
                )
                _profile_log(
                    "hf_rollout.memory_builder",
                    start_time=memory_build_start,
                    batch_size=idx.shape[0],
                    memory_shape=tuple(memory_tokens.shape),
                )

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generate_start = time.perf_counter()
            debug_generation = _debug_generation_enabled()
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                memory_tokens=memory_tokens,
                memory_attention_mask=memory_attention_mask,
                **multi_modal_inputs,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=debug_generation,
                return_dict_in_generate=True,
                use_cache=True,
            )
        _profile_log(
            "hf_rollout.generate",
            start_time=generate_start,
            batch_size=idx.shape[0],
            prompt_length=prompt_length,
            response_length=response_length,
            has_memory=memory_tokens is not None,
            has_multimodal=bool(multi_modal_inputs),
        )

        if _debug_generation_enabled():
            sequences = output.sequences if hasattr(output, "sequences") else output
            first_tokens = sequences[:, prompt_length]
            _debug_log(
                "batch_size={} prompt_length={} first_tokens={}".format(
                    sequences.shape[0],
                    prompt_length,
                    first_tokens.tolist(),
                )
            )
            if getattr(output, "scores", None):
                first_step_scores = output.scores[0].detach().float().cpu()
                top_k = min(10, first_step_scores.shape[-1])
                top_values, top_indices = torch.topk(first_step_scores[0], k=top_k)
                _debug_log(
                    "first_step_topk={}".format(
                        [
                            {
                                "token_id": int(token_id),
                                "score": float(score),
                            }
                            for token_id, score in zip(top_indices.tolist(), top_values.tolist())
                        ]
                    )
                )
                _debug_log(
                    "eos_token_id={} eos_score={}".format(
                        eos_token_id,
                        {
                            int(token_id): float(first_step_scores[0, token_id])
                            for token_id in (
                                eos_token_id
                                if isinstance(eos_token_id, (list, tuple))
                                else [eos_token_id]
                            )
                        },
                    )
                )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = _repeat_position_ids(position_ids, num_return_sequences)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            if memory_tokens is not None:
                memory_tokens = memory_tokens.repeat_interleave(num_return_sequences, dim=0)
            if memory_attention_mask is not None:
                memory_attention_mask = memory_attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        position_ids = _append_response_position_ids(position_ids, response_length)

        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )
        if memory_tokens is not None:
            batch["memory_tokens"] = memory_tokens.to(seq.device)
        if memory_attention_mask is not None:
            batch["memory_attention_mask"] = memory_attention_mask.to(seq.device)

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        _profile_log(
            "hf_rollout.total",
            start_time=total_start,
            batch_size=idx.shape[0],
            prompt_length=prompt_length,
            response_length=response_length,
        )
        return DataProto(batch=batch)
