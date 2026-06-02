from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCHED = False


def _split_memory_items(memory_embeds: Any, memory_seq_lens: Any) -> tuple[torch.Tensor, ...]:
    if memory_embeds is None:
        return tuple()
    if isinstance(memory_embeds, (list, tuple)):
        return tuple(item for item in memory_embeds if item is not None)
    if not isinstance(memory_embeds, torch.Tensor):
        raise TypeError(f"memory_embeds must be a Tensor/list/tuple, got {type(memory_embeds)!r}")
    if memory_embeds.ndim == 3:
        return tuple(memory_embeds[i] for i in range(memory_embeds.shape[0]))
    if memory_embeds.ndim != 2:
        raise ValueError(f"memory_embeds must have rank 2 or 3, got shape={tuple(memory_embeds.shape)}")
    if memory_seq_lens is None:
        return (memory_embeds,)
    if isinstance(memory_seq_lens, torch.Tensor):
        seq_lens = memory_seq_lens.reshape(-1).tolist()
    else:
        seq_lens = list(memory_seq_lens)
    seq_lens = [int(x) for x in seq_lens if int(x) > 0]
    if not seq_lens:
        return tuple()
    total = sum(seq_lens)
    if total != memory_embeds.shape[0]:
        raise ValueError(
            f"memory_seq_lens sum mismatch: total={total} vs memory_embeds.shape[0]={memory_embeds.shape[0]}"
        )
    return tuple(memory_embeds.split(seq_lens, dim=0))


def apply_qwen3_vl_memory_patch(*, memory_placeholder_token_id: int) -> None:
    global _PATCHED
    if _PATCHED:
        return

    from transformers.feature_extraction_utils import BatchFeature
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLMultiModalProcessor
    from vllm.multimodal.inputs import MultiModalFieldConfig

    original_call_hf_processor = Qwen3VLMultiModalProcessor._call_hf_processor
    original_get_mm_fields_config = Qwen3VLMultiModalProcessor._get_mm_fields_config
    original_get_multimodal_embeddings = Qwen3VLForConditionalGeneration.get_multimodal_embeddings
    original_get_input_embeddings = Qwen3VLForConditionalGeneration.get_input_embeddings

    def _call_hf_processor_with_memory(self, prompt, mm_data, mm_processor_kwargs, **kwargs):
        mm_data = dict(mm_data or {})
        memory_items = mm_data.pop("memory", None)
        hf_inputs = original_call_hf_processor(self, prompt, mm_data, mm_processor_kwargs, **kwargs)
        if memory_items is None:
            return hf_inputs
        if isinstance(memory_items, torch.Tensor):
            memory_items = [memory_items]
        elif not isinstance(memory_items, (list, tuple)):
            raise TypeError(f"memory multimodal data must be Tensor/list/tuple, got {type(memory_items)!r}")

        filtered_items: list[torch.Tensor] = []
        memory_seq_lens: list[int] = []
        for item in memory_items:
            if item is None:
                continue
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"memory item must be torch.Tensor, got {type(item)!r}")
            if item.ndim != 2:
                raise ValueError(f"memory item must be rank-2 [N, H], got shape={tuple(item.shape)}")
            if item.shape[0] == 0:
                continue
            filtered_items.append(item)
            memory_seq_lens.append(int(item.shape[0]))

        if not filtered_items:
            return hf_inputs

        hf_inputs = BatchFeature(dict(hf_inputs))
        hf_inputs["memory_embeds"] = torch.cat(filtered_items, dim=0)
        hf_inputs["memory_seq_lens"] = torch.tensor(memory_seq_lens, dtype=torch.long)
        return hf_inputs

    def _get_mm_fields_config_with_memory(self, hf_inputs, hf_processor_mm_kwargs):
        fields = dict(original_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs))
        memory_embeds = hf_inputs.get("memory_embeds")
        memory_seq_lens = hf_inputs.get("memory_seq_lens")
        if memory_embeds is not None and memory_seq_lens is not None:
            fields["memory_embeds"] = MultiModalFieldConfig.flat_from_sizes("memory", memory_seq_lens)
            fields["memory_seq_lens"] = MultiModalFieldConfig.batched("memory")
        return fields

    def _get_multimodal_embeddings_with_memory(self, **kwargs):
        memory_embeds = kwargs.pop("memory_embeds", None)
        memory_seq_lens = kwargs.pop("memory_seq_lens", None)
        mm_embeddings = original_get_multimodal_embeddings(self, **kwargs)
        memory_items = _split_memory_items(memory_embeds, memory_seq_lens)
        if not memory_items:
            return mm_embeddings
        if mm_embeddings is None:
            return memory_items
        if isinstance(mm_embeddings, torch.Tensor):
            mm_embeddings = (mm_embeddings,)
        else:
            mm_embeddings = tuple(mm_embeddings)
        return memory_items + mm_embeddings

    def _get_input_embeddings_with_memory(self, input_ids, multimodal_embeddings=None, **kwargs):
        try:
            return original_get_input_embeddings(
                self,
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=[memory_placeholder_token_id],
                **kwargs,
            )
        except TypeError:
            return original_get_input_embeddings(self, input_ids, multimodal_embeddings=multimodal_embeddings, **kwargs)

    Qwen3VLMultiModalProcessor._call_hf_processor = _call_hf_processor_with_memory
    Qwen3VLMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config_with_memory
    Qwen3VLForConditionalGeneration.get_multimodal_embeddings = _get_multimodal_embeddings_with_memory
    Qwen3VLForConditionalGeneration.get_input_embeddings = _get_input_embeddings_with_memory
    Qwen3VLForConditionalGeneration.memory_placeholder_token_id = memory_placeholder_token_id

    _PATCHED = True
    logger.warning(
        "Applied repo-local vLLM Qwen3-VL memory patch with memory_placeholder_token_id=%s",
        memory_placeholder_token_id,
    )
