from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL.Image import Image
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from jamel.log import log_utils

from .config import (
    DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B,
    DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B,
    DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B,
    QwenVLTextPairConfig,
    resolve_torch_dtype,
)
from .memory_injector import DimensionAligner, InjectedInputs, MemoryInjector
from .screen_compressor import ScreenCompressor

logger = log_utils.get_logger(__name__)


def _infer_llm_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("LLM does not expose a config object.")
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError("Unable to infer LLM hidden_size from model config.")


def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _infer_model_dtype(model: nn.Module) -> torch.dtype:
    try:
        return model.get_input_embeddings().weight.dtype
    except Exception:
        return next(model.parameters()).dtype


def _load_downstream_model(
    model_name: str,
    *,
    torch_dtype: str | torch.dtype | None,
    device_map: str | dict | None,
) -> nn.Module:
    resolved_dtype = resolve_torch_dtype(torch_dtype)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as causal_error:
        logger.warning(
            "Falling back to multimodal downstream loader",
            model_name=model_name,
            error=str(causal_error),
        )
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )


def _load_text_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception as tokenizer_error:
        logger.warning(
            "Falling back to processor tokenizer",
            model_name=model_name,
            error=str(tokenizer_error),
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"Processor for {model_name} does not expose a tokenizer.") from tokenizer_error

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class MemoryAugmentedLLM(nn.Module):
    """
    End-to-end wrapper for Qwen3-VL screen compression plus Qwen3 text decoding.
    """

    def __init__(
        self,
        *,
        compressor: ScreenCompressor,
        llm: nn.Module,
        tokenizer,
        aligner: DimensionAligner | None = None,
        injector: MemoryInjector | None = None,
        pair_config: QwenVLTextPairConfig = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B,
    ) -> None:
        super().__init__()
        self.pair_config = pair_config
        self.compressor = compressor
        self.llm = llm
        self.tokenizer = tokenizer
        llm_hidden_size = _infer_llm_hidden_size(llm)
        self.aligner = aligner or DimensionAligner(
            src_dim=compressor.hidden_size,
            tgt_dim=llm_hidden_size,
        )
        self.injector = injector or MemoryInjector()
        self.llm.eval()

        llm_device = _infer_model_device(self.llm)
        llm_dtype = _infer_model_dtype(self.llm)
        self.aligner = self.aligner.to(device=llm_device, dtype=llm_dtype)

        if llm_hidden_size != self.pair_config.llm_hidden_size:
            raise ValueError(
                f"LLM hidden_size={llm_hidden_size}, expected {self.pair_config.llm_hidden_size}."
            )

        logger.info(
            "Loaded memory-augmented LLM",
            pair_name=self.pair_config.pair_name,
            llm_hidden_size=llm_hidden_size,
            llm_device=str(llm_device),
            llm_dtype=str(llm_dtype),
        )

    @classmethod
    def from_qwen3_vl_8b_to_qwen3_8b(
        cls,
        *,
        compressor_model_name: str = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_model_name,
        llm_model_name: str = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.llm_model_name,
        compressor_dtype: str | torch.dtype | None = "auto",
        llm_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
    ) -> "MemoryAugmentedLLM":
        compressor = ScreenCompressor(
            model_name=compressor_model_name,
            hidden_size=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_hidden_size,
            eos_token_id=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.eos_token_id,
            deepstack_indices=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.deepstack_indices,
            torch_dtype=compressor_dtype,
            device_map=device_map,
        )
        llm = _load_downstream_model(
            llm_model_name,
            torch_dtype=llm_dtype,
            device_map=device_map,
        )
        tokenizer = _load_text_tokenizer(llm_model_name)
        if getattr(llm.config, "pad_token_id", None) is None:
            llm.config.pad_token_id = tokenizer.pad_token_id
        return cls(
            compressor=compressor,
            llm=llm,
            tokenizer=tokenizer,
            pair_config=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B,
        )

    @classmethod
    def from_qwen3_vl_2b_to_qwen3_vl_2b(
        cls,
        *,
        compressor_model_name: str = DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B.compressor_model_name,
        llm_model_name: str = DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B.llm_model_name,
        compressor_dtype: str | torch.dtype | None = "auto",
        llm_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
    ) -> "MemoryAugmentedLLM":
        compressor = ScreenCompressor(
            model_name=compressor_model_name,
            hidden_size=DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B.compressor_hidden_size,
            eos_token_id=DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B.eos_token_id,
            deepstack_indices=DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B.deepstack_indices,
            torch_dtype=compressor_dtype,
            device_map=device_map,
        )
        llm = _load_downstream_model(
            llm_model_name,
            torch_dtype=llm_dtype,
            device_map=device_map,
        )
        tokenizer = _load_text_tokenizer(llm_model_name)
        if getattr(llm.config, "pad_token_id", None) is None:
            llm.config.pad_token_id = tokenizer.pad_token_id
        return cls(
            compressor=compressor,
            llm=llm,
            tokenizer=tokenizer,
            pair_config=DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B,
        )

    @classmethod
    def from_qwen3_vl_2b_to_qwen3_8b(
        cls,
        *,
        compressor_model_name: str = DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B.compressor_model_name,
        llm_model_name: str = DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B.llm_model_name,
        compressor_dtype: str | torch.dtype | None = "auto",
        llm_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
    ) -> "MemoryAugmentedLLM":
        compressor = ScreenCompressor(
            model_name=compressor_model_name,
            hidden_size=DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B.compressor_hidden_size,
            eos_token_id=DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B.eos_token_id,
            deepstack_indices=DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B.deepstack_indices,
            torch_dtype=compressor_dtype,
            device_map=device_map,
        )
        llm = _load_downstream_model(
            llm_model_name,
            torch_dtype=llm_dtype,
            device_map=device_map,
        )
        tokenizer = _load_text_tokenizer(llm_model_name)
        if getattr(llm.config, "pad_token_id", None) is None:
            llm.config.pad_token_id = tokenizer.pad_token_id
        return cls(
            compressor=compressor,
            llm=llm,
            tokenizer=tokenizer,
            pair_config=DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B,
        )

    def encode_memory(
        self,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
    ) -> torch.Tensor:
        memory_tokens, memory_attention_mask = self._encode_and_align_memory_tokens(
            screenshots=screenshots,
            action_texts=action_texts,
            expected_batch_size=None,
        )
        return self._format_memory_output(memory_tokens, memory_attention_mask)

    def prepare_inputs(
        self,
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        query_texts: str | Sequence[str],
    ) -> tuple[InjectedInputs, dict[str, torch.Tensor], torch.Tensor]:
        query_batch = self._ensure_query_batch(query_texts)
        memory_tokens, memory_attention_mask = self._encode_and_align_memory_tokens(
            screenshots=screenshots,
            action_texts=action_texts,
            expected_batch_size=len(query_batch),
        )

        llm_device = _infer_model_device(self.llm)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_queries = self.tokenizer(
            query_batch,
            return_tensors="pt",
            padding=True,
        )
        input_ids = tokenized_queries["input_ids"].to(llm_device)
        attention_mask = tokenized_queries.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(llm_device)

        injected_inputs = self.injector.inject(
            model=self.llm,
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tokenized_queries["input_ids"] = input_ids
        if attention_mask is not None:
            tokenized_queries["attention_mask"] = attention_mask
        tokenized_queries["memory_attention_mask"] = memory_attention_mask.to(llm_device)
        return injected_inputs, tokenized_queries, memory_tokens

    def forward(
        self,
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        query_texts: str | Sequence[str],
        labels: torch.Tensor | None = None,
        **llm_forward_kwargs,
    ):
        injected_inputs, _, _ = self.prepare_inputs(
            screenshots=screenshots,
            action_texts=action_texts,
            query_texts=query_texts,
        )
        model_kwargs = injected_inputs.to_model_kwargs()
        if labels is not None:
            model_kwargs["labels"] = self._prefix_labels(
                labels=labels,
                prefix_length=injected_inputs.prefix_length,
                device=model_kwargs["inputs_embeds"].device,
            )
        return self.llm(**model_kwargs, **llm_forward_kwargs)

    def generate(
        self,
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        query_texts: str | Sequence[str],
        **generate_kwargs,
    ) -> dict[str, torch.Tensor | list[str] | InjectedInputs]:
        injected_inputs, tokenized_queries, aligned_memory = self.prepare_inputs(
            screenshots=screenshots,
            action_texts=action_texts,
            query_texts=query_texts,
        )

        generate_inputs = injected_inputs.to_model_kwargs()
        generate_inputs.setdefault("max_new_tokens", 16)
        generate_inputs.update(generate_kwargs)

        if "pad_token_id" not in generate_inputs and self.tokenizer.pad_token_id is not None:
            generate_inputs["pad_token_id"] = self.tokenizer.pad_token_id
        if "eos_token_id" not in generate_inputs and self.tokenizer.eos_token_id is not None:
            generate_inputs["eos_token_id"] = self.tokenizer.eos_token_id

        generated_ids = self.llm.generate(**generate_inputs)
        decoded_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return {
            "memory": aligned_memory,
            "memory_attention_mask": tokenized_queries["memory_attention_mask"],
            "generated_ids": generated_ids,
            "decoded_texts": decoded_texts,
            "injected_inputs": injected_inputs,
            "query_input_ids": tokenized_queries["input_ids"],
        }

    @staticmethod
    def _ensure_query_batch(query_texts: str | Sequence[str]) -> list[str]:
        if isinstance(query_texts, str):
            return [query_texts]
        normalized = list(query_texts)
        if not normalized:
            raise ValueError("At least one query text is required.")
        return normalized

    @staticmethod
    def _ensure_text_batch(texts: str | Sequence[str], batch_size: int) -> list[str]:
        if isinstance(texts, str):
            return [texts] * batch_size
        normalized = list(texts)
        if len(normalized) != batch_size:
            raise ValueError(
                f"Expected {batch_size} text items, but received {len(normalized)}."
            )
        return normalized

    @staticmethod
    def _prefix_labels(
        *,
        labels: torch.Tensor,
        prefix_length: int,
        device: torch.device,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        labels = labels.to(device)
        prefix = torch.full(
            (labels.shape[0], prefix_length),
            ignore_index,
            dtype=labels.dtype,
            device=device,
        )
        return torch.cat([prefix, labels], dim=1)

    def _encode_and_align_memory_tokens(
        self,
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        expected_batch_size: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_groups, text_groups = self._normalize_memory_groups(
            screenshots=screenshots,
            action_texts=action_texts,
            expected_batch_size=expected_batch_size,
        )

        flat_images: list[Image] = []
        flat_texts: list[str] = []
        group_sizes: list[int] = []
        for image_group, text_group in zip(image_groups, text_groups):
            if len(image_group) != len(text_group):
                raise ValueError("Each memory image group must match its text group length.")
            if not image_group:
                raise ValueError("Each sample must contain at least one memory pair.")
            flat_images.extend(image_group)
            flat_texts.extend(text_group)
            group_sizes.append(len(image_group))

        raw_memory = self.compressor.compress_batch(flat_images, flat_texts)
        aligner_device = _infer_model_device(self.llm)
        aligner_dtype = _infer_model_dtype(self.llm)
        raw_memory = raw_memory.to(device=aligner_device, dtype=aligner_dtype)
        aligned_memory = self.aligner(raw_memory)
        return self._pack_memory_tokens(aligned_memory, group_sizes)

    @staticmethod
    def _normalize_memory_groups(
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        expected_batch_size: int | None,
    ) -> tuple[list[list[Image]], list[list[str]]]:
        image_groups = MemoryAugmentedLLM._normalize_image_groups(
            screenshots=screenshots,
            expected_batch_size=expected_batch_size,
        )
        text_groups = MemoryAugmentedLLM._normalize_text_groups(
            action_texts=action_texts,
            expected_batch_size=expected_batch_size,
        )
        if len(image_groups) != len(text_groups):
            raise ValueError("Top-level image groups and text groups must have the same batch size.")
        return image_groups, text_groups

    @staticmethod
    def _normalize_image_groups(
        *,
        screenshots: Image | Sequence[Image] | Sequence[Sequence[Image]],
        expected_batch_size: int | None,
    ) -> list[list[Image]]:
        if isinstance(screenshots, Image):
            return [[screenshots]]

        screenshot_list = list(screenshots)
        if not screenshot_list:
            raise ValueError("At least one screenshot is required.")

        if isinstance(screenshot_list[0], Image):
            if expected_batch_size is None or expected_batch_size == 1:
                return [list(screenshot_list)]
            if len(screenshot_list) == expected_batch_size:
                return [[image] for image in screenshot_list]
            raise ValueError(
                "Ambiguous screenshot input. Use nested lists to provide multiple memory pairs per sample."
            )

        image_groups = [list(image_group) for image_group in screenshot_list]
        if expected_batch_size is not None and len(image_groups) != expected_batch_size:
            raise ValueError(
                f"Expected {expected_batch_size} screenshot groups, but received {len(image_groups)}."
            )
        return image_groups

    @staticmethod
    def _normalize_text_groups(
        *,
        action_texts: str | Sequence[str] | Sequence[Sequence[str]],
        expected_batch_size: int | None,
    ) -> list[list[str]]:
        if isinstance(action_texts, str):
            return [[action_texts]]

        text_list = list(action_texts)
        if not text_list:
            raise ValueError("At least one action text is required.")

        if isinstance(text_list[0], str):
            if expected_batch_size is None or expected_batch_size == 1:
                return [list(text_list)]
            if len(text_list) == expected_batch_size:
                return [[text] for text in text_list]
            raise ValueError(
                "Ambiguous action_texts input. Use nested lists to provide multiple memory pairs per sample."
            )

        text_groups = [list(text_group) for text_group in text_list]
        if expected_batch_size is not None and len(text_groups) != expected_batch_size:
            raise ValueError(
                f"Expected {expected_batch_size} action text groups, but received {len(text_groups)}."
            )
        return text_groups

    @staticmethod
    def _pack_memory_tokens(
        aligned_memory: torch.Tensor,
        group_sizes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(group_sizes)
        max_group_size = max(group_sizes)
        hidden_size = aligned_memory.shape[-1]

        packed_memory = aligned_memory.new_zeros((batch_size, max_group_size, hidden_size))
        memory_attention_mask = torch.zeros(
            (batch_size, max_group_size),
            dtype=torch.long,
            device=aligned_memory.device,
        )

        offset = 0
        for row_idx, group_size in enumerate(group_sizes):
            next_offset = offset + group_size
            packed_memory[row_idx, :group_size] = aligned_memory[offset:next_offset]
            memory_attention_mask[row_idx, :group_size] = 1
            offset = next_offset

        return packed_memory, memory_attention_mask

    @staticmethod
    def _format_memory_output(
        memory_tokens: torch.Tensor,
        memory_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_counts = memory_attention_mask.sum(dim=-1).tolist()
        if memory_tokens.shape[0] == 1:
            valid_count = int(valid_counts[0])
            return memory_tokens[0, :valid_count]
        if all(int(count) == 1 for count in valid_counts):
            return memory_tokens[:, 0, :]
        return memory_tokens
