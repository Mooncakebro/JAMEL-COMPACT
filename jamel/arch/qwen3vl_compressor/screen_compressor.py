from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import os
import time

import torch
from PIL.Image import Image
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from jamel.log import log_utils

from .config import DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B, resolve_torch_dtype


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


def _infer_model_hidden_size(model) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config object.")

    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)

    raise ValueError("Unable to infer hidden_size from model config.")


def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


class ScreenCompressor(nn.Module):
    """
    Compress a screenshot and an action description into a single memory token
    using the EOS hidden state of a Qwen3-VL model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_model_name,
        *,
        hidden_size: int | str | None = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_hidden_size,
        eos_token_id: int = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.eos_token_id,
        deepstack_indices: Sequence[int] = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.deepstack_indices,
        torch_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
        processor=None,
        model=None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.eos_token_id = int(eos_token_id)
        self.deepstack_indices = tuple(deepstack_indices)

        self.processor = processor or AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        resolved_device_map = None if device_map in {"local_cuda", "current_cuda"} else device_map
        self.model = model or AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=resolved_dtype,
            device_map=resolved_device_map,
            trust_remote_code=True,
        )
        if model is None and device_map in {"local_cuda", "current_cuda"} and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda", torch.cuda.current_device()))
        self.model.eval()
        self.device = _infer_model_device(self.model)
        _profile_log(
            "memory.compressor.init",
            model_name=model_name,
            requested_device_map=device_map,
            resolved_device_map=resolved_device_map,
            resolved_dtype=resolved_dtype,
            model_device=str(self.device),
        )

        actual_hidden_size = _infer_model_hidden_size(self.model)
        if hidden_size is None or hidden_size == "auto":
            self.hidden_size = actual_hidden_size
        else:
            self.hidden_size = int(hidden_size)
        if actual_hidden_size != self.hidden_size:
            raise ValueError(
                f"{model_name} hidden_size={actual_hidden_size}, expected {self.hidden_size}."
            )

    @classmethod
    def from_qwen3_vl_8b(
        cls,
        *,
        model_name: str = DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_model_name,
        torch_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
    ) -> "ScreenCompressor":
        return cls(
            model_name=model_name,
            hidden_size=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.compressor_hidden_size,
            eos_token_id=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.eos_token_id,
            deepstack_indices=DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B.deepstack_indices,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def compress(
        self,
        image: Image,
        text: str,
        *,
        return_layer_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        compressed = self.compress_batch(
            images=[image],
            texts=[text],
            return_layer_features=return_layer_features,
        )
        if return_layer_features:
            memory, layer_memories = compressed
            return memory[:1], {layer: values[:1] for layer, values in layer_memories.items()}
        return compressed[:1]

    def compress_batch(
        self,
        images: Sequence[Image],
        texts: Sequence[str],
        *,
        return_layer_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        if len(images) != len(texts):
            raise ValueError("images and texts must have the same batch size.")
        if not images:
            raise ValueError("compress_batch requires at least one image/text pair.")

        total_start = time.perf_counter()
        template_start = time.perf_counter()
        templated_texts = [
            self.processor.apply_chat_template(
                self._build_messages(image=image, text=text),
                tokenize=False,
                add_generation_prompt=False,
            )
            for image, text in zip(images, texts)
        ]
        _profile_log(
            "memory.compressor.chat_template",
            start_time=template_start,
            batch_size=len(images),
        )

        processor_start = time.perf_counter()
        batch_inputs = self.processor(
            text=templated_texts,
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
        _profile_log(
            "memory.compressor.processor",
            start_time=processor_start,
            batch_size=len(images),
        )

        eos_start = time.perf_counter()
        input_ids, attention_mask = self._ensure_batch_has_trailing_eos(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs.get("attention_mask"),
        )
        batch_inputs["input_ids"] = input_ids
        batch_inputs["attention_mask"] = attention_mask
        _profile_log(
            "memory.compressor.ensure_eos",
            start_time=eos_start,
            batch_size=len(images),
            seq_len=int(batch_inputs["input_ids"].shape[-1]),
        )

        move_start = time.perf_counter()
        batch_inputs = self._move_tensor_batch(batch_inputs, self.device)
        _profile_log(
            "memory.compressor.move_to_device",
            start_time=move_start,
            batch_size=len(images),
            model_device=str(self.device),
        )

        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(
                **batch_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        _profile_log(
            "memory.compressor.forward",
            start_time=forward_start,
            batch_size=len(images),
        )

        gather_start = time.perf_counter()
        eos_positions = self._find_last_eos_positions(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
        ).to(outputs.hidden_states[-1].device)
        memory = self._gather_eos_embeddings(outputs.hidden_states[-1], eos_positions)
        _profile_log(
            "memory.compressor.gather_eos",
            start_time=gather_start,
            batch_size=len(images),
            hidden_size=int(memory.shape[-1]),
        )
        _profile_log(
            "memory.compressor.total",
            start_time=total_start,
            batch_size=len(images),
            hidden_size=int(memory.shape[-1]),
        )

        if not return_layer_features:
            return memory

        layer_memories: dict[int, torch.Tensor] = {}
        for layer_idx in self.deepstack_indices:
            if layer_idx >= len(outputs.hidden_states):
                raise ValueError(
                    f"Requested layer {layer_idx}, but only {len(outputs.hidden_states) - 1} layers are available."
                )
            layer_memories[layer_idx] = self._gather_eos_embeddings(
                outputs.hidden_states[layer_idx],
                eos_positions,
            )

        return memory, layer_memories

    def _build_messages(self, image: Image, text: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def _ensure_batch_has_trailing_eos(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 tensor.")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(dtype=torch.long)

        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.eos_token_id

        trimmed_rows: list[torch.Tensor] = []
        lengths = attention_mask.sum(dim=-1).tolist()
        for row_input_ids, row_length in zip(input_ids, lengths):
            valid_tokens = row_input_ids[: int(row_length)]
            if valid_tokens.numel() == 0 or int(valid_tokens[-1].item()) != self.eos_token_id:
                eos = torch.tensor([self.eos_token_id], dtype=row_input_ids.dtype)
                valid_tokens = torch.cat([valid_tokens, eos], dim=0)
            trimmed_rows.append(valid_tokens)

        max_len = max(row.shape[0] for row in trimmed_rows)
        padded_input_ids = input_ids.new_full((len(trimmed_rows), max_len), fill_value=pad_token_id)
        padded_attention_mask = attention_mask.new_zeros((len(trimmed_rows), max_len))

        for idx, row in enumerate(trimmed_rows):
            row_len = row.shape[0]
            padded_input_ids[idx, :row_len] = row
            padded_attention_mask[idx, :row_len] = 1

        return padded_input_ids, padded_attention_mask

    def _find_last_eos_positions(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        eos_mask = input_ids.eq(self.eos_token_id) & attention_mask.bool()
        if not torch.all(eos_mask.any(dim=-1)):
            raise ValueError("Each sequence must contain at least one EOS token.")

        reversed_positions = torch.argmax(torch.flip(eos_mask, dims=[-1]).to(torch.long), dim=-1)
        return input_ids.shape[-1] - 1 - reversed_positions

    @staticmethod
    def _gather_eos_embeddings(hidden_states: torch.Tensor, eos_positions: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_indices, eos_positions]

    @staticmethod
    def _move_tensor_batch(batch: dict, device: torch.device) -> dict:
        moved_batch = {}
        for key, value in batch.items():
            moved_batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return moved_batch
