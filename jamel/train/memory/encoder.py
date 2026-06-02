from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional
from datetime import datetime, timezone
import os
import time

import numpy as np
import torch
from PIL import Image

from jamel.arch.qwen3vl_compressor.screen_compressor import ScreenCompressor


def _profile_enabled() -> bool:
    return os.environ.get("JAMEL_PROFILE", "0") == "1"


def _profile_log(event: str, *, start_time: float | None = None, **kwargs: Any) -> None:
    if not _profile_enabled():
        return
    payload = {key: value for key, value in kwargs.items()}
    if start_time is not None:
        payload["elapsed_s"] = round(time.perf_counter() - start_time, 6)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    extra = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[PROFILE {timestamp}] {event} {extra}".rstrip(), flush=True)


def _describe_image(image: Any) -> dict[str, Any]:
    if isinstance(image, Image.Image):
        return {
            "python_type": type(image).__name__,
            "pil_mode": image.mode,
            "pil_size": list(image.size),
        }
    if isinstance(image, np.ndarray):
        return {
            "python_type": type(image).__name__,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": int(image.min()) if image.size else None,
            "max": int(image.max()) if image.size else None,
        }
    return {
        "python_type": type(image).__name__,
        "repr": repr(image),
    }


def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image if image.dtype == np.uint8 else image.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}, value={_describe_image(image)!r}")


def _normalize_memory_rows(
    token: torch.Tensor,
    *,
    hidden_size: int,
) -> torch.Tensor:
    if token.ndim == 1:
        if token.shape[0] != hidden_size:
            raise ValueError(
                f"Invalid memory token width: expected {hidden_size}, got {token.shape[0]}."
            )
        token = token.unsqueeze(0)
    elif token.ndim == 2:
        if token.shape[1] != hidden_size:
            raise ValueError(
                f"Invalid memory token width: expected {hidden_size}, got {token.shape[1]}."
            )
    else:
        raise ValueError(
            f"Memory token cache/compressor output must be rank-1 or rank-2, got rank-{token.ndim}."
        )
    return token.to(dtype=torch.float32, device="cpu")


class OnlineHistoryMemoryBuilder:
    def __init__(
        self,
        *,
        compressor_model_name: str,
        memory_hidden_size: int | str | None,
        history_window: int = 4,
        max_memory_items: int | None = None,
        history_action_prefix: str = "Previous action:",
        torch_dtype: str | torch.dtype | None = "auto",
        device_map: str | dict | None = "auto",
        cache_history_memory: bool = True,
        compressor: Optional[ScreenCompressor] = None,
    ) -> None:
        self.history_window = max(1, int(history_window))
        self.max_memory_items = None if max_memory_items is None else max(1, int(max_memory_items))
        self.history_action_prefix = history_action_prefix
        self.cache_history_memory = bool(cache_history_memory)
        self.compressor = compressor or ScreenCompressor(
            model_name=compressor_model_name,
            hidden_size=memory_hidden_size,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.memory_hidden_size = int(self.compressor.hidden_size)

    def build_memory_inputs(
        self,
        *,
        batch_size: int,
        history_records: Sequence[Sequence[dict[str, Any]]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_start = time.perf_counter()
        history_records = history_records or [[] for _ in range(batch_size)]

        flat_images: list[Image.Image] = []
        flat_texts: list[str] = []
        sample_tokens: list[list[torch.Tensor | tuple[str, int]]] = []
        cache_hits = 0
        cache_misses = 0

        history_slots = self.history_window
        if self.max_memory_items is not None:
            history_slots = min(history_slots, self.max_memory_items)
        prepare_start = time.perf_counter()
        for idx in range(batch_size):
            records = list(history_records[idx])[-history_slots:]
            images: list[Image.Image] = []
            tokens_for_sample: list[torch.Tensor | tuple[str, int]] = []
            for record in records:
                action = str(record.get("action", "")).strip() or "unknown"
                action_text = f"{self.history_action_prefix} {action}"
                cached_token = record.get("_cached_memory_token") if self.cache_history_memory else None
                if isinstance(cached_token, torch.Tensor):
                    cache_hits += 1
                    tokens_for_sample.append(cached_token.to(dtype=torch.float32).cpu())
                    continue

                cache_misses += 1
                image_value = record["image_obs"]
                images.append(_to_pil_image(image_value))
                flat_images.append(images[-1])
                flat_texts.append(action_text)
                tokens_for_sample.append(("history", len(flat_images) - 1))

            sample_tokens.append(tokens_for_sample)
        _profile_log(
            "memory.prepare_inputs",
            start_time=prepare_start,
            batch_size=batch_size,
            total_pairs=len(flat_images),
            history_slots=history_slots,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        computed_memory: torch.Tensor | None = None
        if flat_images:
            compress_start = time.perf_counter()
            computed_memory = self.compressor.compress_batch(flat_images, flat_texts).detach().cpu()
            _profile_log(
                "memory.compress_batch",
                start_time=compress_start,
                batch_size=batch_size,
                total_pairs=len(flat_images),
                hidden_size=computed_memory.shape[-1],
            )
        else:
            _profile_log(
                "memory.compress_batch",
                batch_size=batch_size,
                total_pairs=0,
                hidden_size=self.memory_hidden_size,
                elapsed_s=0.0,
            )

        resolved_tokens: list[list[torch.Tensor]] = []
        group_sizes: list[int] = []
        for idx, tokens_for_sample in enumerate(sample_tokens):
            resolved_sample: list[torch.Tensor] = []
            history_records_for_sample = list(history_records[idx])[-history_slots:]
            history_record_idx = 0
            for token_ref in tokens_for_sample:
                if isinstance(token_ref, torch.Tensor):
                    resolved_sample.append(
                        _normalize_memory_rows(
                            token_ref,
                            hidden_size=self.memory_hidden_size,
                        )
                    )
                    history_record_idx += 1
                    continue
                kind, flat_idx = token_ref
                assert computed_memory is not None
                token = _normalize_memory_rows(
                    computed_memory[flat_idx],
                    hidden_size=computed_memory.shape[-1],
                )
                resolved_sample.append(token)
                if kind == "history" and self.cache_history_memory:
                    history_records_for_sample[history_record_idx]["_cached_memory_token"] = token.clone()
                    history_record_idx += 1
            group_sizes.append(sum(token.shape[0] for token in resolved_sample))
            resolved_tokens.append(resolved_sample)

        max_group_size = max(group_sizes, default=0)
        pack_start = time.perf_counter()
        hidden_size = self.memory_hidden_size if computed_memory is None else computed_memory.shape[-1]
        packed_memory = torch.zeros((batch_size, max_group_size, hidden_size), dtype=torch.float32)
        memory_attention_mask = torch.zeros((batch_size, max_group_size), dtype=torch.long)

        for row_idx, group_size in enumerate(group_sizes):
            if group_size == 0:
                continue
            row_tokens = torch.cat(resolved_tokens[row_idx], dim=0)
            packed_memory[row_idx, :group_size] = row_tokens
            memory_attention_mask[row_idx, :group_size] = 1

        _profile_log(
            "memory.pack_outputs",
            start_time=pack_start,
            batch_size=batch_size,
            max_group_size=max_group_size,
        )
        _profile_log(
            "memory.build_total",
            start_time=total_start,
            batch_size=batch_size,
            total_pairs=len(flat_images),
        )

        return packed_memory, memory_attention_mask
