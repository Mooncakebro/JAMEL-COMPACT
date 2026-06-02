from __future__ import annotations

from dataclasses import dataclass

import torch

QWEN3_VL_EOS_TOKEN_ID = 151645
QWEN3_VL_VISION_START_TOKEN_ID = 151652
QWEN3_VL_VISION_END_TOKEN_ID = 151653

DEFAULT_QWEN3_VL_8B_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QWEN3_VL_2B_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_QWEN3_8B_MODEL_NAME = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class QwenVLTextPairConfig:
    pair_name: str
    compressor_model_name: str
    llm_model_name: str
    compressor_hidden_size: int
    llm_hidden_size: int
    max_context_length: int
    eos_token_id: int = QWEN3_VL_EOS_TOKEN_ID
    vision_start_token_id: int = QWEN3_VL_VISION_START_TOKEN_ID
    vision_end_token_id: int = QWEN3_VL_VISION_END_TOKEN_ID
    deepstack_indices: tuple[int, ...] = ()

    @property
    def requires_alignment(self) -> bool:
        return self.compressor_hidden_size != self.llm_hidden_size


DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B = QwenVLTextPairConfig(
    pair_name="qwen3_vl_8b_to_qwen3_8b",
    compressor_model_name=DEFAULT_QWEN3_VL_8B_MODEL_NAME,
    llm_model_name=DEFAULT_QWEN3_8B_MODEL_NAME,
    compressor_hidden_size=4096,
    llm_hidden_size=4096,
    max_context_length=4096,
    deepstack_indices=(8, 16, 24),
)

DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B = QwenVLTextPairConfig(
    pair_name="qwen3_vl_2b_to_qwen3_vl_2b",
    compressor_model_name=DEFAULT_QWEN3_VL_2B_MODEL_NAME,
    llm_model_name=DEFAULT_QWEN3_VL_2B_MODEL_NAME,
    compressor_hidden_size=2048,
    llm_hidden_size=2048,
    max_context_length=4096,
    deepstack_indices=(5, 11, 17),
)

DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B = QwenVLTextPairConfig(
    pair_name="qwen3_vl_2b_to_qwen3_8b",
    compressor_model_name=DEFAULT_QWEN3_VL_2B_MODEL_NAME,
    llm_model_name=DEFAULT_QWEN3_8B_MODEL_NAME,
    compressor_hidden_size=2048,
    llm_hidden_size=4096,
    max_context_length=4096,
    deepstack_indices=(5, 11, 17),
)


def resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> str | torch.dtype:
    if torch_dtype is None or torch_dtype == "auto":
        return "auto"
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    normalized = str(torch_dtype).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
