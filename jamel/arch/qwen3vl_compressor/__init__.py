"""
Qwen3-VL based screen compression and memory injection modules.
"""

from .config import (
    DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B,
    DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B,
    DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B,
    QwenVLTextPairConfig,
    resolve_torch_dtype,
)
from .memory_injector import DimensionAligner, InjectedInputs, MemoryInjector
from .model import MemoryAugmentedLLM
from .screen_compressor import ScreenCompressor

__all__ = [
    "DEFAULT_QWEN3_VL_8B_TO_QWEN3_8B",
    "DEFAULT_QWEN3_VL_2B_TO_QWEN3_VL_2B",
    "DEFAULT_QWEN3_VL_2B_TO_QWEN3_8B",
    "DimensionAligner",
    "InjectedInputs",
    "MemoryAugmentedLLM",
    "MemoryInjector",
    "QwenVLTextPairConfig",
    "ScreenCompressor",
    "resolve_torch_dtype",
]
