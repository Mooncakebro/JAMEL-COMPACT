"""
JAMEL-COMPACT: Unified LLM with per-layer side memory.

Subpackage containing the model, data, training, and evaluation code for
the JAMEL-COMPACT framework — a single pretrained LLM (e.g. Qwen3-VL-2B/8B)
augmented with per-layer side memory modules that evolve via FiLM-GRU
prediction and Kalman-Filter correction.
"""

from .model import JAMELCompactWrapper, SideMemoryModule, FiLMGRUCell
from .config import CompactConfig

__all__ = [
    "JAMELCompactWrapper",
    "SideMemoryModule",
    "FiLMGRUCell",
    "CompactConfig",
]