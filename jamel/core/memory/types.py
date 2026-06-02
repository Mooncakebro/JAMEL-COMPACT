from enum import Enum, auto
import importlib
from typing import Dict

from .base import MemoryBase
from .. import memory

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)

class MemoryType(Enum):
    window = auto()
    explicit = auto()
    parametric = auto()

MEMORY_CLS_MAPPING = {
    MemoryType.window.name: memory.window.WindowMemory,
    MemoryType.explicit.name: memory.explicit.ExplicitMemory,
    MemoryType.parametric.name: memory.parametric.ParametricMemory,
}

def get_memory_cls(memory_type: MemoryType, memory_cls_mapping=None) -> MemoryBase:
    logger.info(f"trying to load memory type: {memory_type}")
    memory_cls_mapping = memory_cls_mapping or MEMORY_CLS_MAPPING
    memory_cls = memory_cls_mapping[memory_type]
    logger.info(f"loaded memory cls: {memory_cls}")
    return memory_cls

if __name__ == "__main__":
    memory_cls = get_memory_cls('window')
    print(memory_cls)
    