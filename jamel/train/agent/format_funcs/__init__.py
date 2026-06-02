from jamel.core.memory.types import MemoryType
from jamel.train.agent.format_funcs.base import FormatFunc
from .context_memory import format_web_explorer_example_w_context_memory
from .parametric_memory import (
    format_web_explorer_example_w_parametric_memory,
    format_web_explorer_example_w_parametric_memory_policy,
)

memory_format_func_router = {
    # MemoryType.explicit.name: format_web_explorer_example_w_context_memory,
    # MemoryType.window.name: format_web_explorer_example_w_context_memory,
    MemoryType.parametric.name: format_web_explorer_example_w_parametric_memory,
}

def get_memory_format_func(memory_type: 'MemoryType') -> FormatFunc:
    return memory_format_func_router[memory_type]

policy_format_func_router = {
    MemoryType.explicit.name: format_web_explorer_example_w_context_memory,
    MemoryType.window.name: format_web_explorer_example_w_context_memory,
    MemoryType.parametric.name: format_web_explorer_example_w_parametric_memory_policy,
}

def get_policy_format_func(memory_type: 'MemoryType') -> FormatFunc:
    return policy_format_func_router[memory_type]
