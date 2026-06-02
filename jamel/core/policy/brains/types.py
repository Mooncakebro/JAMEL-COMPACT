
from enum import Enum, auto

from jamel.core.policy.brains.base import BrainBase
from jamel.core.policy.brains.explicit_memory.prompt import get_user_prompt as explicit_memory_get_user_prompt
from jamel.core.policy.brains.naive.prompt import get_user_prompt as naive_get_user_prompt
from jamel.core.policy.brains.parametric_memory.prompt import get_user_prompt as parametric_memory_get_user_prompt
from .. import brains

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)

class BrainType(Enum):
    naive = auto()
    explicit_memory = auto()
    parametric_memory = auto()

BRAIN_CLS_MAPPING = {
    BrainType.naive.name: brains.NaiveBrain,
    BrainType.explicit_memory.name: brains.BrainWithExplicitMemory,
    BrainType.parametric_memory.name: brains.BrainWithParametricMemory,
}

BRAIN_PROMPT_FUNC_MAPPING = {
    BrainType.naive.name: naive_get_user_prompt,
    BrainType.explicit_memory.name: explicit_memory_get_user_prompt,
    BrainType.parametric_memory.name: parametric_memory_get_user_prompt,
}

def get_brain_cls(brain_type: BrainType, brain_cls_mapping=None) -> BrainBase:
    logger.info(f"trying to load brain type: {brain_type}")
    brain_cls_mapping = brain_cls_mapping or BRAIN_CLS_MAPPING
    brain_cls = brain_cls_mapping[brain_type]
    logger.info(f"loaded brain cls: {brain_cls}")
    return brain_cls


def get_brain_prompt_func(brain_type: BrainType):
    logger.info(f"trying to load brain prompt func: {brain_type}")
    prompt_func = BRAIN_PROMPT_FUNC_MAPPING[brain_type]
    logger.info(f"loaded brain prompt func: {prompt_func}")
    return prompt_func

if __name__ == "__main__":
    brain_cls = get_brain_cls('naive')
    print(brain_cls)
