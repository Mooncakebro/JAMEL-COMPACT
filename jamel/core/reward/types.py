import enum
from typing import List, Protocol
from jamel.core.env.types import BaseStepHistory

class RewardFunc(Protocol):
    def __call__(self, current_step: BaseStepHistory,
    history: List[BaseStepHistory], *args, **kwds):
        return super().__call__(*args, **kwds)
