from __future__ import annotations
from typing import Any
from jamel.core.env.web.utils import StepHistory

class MemoryBase:
    def __init__(self, policy_agent):
        self.policy_agent = policy_agent

    def update_memory(self):
        pass

    def get_memory(self) -> Any:
        pass
    
    def reset(self):
        pass