from __future__ import annotations
from typing import TYPE_CHECKING, List
from jamel.core.env.web.utils import StepHistory
if TYPE_CHECKING:
    from jamel.core.policy.agent import PolicyAgent

class ExplicitMemory:
    def __init__(self, policy_agent: 'PolicyAgent'):
        self.policy_agent = policy_agent
        self.explicit_memory_cell = None
        pass

    def update_memory(self): # 每次更新时完全覆写
        # TODO: 从 PolicyAgent 的结果中更新内容
        updated_memory_cell = self.policy_agent.history[-1].parsed_content['updated_memory']
        self.explicit_memory_cell = updated_memory_cell
    
    def get_memory(self):
        return self.explicit_memory_cell

    def reset(self):
        self.explicit_memory_cell = None