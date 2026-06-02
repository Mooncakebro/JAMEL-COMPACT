import json
from typing import TYPE_CHECKING, List
from jamel.core.env.web.utils import StepHistory
if TYPE_CHECKING:
    from jamel.core.policy.agent import PolicyAgent

class WindowMemory:
    def __init__(self, policy_agent: 'PolicyAgent'):
        self.policy_agent = policy_agent
        self.memory_cell = []

    def reset(self):
        self.memory_cell = []

    def update_memory(self):
        self.memory_cell = self.policy_agent.history[-3:]

    def get_memory(self):
        recent_history = json.dumps([step_history.before_observation for step_history in self.memory_cell], ensure_ascii=False) # 将来会替换为 memory 机制
        return str(recent_history)