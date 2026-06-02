from typing import Callable, List
from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.base import MemoryBase
from jamel.models.service.base import ModelInterface
from .utils import Decision

class BrainBase:
    """决策大脑，使用 LLM 做出行动决策"""

    def __init__(
        self,
        model: ModelInterface,
        get_action_space: Callable,
        get_user_prompt: Callable=None
    ):
        self.model = model
        self.get_action_space=get_action_space
        self.get_user_prompt = get_user_prompt
    
    def decide_action(
        self,
        observation: str,
        user_goal: str,
        history: List[StepHistory],
        memory: MemoryBase,
    ) -> Decision:
        """
        决定下一步行动

        Args:
            observation: 页面观察结果
            user_goal: 用户目标
            history: 历史记录

        Returns:
            决策结果
        """