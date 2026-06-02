"""
思考模块 - 负责 LLM 决策。需要定义动作空间提供给模型。
"""
from dataclasses import dataclass
import json
from typing import Callable, Dict, Any, List, Optional
from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.base import MemoryBase
from jamel.models.service.base import ModelInterface
from jamel.log import log_utils
from jamel.utils.response_utils import split_fields

from .prompt import get_user_prompt, get_user_prompt_completion
from ..utils import Decision

logger = log_utils.get_logger(__name__)
class NaiveBrain:
    """决策大脑，使用 LLM 做出行动决策"""

    def __init__(
        self,
        model: ModelInterface,
        get_action_space: Callable,
        get_user_prompt: Callable = get_user_prompt
    ):
        self.model = model
        self.get_action_space=get_action_space
        self.get_user_prompt = get_user_prompt

    def _parse_fields_from_raw_content(self, raw_content: str):
        field_list = [("thought", "--- Thought ---"), ("action", "--- Action ---")]
        parsed_fields = split_fields(raw_content, field_list)
        # assert parsed_fields['action'], "action should always exist!"
        return parsed_fields

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
        memory_content = memory.get_memory()

        user_prompt = get_user_prompt(observation=observation, memory=memory_content, action_space=self.get_action_space())
        # user_prompt = get_user_prompt_completion(observation=observation, memory=memory_content, action_space=self.get_action_space())

        response = self.model.get_chat_response(
            messages=[
                # {"role": "user", "content": 'Hello! Who are you?'},
                # {"role": "assistant", "content": 'I am a helpful assistant.'},
                {"role": "user", "content": user_prompt}
            ]
        )
        logger.info("LLM 响应成功", response=response)
        content = response["choices"][0]["message"]["content"]

        # 尝试解析 JSON
        parsed_content = self._parse_fields_from_raw_content(content)
        logger.info("LLM 决策成功", parsed_content=parsed_content)
        return Decision(
            parsed_content=parsed_content,
            memory_content=memory_content,
            raw_content=content,
            llm_response=response
        )
