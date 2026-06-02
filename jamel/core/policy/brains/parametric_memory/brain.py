import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from functools import partial

import tenacity
from jamel.core.env.web.utils import StepHistory
from jamel.core.memory.base import MemoryBase
from jamel.models.service.base import ModelInterface
from jamel.log import log_utils
from jamel.utils.response_utils import split_fields
from jamel.utils.general_utils import log_retry_error_with_traceback, no_retry_get_response_error_types

from .prompt import get_user_prompt
from ..utils import Decision

logger = log_utils.get_logger(__name__)

def generate_with_format_control(model: ModelInterface, messages: List, fields: List[Tuple[str, str]]):
    parsed_fields = {}
    full_content = ''
    for i, (key, tag) in enumerate(fields):
        if not parsed_fields.get(key):
            logger.info(f"field {key} not found in the response! try to force generate.", missing_key=key, missing_tag=tag)
            current_prefix = "\n".join(f'{parsed_tag}\n{parsed_fields[parsed_key]}' for parsed_key, parsed_tag in fields[:i]) + f'\n{tag}\n'
            # 调用模型续写
            response = model.get_chat_response(
                messages=messages + [{"role": "assistant", "content": current_prefix}]
            )
            new_part = response["choices"][0]["message"]["content"]
            full_content = current_prefix + new_part
            logger.info(f"try to parse field from new prefix", new_prefix=full_content, fields=fields)
            parsed_fields = split_fields(full_content, fields)
    return parsed_fields

class BrainWithParametricMemory:
    # 配置期望的字段顺序
    REQUIRED_FIELDS = [
        ("memory", "--- Memory ---"),
        ("thought", "--- Thought ---"),
        ("action", "--- Action ---")
    ]

    def __init__(self, model, get_action_space, get_user_prompt=get_user_prompt):
        self.model = model
        self.get_action_space = get_action_space
        self.get_user_prompt = get_user_prompt

    def _process_and_validate(self, parsed: dict[str, str]) -> Tuple[str, dict]:
        """调用 split_fields 并校验核心字段"""
        if not parsed.get('action'):
            logger.warning("模型输出解析失败：缺少 action 字段", parsed=parsed)
            raise ValueError("Required field 'action' is missing in LLM response.")
        # action 要做裁剪，把第一行以后的操作全部裁剪掉。
        parsed_action = parsed["action"]
        parsed["action"] = parsed_action.strip().split("\n")[0] 
        logger.info("模型输出解析成功", parsed=parsed)
        
        full_content = "\n".join(f'{parsed_tag}\n{parsed[parsed_key]}' for parsed_key, parsed_tag in self.REQUIRED_FIELDS)
        return full_content, parsed
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        retry=tenacity.retry_if_not_exception_type(no_retry_get_response_error_types),
        wait=tenacity.wait_fixed(30),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        after=partial(log_retry_error_with_traceback, handler=lambda s: logger.info(f"Attempt failed", error=s))
    )
    def decide_action(
        self,
        observation: str,
        user_goal: str,
        history: List[StepHistory],
        memory: MemoryBase,
    ) -> Decision:
        # 1. 准备数据
        user_prompt = self.get_user_prompt(observation=observation, action_space=self.get_action_space())
        messages = [{"role": "user", "content": user_prompt}]
        logger.info("开始决策生成", observation=observation, user_goal=user_goal)

        # 2. 核心生成逻辑：交给外部函数处理繁琐的迭代
        parsed_content = generate_with_format_control(
            model=self.model,
            messages=messages,
            fields=self.REQUIRED_FIELDS
        )

        # 3. 解析与校验
        full_content, parsed_content = self._process_and_validate(parsed_content) # 去掉了多余的部分
        
        logger.info("LLM 决策成功", parsed_content=parsed_content)

        return Decision(
            parsed_content=parsed_content,
            memory_content=parsed_content['memory'],
            raw_content=full_content,
            llm_response={"choices": [{"message": {"content": full_content}}]} 
        )
