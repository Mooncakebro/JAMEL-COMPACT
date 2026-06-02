from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from ..types import BaseStepHistory

@dataclass
class StepHistory(BaseStepHistory):
    before_obs: dict
    after_obs: dict
    before_info: dict
    after_info: dict
    before_observation: str
    after_observation: str
    step: Any
    reward: float
    raw_content: str # llm 解析前的结果
    memory_content: Any
    parsed_content: Dict[str, str] # llm 解析后的结果
    result: Dict[str, Any]
    timestamp: Any
    extra_fields: Dict[str, Dict|List|str|bool|int|float] = None # 用来包括额外的字段。必须是可序列化的（基本类型），否则会报错。

    def to_dict(self):
        '''
        只能包含可以序列化的数据。
        '''
        return {
            "step": self.step,
            "before_observation_str": self.before_observation,
            "after_observation_str": self.after_observation,
            "reward": self.reward,
            "memory_content": self.memory_content,
            "raw_content": self.raw_content,
            "parsed_content": self.parsed_content,
            "result": self.result,
            "timestamp": str(self.timestamp),
            "extra_fields": self.extra_fields
        }
