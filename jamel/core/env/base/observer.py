
from typing import Any, Dict

class Observer:
    """观察者，将 gym 转换为提供给 Agent 观察的格式"""
    @staticmethod
    def get_observation(obs: Dict) -> Any:
        pass