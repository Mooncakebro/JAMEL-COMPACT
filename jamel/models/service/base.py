"""
基础模型接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class ModelInterface(ABC):
    """模型基础接口"""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def get_chat_response(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        获取 OpenAI 格式的响应

        Args:
            messages: 对话消息列表
            **kwargs: 其他参数

        Returns:
            OpenAI 格式的响应
        """
        pass