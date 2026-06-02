"""
模型接口和实现
"""

from .service.base import ModelInterface
from .openai_model import OpenAIModel

__all__ = [
    "ModelInterface",
    "OpenAIModel",
]