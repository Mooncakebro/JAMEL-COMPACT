"""
OpenAI 模型封装
"""

import openai
from typing import Dict, List, Any, Optional
from jamel.config import get_settings
from .service.base import ModelInterface
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)

class OpenAIModel(ModelInterface):
    """OpenAI API 模型封装"""

    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        super().__init__(model_path=model_name)
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.base_url = base_url
        self.api_key = api_key
        self._setup_client()

    def _setup_client(self):
        """设置 OpenAI 客户端"""
        self.client = openai.OpenAI(
            api_key=self.api_key or self.settings.openai_api_key,
            base_url=self.base_url or self.settings.openai_base_url
        )


    def get_chat_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        获取 OpenAI 格式的响应
        """
        model = model or self.model_name
        inference_kwargs = {**self.settings.inference_kwargs, **kwargs}
        logger.info(f"开始请求：", model=model, inference_kwargs=inference_kwargs)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=inference_kwargs
        )
        logger.info(f"请求成功", model=model, response=response)

        return {
            "choices": [
                {
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason,
                    "index": choice.index
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else {}
        }
            

    def get_completion_response(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        获取补全响应（转换为 chat 格式）
        """
        messages = [{"role": "user", "content": prompt}]
        return self.get_chat_response(messages, **kwargs)