'''
用 transformers 库直接推理
'''
from pathlib import Path
import re
import PIL.Image
import torch

from typing import NamedTuple
from PIL.Image import Image
import PIL
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
import structlog

from .base import ModelInterface
from jamel.utils.general_utils import summarize_str

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)

IMAGE_START="{|{|"
IMAGE_END="|}|}"

def _get_text_and_images(raw_messages):
    '''
    将 OpenAI 格式的 messages 转换成 text 和 images（目前尚未支持 Images）
    '''
    # from qwen_vl_utils import process_vision_info
    # image_inputs, video_inputs = process_vision_info(raw_messages)
    # images = image_inputs
    images = []
    image_str_count = 0
    
    processed_messages = []
    for raw_message in raw_messages:
        content = raw_message['content']
        if isinstance(content, str):
            processed_message = {"role": raw_message['role'], "content": content}
            processed_messages.append(processed_message)
        elif isinstance(content, list):
            content_str = ""
            for content_item in content:
                if content_item['type'] == 'text':
                    content_str += content_item['text']
                elif content_item['type'] == 'image':
                    content_str += "<image>"
                    image_str_count += 1
            processed_message = {"role": raw_message['role'], "content": content_str}
            processed_messages.append(processed_message)
    assert image_str_count == len(images)
    return processed_messages, images

def get_regex_processor(regex_pattern: str, tokenizer):
    import xgrammar as xgr
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_regex(regex_pattern)

    # 3. 创建 Hugging Face 兼容的 LogitsProcessor
    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    return xgr_logits_processor

class Qwen2Model(ModelInterface):
    '''
    兼容 qwen2 和 qwen2.5 的文本模型
    '''
    def __init__(self, model_path, jinja_template_path=None):
        super().__init__(model_path=model_path)
        # 初始化模型（可换成你自己的 checkpoint）
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path)
        if jinja_template_path is not None:
            logger.info(f"load custom jinja template: {jinja_template_path}")
            self.tokenizer.chat_template = Path(jinja_template_path).read_text()

    def get_chat_response(self, **kwargs):
        from qwen_vl_utils import process_vision_info
        params = kwargs.copy()
        context_logger = logger

        model = params.pop("model")
        chat_template_kwargs = params.pop('chat_template_kwargs', {})
        extra_body: dict = params.pop('extra_body', {})

        # process regex
        if 'regex_pattern' in extra_body:
            regex_pattern = extra_body.pop('regex_pattern')
            regex_processor = get_regex_processor(regex_pattern=regex_pattern, tokenizer=self.tokenizer)
            logger.info(f"load regex processor", regex_pattern=regex_pattern)
            extra_body['logits_processor'] = regex_processor

        raw_messages = params.pop("messages")
        messages, _ = _get_text_and_images(raw_messages)
        context_logger.info("reshape messages for pure text...")
        
        params.pop("stream", None)
        # set default max new tokens to 100
        max_tokens = params.pop("max_tokens", 100)
        params.setdefault("max_new_tokens", max_tokens)
        
        context_logger = context_logger.bind(**params)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            **chat_template_kwargs
        )
        context_logger.info(f"Applied template: {summarize_str(500)(text)}")
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        context_logger.info("Generating response...")
        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **params,
                **extra_body
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        raw_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        context_logger=context_logger.bind(response=summarize_str(200)(raw_response))
        context_logger.info("Generated response:")

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response}
                }
            ]
        }

class Qwen2VLModel(ModelInterface):
    '''
    兼容 qwen2 和 qwen2.5 的多模态模型
    '''
    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        # 初始化模型（可换成你自己的 checkpoint）
        self.model: Qwen2VLForConditionalGeneration = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.think_start_token = "<think>"
        self.think_end_token = "</think>"
        self.answer_start_token = "<answer>"
        self.answer_end_token = "</answer>"

    def preprocess_messages(self, messages: list[dict]):
        '''
        openai format: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"} 
        however, qwen format: {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
        That's why we need to convert from openai to qwen.
        '''
        for message in messages:
            if isinstance(message['content'], list):
                for content_item in message['content']:
                    if isinstance(content_item, dict) and 'image_url' in content_item and isinstance(content_item['image_url'], dict) and 'url' in content_item['image_url']:
                        content_item['image_url'] = content_item['image_url']['url']

    def get_chat_response(self, **kwargs):
        from qwen_vl_utils import process_vision_info
        params = kwargs.copy()
        context_logger = logger
        model = params.pop("model")
        messages = params.pop("messages")
        params.pop("stream", None)
        thinking = params.pop("thinking", None)
        # set default max new tokens to 4096
        max_tokens = params.pop("max_tokens", 4096)
        params.setdefault("max_new_tokens", max_tokens)
        self.preprocess_messages(messages)
        context_logger = context_logger.bind(messages=(lambda x: x[:100] + f"...({len(x) - 200} more chars)" + x[-100:] if len(x) > 200 else x)(str(messages)), **params)
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # process thinking token (usually processed in jinja)
        if thinking is not None:
            if thinking.get('type', 'disabled') == 'enabled':
                text = text + self.think_start_token
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **params
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        context_logger.bind(output_len=len(generated_ids_trimmed))
        context_logger.info(f"Generated response: {output_texts}")

        def process_output_text(text: str):
            if self.think_end_token in text:
                reasoning, answer = text.split(self.think_end_token, maxsplit=1) # 标准的做法本应是根据 token id 进行 split
                answer = answer.strip().removeprefix(self.answer_start_token).removesuffix(self.answer_end_token) # 同理，应该用 token id 的
                return {
                    "content": answer,
                    "reasoning_content": reasoning
                }
            return {
                "content": text,
            }
                
        return {
            "choices": [
                {
                    "message": {"role": "assistant", **process_output_text(output_text)}
                } for output_text in output_texts
            ]
        }

class Qwen3Model(Qwen2Model):
    pass

class Qwen2BaseCompletionModel(ModelInterface):
    """
    基于预训练基础模型的文本补全实现
    支持图文交错输入的补全任务
    """
    def __init__(self, model_path):
        super().__init__(model_path=model_path)
        # 初始化基础模型（未经过指令微调）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True  # 某些模型可能需要
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 对于多模态基础模型，可能需要额外的处理器
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.is_multimodal = True
        except:
            self.processor = self.tokenizer
            self.is_multimodal = False

    def get_completion_response(self, prompt: str, images: list = None, **kwargs):
        """
        文本补全接口
        Args:
            prompt: 文本提示词（可以包含 <image> 标记）
            images: PIL Image 对象列表
            **kwargs: 其他生成参数
        """
        params = kwargs.copy()
        context_logger = logger

        # 设置默认参数
        model = params.pop("model")
        params.pop("stream", None)
        max_tokens = params.pop("max_tokens", 2048)
        params.setdefault("max_new_tokens", max_tokens)
        params.setdefault("do_sample", True)
        params.setdefault("temperature", 0.7)
        params.setdefault("top_p", 0.9)

        context_logger = context_logger.bind(
            prompt_length=len(prompt),
            num_images=len(images) if images else 0,
            **params
        )

        if self.is_multimodal and images:
            # 多模态处理
            context_logger.info("Processing multimodal completion...")

            # 使用 processor 处理图文输入
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        else:
            # 纯文本处理
            context_logger.info("Processing text-only completion...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        context_logger = context_logger.bind(inputs=summarize_str(500)(inputs))
        context_logger.info("Generating completion...")

        self.model.eval()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **params
            )

        # 只保留生成的部分（去掉输入 prompt）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码生成的文本
        if self.is_multimodal and images:
            completion_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        else:
            completion_texts = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        raw_completion = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        completion_text = completion_texts[0] if completion_texts else ""

        context_logger.bind(
            completion_length=len(completion_text),
            completion_text=summarize_str(500)(raw_completion)
        ).info("Completion generated")

        # 返回 OpenAI 兼容格式的补全响应
        return {
            "choices": [
                {
                    "text": completion_text,
                    "index": 0,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
            }
        }

    def get_chat_response(self, **kwargs):
        """
        为了兼容现有接口，将补全包装成 chat 格式。实际上就是拼接了所有前缀。
        """
        logger.warning("Base 模型没有 chat 接口！因此采用代码补全的方式替代聊天。")
        params = kwargs.copy()
        messages = params.pop("messages", [])

        if not messages:
            return {"choices": [{"message": {"content": ""}}]}

        # 从 messages 中提取最后的 user 消息作为补全的 prompt
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg
                break

        if not user_message:
            return {"choices": [{"message": {"content": ""}}]}

        content = user_message.get("content", "")

        # 处理图文内容
        images = []
        if isinstance(content, list):
            # 多模态消息格式
            prompt_text = ""
            for item in content:
                if item.get("type") == "text":
                    prompt_text += item.get("text", "")
                elif item.get("type") == "image_url":
                    # TODO: 处理 base64 图像
                    pass
        else:
            # 纯文本消息
            prompt_text = content

        # 调用补全接口
        completion_result = self.get_completion_response(
            prompt=prompt_text,
            images=images,
            **params
        )

        # 将补全结果转换为 chat 格式
        completion_text = completion_result["choices"][0]["text"]

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    }
                }
            ],
            "usage": completion_result.get("usage", {})
        }

class Qwen2VLBaseCompletionModel(Qwen2BaseCompletionModel):
    """
    专门用于 Qwen2VL 基础模型的补全实现
    """
    def __init__(self, model_path):
        super().__init__(model_path)
        # 使用 VL 专用的模型和处理器
        self.model: Qwen2VLForConditionalGeneration = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.is_multimodal = True

# # 获取结果文本
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
    args = parser.parse_args()

    plan_model = Qwen2VLModel(args.model_path)
