from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from jamel.config.settings import get_settings
from jamel.log import log_utils
from jamel.models.service.base import ModelInterface

logger = log_utils.get_logger(__name__)


def _normalize_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for raw_message in raw_messages:
        content = raw_message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": raw_message["role"], "content": content})
            continue
        if isinstance(content, list):
            content_str = ""
            for item in content:
                if item.get("type") == "text":
                    content_str += item.get("text", "")
                else:
                    raise NotImplementedError(
                        "The online FSDP PG pipeline currently supports text-only messages."
                    )
            messages.append({"role": raw_message["role"], "content": content_str})
            continue
        raise TypeError(f"Unsupported message content type: {type(content)!r}")
    return messages


def _top_p_sample(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        next_token = torch.argmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprob = log_probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
        return next_token, token_logprob

    scaled_logits = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled_logits, dim=-1)

    if 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sample_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        next_token = sorted_indices.gather(-1, sample_idx.unsqueeze(-1)).squeeze(-1)
        selected_prob = sorted_probs.gather(-1, sample_idx.unsqueeze(-1)).squeeze(-1)
        token_logprob = torch.log(selected_prob.clamp_min(1e-12))
        return next_token, token_logprob

    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_logprob = torch.log(
        probs.gather(-1, next_token.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    )
    return next_token, token_logprob


class FSDPTextPolicyModel(ModelInterface):
    def __init__(self, model_path: str, model, tokenizer, device: torch.device):
        super().__init__(model_path=model_path)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.settings = get_settings()

        inference_kwargs = dict(self.settings.inference_kwargs)
        self.default_chat_template_kwargs = dict(
            inference_kwargs.pop("chat_template_kwargs", {})
        )
        self.default_generation_kwargs = inference_kwargs

    def _prepare_prompt(
        self,
        messages: List[Dict[str, Any]],
        chat_template_kwargs: dict,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        processed_messages = _normalize_messages(messages)
        prompt_text = self.tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            **chat_template_kwargs,
        )
        model_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)
        return prompt_text, input_ids, attention_mask

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[List[int], List[float], str]:
        generated_ids: List[int] = []
        generated_logprobs: List[float] = []
        stop_token_ids = set(self.tokenizer.all_special_ids or [])
        stop_reason = "max_tokens"

        current_input_ids = input_ids
        current_attention_mask = attention_mask

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token, token_logprob = _top_p_sample(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                )

                token_id = int(next_token.item())
                generated_ids.append(token_id)
                generated_logprobs.append(float(token_logprob.item()))

                next_token_tensor = next_token.view(1, 1).to(current_input_ids.device)
                current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=-1)
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=current_attention_mask.dtype,
                            device=self.device,
                        ),
                    ],
                    dim=-1,
                )

                if token_id in stop_token_ids:
                    stop_reason = "special_token"
                    break

        return generated_ids, generated_logprobs, stop_reason

    def get_chat_response(
        self,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        params = {**self.default_generation_kwargs, **kwargs}
        extra_body = params.pop("extra_body", {}) or {}
        params.update(extra_body)
        params.pop("stream", None)

        chat_template_kwargs = {
            **self.default_chat_template_kwargs,
            **params.pop("chat_template_kwargs", {}),
        }
        max_tokens = int(params.pop("max_tokens", self.settings.rollout_max_tokens))
        temperature = float(params.pop("temperature", self.settings.rollout_temperature))
        top_p = float(params.pop("top_p", self.settings.rollout_top_p))

        prompt_text, input_ids, attention_mask = self._prepare_prompt(
            messages=messages,
            chat_template_kwargs=chat_template_kwargs,
        )
        generated_ids, generated_logprobs, stop_reason = self._generate_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        rollout_chunk = {
            "prompt_text": prompt_text,
            "prompt_token_ids": input_ids[0].tolist(),
            "completion_token_ids": list(generated_ids),
            "completion_token_logprobs": list(generated_logprobs),
            "completion_logprob_sum": float(sum(generated_logprobs)),
            "response_text": response_text,
            "stop_reason": stop_reason,
            "temperature": temperature,
            "top_p": top_p,
        }

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": int(input_ids.shape[-1]),
                "completion_tokens": len(generated_ids),
                "total_tokens": int(input_ids.shape[-1]) + len(generated_ids),
            },
            "rollout_data": {
                "chunks": [rollout_chunk],
                "completion_logprob_sum": rollout_chunk["completion_logprob_sum"],
            },
        }
