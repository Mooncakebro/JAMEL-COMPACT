from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class InjectedInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    prefix_length: int
    prompt_lengths: torch.Tensor

    def to_model_kwargs(self) -> dict[str, torch.Tensor]:
        model_kwargs = {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask": self.attention_mask,
        }
        if self.position_ids is not None:
            model_kwargs["position_ids"] = self.position_ids
        return model_kwargs


class DimensionAligner(nn.Module):
    def __init__(self, src_dim: int, tgt_dim: int) -> None:
        super().__init__()
        self.src_dim = int(src_dim)
        self.tgt_dim = int(tgt_dim)

        if self.src_dim == self.tgt_dim:
            self.proj = None
        else:
            self.proj = nn.Linear(self.src_dim, self.tgt_dim)
        self.norm = nn.LayerNorm(self.tgt_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return self.norm(hidden_states)
        return self.norm(self.proj(hidden_states))


class MemoryInjector(nn.Module):
    def __init__(self, *, create_position_ids: bool = True) -> None:
        super().__init__()
        self.create_position_ids = create_position_ids

    def inject(
        self,
        *,
        model: nn.Module,
        memory_tokens: torch.Tensor,
        memory_attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> InjectedInputs:
        if memory_tokens.ndim == 2:
            memory_tokens = memory_tokens.unsqueeze(1)
        if memory_tokens.ndim != 3:
            raise ValueError("memory_tokens must have shape [B, H] or [B, N, H].")
        if memory_attention_mask is not None and memory_attention_mask.ndim == 1:
            memory_attention_mask = memory_attention_mask.unsqueeze(0)

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            embed_layer = model.get_input_embeddings()
            embed_device = embed_layer.weight.device
            input_ids = input_ids.to(embed_device)
            inputs_embeds = embed_layer(input_ids)
        else:
            embed_device = inputs_embeds.device

        batch_size = inputs_embeds.shape[0]
        prefix_length = memory_tokens.shape[1]
        memory_tokens = memory_tokens.to(device=embed_device, dtype=inputs_embeds.dtype)
        if memory_attention_mask is None:
            memory_attention_mask = torch.ones(
                (batch_size, prefix_length),
                dtype=torch.long,
                device=embed_device,
            )
        else:
            memory_attention_mask = memory_attention_mask.to(
                device=embed_device,
                dtype=torch.long,
            )
        if memory_attention_mask.shape != (batch_size, prefix_length):
            raise ValueError(
                "memory_attention_mask must match memory_tokens batch and prefix dimensions."
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, inputs_embeds.shape[1]),
                dtype=torch.long,
                device=embed_device,
            )
        else:
            attention_mask = attention_mask.to(device=embed_device, dtype=torch.long)

        prompt_lengths = attention_mask.sum(dim=-1)
        combined_inputs_embeds = torch.cat([memory_tokens, inputs_embeds], dim=1)
        combined_attention_mask = torch.cat([memory_attention_mask, attention_mask], dim=1)
        combined_position_ids = self._build_position_ids(
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            prefix_length=prefix_length,
        )

        return InjectedInputs(
            inputs_embeds=combined_inputs_embeds,
            attention_mask=combined_attention_mask,
            position_ids=combined_position_ids,
            prefix_length=prefix_length,
            prompt_lengths=prompt_lengths,
        )

    def _build_position_ids(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        prefix_length: int,
    ) -> torch.Tensor | None:
        if position_ids is not None:
            position_ids = position_ids.to(attention_mask.device)
            if position_ids.ndim == 2:
                prefix_positions = torch.arange(
                    prefix_length,
                    device=attention_mask.device,
                    dtype=position_ids.dtype,
                ).unsqueeze(0).expand(position_ids.shape[0], -1)
                shifted_positions = position_ids + prefix_length
                return torch.cat([prefix_positions, shifted_positions], dim=1)
            if position_ids.ndim == 3:
                return self._build_multimodal_position_ids(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    prefix_length=prefix_length,
                )
            raise ValueError(
                f"Unsupported position_ids rank: expected 2 or 3, got {position_ids.ndim}."
            )

        if not self.create_position_ids:
            return None

        generated_position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return generated_position_ids.masked_fill(attention_mask == 0, 0)

    def _build_multimodal_position_ids(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prefix_length: int,
    ) -> torch.Tensor:
        batch_size = attention_mask.shape[0]
        # Match the native Qwen3-VL convention used by transformers:
        # rank-3 multimodal position_ids are channel-first [C, B, S], usually C=4
        # where channel 0 is the text position track and channels 1..3 are MRoPE tracks.
        if position_ids.shape[1] != batch_size:
            raise ValueError(
                "Rank-3 multimodal position_ids must be channel-first [C, B, S]. "
                f"Got shape={tuple(position_ids.shape)} with batch_size={batch_size}."
            )

        prefix_positions = torch.arange(
            prefix_length,
            device=attention_mask.device,
            dtype=position_ids.dtype,
        )
        num_channels = position_ids.shape[0]
        prefix_positions = prefix_positions.view(1, 1, prefix_length).expand(
            num_channels, batch_size, prefix_length
        )
        shifted_positions = position_ids + prefix_length
        return torch.cat([prefix_positions, shifted_positions], dim=-1)
