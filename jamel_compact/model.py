"""
JAMEL-COMPACT model: wraps a pretrained LLM with per-layer side memory.

This module implements the core architecture:
  - FiLMGRUCell: action-modulated GRU for memory state prediction
  - SideMemoryModule: per-layer memory with Predict→Correct→Inject cycle
  - JAMELCompactWrapper: wraps a HuggingFace LLM (e.g. Qwen3-VL-2B/8B)

The pretrained LLM's self-attention and FFN are NOT replaced — they are
called in-place.  Only the side memory modules are new parameters.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from .config import CompactConfig


# ═══════════════════════════════════════════════════════════════════════════════
# FiLM-GRU Cell
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMGRUCell(nn.Module):
    """
    FiLM-modulated GRU cell for memory state prediction.

    The action embedding modulates the GRU state transition via
    Feature-wise Linear Modulation (FiLM):

        γ, β = MLP(a_emb)
        h_new = GRU(W_proj(a_emb), γ ⊙ h_old + β)

    This makes the action an explicit *control variable* that steers
    how memory evolves — analogous to u_t in state-space models.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
        )
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_prev: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: [B*N_m, d_mem] — previous memory state (flattened)
            a_emb:  [B*N_m, d_mem] — action embedding (expanded to all tokens)
        Returns:
            h_new:  [B*N_m, d_mem] — predicted memory state
        """
        gamma_beta = self.film_mlp(a_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h_modulated = gamma * h_prev + beta
        gru_input = self.action_proj(a_emb)
        return self.gru(gru_input, h_modulated)


# ═══════════════════════════════════════════════════════════════════════════════
# Side Memory Module (per layer)
# ═══════════════════════════════════════════════════════════════════════════════

class SideMemoryModule(nn.Module):
    """
    Per-layer side memory with reduced dimension d_mem.

    Dimension flow:
      • Main stream H, observation Z_t, action embed:  d (e.g. 2048 or 4096)
      • Memory state M, FiLM-GRU, cross-attn, Kalman gate:  d_mem (e.g. 512)
      • Down-projections: d → d_mem  (before memory operations)
      • Up-projections:   d_mem → d  (before injecting back into main stream)

    Algorithm per time step t, per layer l:
      1. Predict:  M̂ = FiLM-GRU(M_{t-1}, x_act^{t-1})
                   Ĉ = λ_l · C_{t-1}
      2. Observe:  Z_t = Pool(H_self_attn[img, inst])
      3. Correct:  ΔM = CrossAttn(Q=M̂, KV=Z_t)
                   K  = σ(W[Z_t; M̂]) ⊙ (1-Ĉ)
                   M  = M̂ + K ⊙ ΔM
                   C  = Ĉ + α(1-Ĉ)·cos_sim(Z_t, M̂)
      4. Inject:   H̃ = H + W_inj · CrossAttn(Q=H, KV=M)
    """

    def __init__(self, layer_idx: int, num_layers: int, hidden_dim: int,
                 mem_dim: int = 512, num_mem: int = 16, num_heads: int = 8,
                 config: Optional[CompactConfig] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_mem = num_mem
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim

        # ── Down/up projections (d ↔ d_mem) ──
        self.obs_down = nn.Linear(hidden_dim, mem_dim)
        self.action_down = nn.Linear(hidden_dim, mem_dim)
        self.h_down = nn.Linear(hidden_dim, mem_dim)
        self.delta_up = nn.Linear(mem_dim, hidden_dim)

        # ── Memory Predict: FiLM-GRU (in d_mem) ──
        self.gru = FiLMGRUCell(mem_dim)

        # ── Memory Update: Innovation cross-attention (in d_mem) ──
        self.mem_cross_attn = nn.MultiheadAttention(
            mem_dim, num_heads, batch_first=True,
        )
        self.innovation_proj = nn.Linear(mem_dim, mem_dim)

        # ── Kalman Gain gate (in d_mem) ──
        self.k_gate = nn.Sequential(
            nn.Linear(mem_dim * 2, mem_dim),
            nn.Sigmoid(),
        )

        # ── Memory Injection cross-attention (in d_mem) ──
        self.inject_cross_attn = nn.MultiheadAttention(
            mem_dim, num_heads, batch_first=True,
        )
        self.inject_norm = nn.LayerNorm(hidden_dim)

        # ── Hierarchical hyperparameters ──
        if config is not None:
            lam_s, lam_m, lam_d = (config.lambda_shallow, config.lambda_mid,
                                   config.lambda_deep)
            inj_s, inj_m, inj_d = (config.inject_shallow, config.inject_mid,
                                   config.inject_deep)
            self.alpha = config.alpha_confidence
        else:
            lam_s, lam_m, lam_d = 0.70, 0.85, 0.95
            inj_s, inj_m, inj_d = 0.8, 0.5, 0.3
            self.alpha = 0.1

        if layer_idx < num_layers // 3:
            self.lambda_decay, self.inject_weight = lam_s, inj_s
        elif layer_idx < 2 * num_layers // 3:
            self.lambda_decay, self.inject_weight = lam_m, inj_m
        else:
            self.lambda_decay, self.inject_weight = lam_d, inj_d

        # ── Learnable initial memory (in d_mem) ──
        self.init_memory = nn.Parameter(torch.randn(num_mem, mem_dim) * 0.02)

    def predict(self, m_prev: torch.Tensor, c_prev: torch.Tensor,
                action_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """FiLM-GRU predict + confidence decay. All in d_mem space."""
        B, N_m, d_mem = m_prev.shape
        a_down = self.action_down(action_embed)  # [B, d_mem]
        m_prev_flat = m_prev.reshape(B * N_m, d_mem)
        a_flat = a_down.unsqueeze(1).expand(-1, N_m, -1).reshape(B * N_m, d_mem)
        m_hat = self.gru(m_prev_flat, a_flat).view(B, N_m, d_mem)
        c_hat = self.lambda_decay * c_prev
        return m_hat, c_hat

    def correct(self, m_hat: torch.Tensor, c_hat: torch.Tensor,
                z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kalman filter update. Cross-attention in d_mem space."""
        B, N_m, d_mem = m_hat.shape
        z_down = self.obs_down(z_t)       # [B, d_mem]
        z_exp = z_down.unsqueeze(1)       # [B, 1, d_mem]

        # Innovation (per-token)
        delta_raw, _ = self.mem_cross_attn(m_hat, z_exp, z_exp)
        delta_m = self.innovation_proj(delta_raw)

        # Kalman gain (per-token)
        z_broadcast = z_down.unsqueeze(1).expand(-1, N_m, -1)
        k_base = self.k_gate(torch.cat([z_broadcast, m_hat], dim=-1))
        k_gain = k_base * (1.0 - c_hat).unsqueeze(-1)

        # Update memory
        m_new = m_hat + k_gain * delta_m

        # Update confidence (per-token)
        z_norm = F.normalize(z_down, dim=-1)
        m_norm = F.normalize(m_hat, dim=-1)
        match = (z_norm.unsqueeze(1) * m_norm).sum(dim=-1).clamp(0, 1)
        c_new = c_hat + self.alpha * (1.0 - c_hat) * match

        return m_new, c_new

    def inject(self, h: torch.Tensor, m_new: torch.Tensor) -> torch.Tensor:
        """Read memory into main stream via cross-attention."""
        h_down = self.h_down(h)
        delta_down, _ = self.inject_cross_attn(h_down, m_new, m_new)
        delta_up = self.delta_up(delta_down)
        return self.inject_norm(h + self.inject_weight * delta_up)

    def extract_observation(self, h: torch.Tensor, n_act: int = 1) -> torch.Tensor:
        """Pool all hidden states into a single observation vector [B, d].

        Note: Unlike the prototype, the real VLM input sequence doesn't have
        a separate action token at position 0. The sequence is text+image+text
        from the processor. We pool ALL tokens to get a summary observation.
        """
        return h.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# JAMEL-COMPACT Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class JAMELCompactWrapper(nn.Module):
    """
    Wraps a pretrained HuggingFace LLM (e.g. Qwen3-VL-2B/8B) with per-layer
    side memory, WITHOUT modifying the pretrained model's internal structure.

    The pretrained layer's self_attn and FFN are called in-place — their
    weights are loaded from the checkpoint.  Only SideMemoryModule parameters
    are new (randomly initialized).

    Supports save/load of the full model (base + side memory) and standalone
    side-memory-only checkpoints.
    """

    def __init__(self, config: CompactConfig):
        super().__init__()
        self.config = config

        # ── Load pretrained LLM ──
        # Qwen3-VL is a multimodal model — AutoModelForCausalLM won't work.
        # We try causal first (for text-only models like Qwen3-8B), then fall
        # back to ImageTextToText (for vision-language models like Qwen3-VL).
        dtype = torch.bfloat16 if config.bf16 else torch.float32
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except (ValueError, OSError) as e:
            print(f"[model] AutoModelForCausalLM failed ({e}), "
                  f"trying AutoModelForImageTextToText...")
            from transformers import AutoModelForImageTextToText
            self.llm = AutoModelForImageTextToText.from_pretrained(
                config.base_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name, trust_remote_code=True,
        )
        try:
            self.processor = AutoProcessor.from_pretrained(
                config.base_model_name, trust_remote_code=True,
            )
        except Exception:
            self.processor = None

        # ── Infer architecture ──
        self.hidden_dim = self._infer_hidden_size(self.llm)
        self.num_layers = self._infer_num_layers(self.llm)
        self.num_mem = config.num_mem_tokens
        self.mem_dim = config.mem_dim
        self.num_act_tokens = config.num_act_tokens

        # ── Create side memory modules (NEW parameters) ──
        self.side_memories = nn.ModuleList([
            SideMemoryModule(
                l, self.num_layers, self.hidden_dim,
                mem_dim=config.mem_dim,
                num_mem=config.num_mem_tokens,
                num_heads=config.num_heads,
                config=config,
            )
            for l in range(self.num_layers)
        ])

        # ── Action embedding (NEW) ──
        self.action_embed = nn.Linear(self.hidden_dim, self.hidden_dim)

        # ── Cast new modules to the same dtype as the pretrained LLM ──
        # (All new parameters are created in float32 by default)
        llm_dtype = next(self.llm.parameters()).dtype
        self.action_embed = self.action_embed.to(dtype=llm_dtype)
        for sm in self.side_memories:
            sm.to(dtype=llm_dtype)

        # ── Optionally freeze the base LLM ──
        if config.freeze_base:
            for param in self.llm.parameters():
                param.requires_grad = False

        if config.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False

    # ── Architecture helpers ──

    @staticmethod
    def _infer_hidden_size(model) -> int:
        config = model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            return int(config.text_config.hidden_size)
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise ValueError("Cannot infer hidden_size")

    @staticmethod
    def _infer_num_layers(model) -> int:
        config = model.config
        if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
            return int(config.text_config.num_hidden_layers)
        if hasattr(config, "num_hidden_layers"):
            return int(config.num_hidden_layers)
        raise ValueError("Cannot infer num_layers")

    def _get_decoder_layers(self):
        # Try multiple paths to find decoder layers across model architectures.
        # Cast to list to check if already a ModuleList/Sequential.
        model = self.llm

        candidates = [
            # Qwen3-VL via AutoModelForImageTextToText:
            #   model (Qwen3VLModel) → language_model → layers
            lambda m: getattr(getattr(m, "model", None), "language_model", None),
            # Qwen3 text-only: model → layers
            lambda m: getattr(m, "model", None),
            # Llama/Mistral: transformer → h
            lambda m: getattr(m, "transformer", None),
            # Direct language_model attribute
            lambda m: getattr(m, "language_model", None),
        ]

        for get_wrapper in candidates:
            wrapper = get_wrapper(model)
            if wrapper is None:
                continue
            # wrapper may already be the layer list, or may have .layers / .h
            if hasattr(wrapper, "layers"):
                return wrapper.layers
            if hasattr(wrapper, "h"):
                return wrapper.h
            # wrapper might itself be a layer list
            if isinstance(wrapper, (list, nn.ModuleList)):
                return wrapper

        # Last resort: print model attributes for debugging
        attrs = sorted([a for a in dir(model) if not a.startswith("_") and not callable(getattr(model, a))])
        raise ValueError(
            f"Cannot find decoder layers in {type(model).__name__}. "
            f"Top-level attributes: {attrs[:20]}"
        )

    def _get_lm_head(self):
        """Get the LM head, handling multimodal model wrappers."""
        if hasattr(self.llm, "lm_head"):
            return self.llm.lm_head
        if hasattr(self.llm, "language_model") and hasattr(self.llm.language_model, "lm_head"):
            return self.llm.language_model.lm_head
        raise ValueError("Cannot find lm_head in this model")

    def _compute_position_embeddings(self, h: torch.Tensor,
                                     attention_mask: torch.Tensor):
        """
        Compute rotary position embeddings for Qwen3-VL layers.

        Qwen3-VL's decoder layer requires `position_embeddings` as a kwarg.
        These are computed by the model's `rotary_emb` module from position IDs.
        """
        # Find the rotary embedding module
        rotary_emb = self._find_rotary_emb()
        if rotary_emb is None:
            return None  # model doesn't use RoPE

        # Build position IDs from attention mask
        if attention_mask.dim() == 2:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = torch.arange(
                h.shape[1], device=h.device
            ).unsqueeze(0).expand(h.shape[0], -1)

        # Compute rotary embeddings for all positions
        cos, sin = rotary_emb(h, position_ids)
        return (cos.to(dtype=h.dtype), sin.to(dtype=h.dtype))

    def _find_rotary_emb(self):
        """Find the rotary embedding module in the model."""
        model = self.llm
        # Qwen3-VL: model.rotary_emb
        for path in [
            lambda m: getattr(m, "rotary_emb", None),
            lambda m: getattr(getattr(m, "model", None), "rotary_emb", None),
            lambda m: getattr(getattr(getattr(m, "model", None), "language_model", None), "rotary_emb", None),
        ]:
            result = path(model)
            if result is not None:
                print(f"[model] Found rotary_emb via {path.__name__}")
                return result
        return None

    def _get_input_embeddings(self):
        """Get token embeddings, handling multimodal model wrappers."""
        try:
            return self.llm.get_input_embeddings()
        except (AttributeError, NotImplementedError):
            pass
        if hasattr(self.llm, "language_model"):
            return self.llm.language_model.get_input_embeddings()
        raise ValueError("Cannot find input embeddings in this model")

    def _has_visual_encoder(self) -> bool:
        """Check if the model has a visual encoder (for VLMs)."""
        return hasattr(self.llm, "visual") or (
            hasattr(self.llm, "model") and hasattr(self.llm.model, "visual")
        )

    @staticmethod
    def _build_causal_attention_mask(
        attention_mask: torch.Tensor, dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert a 2D padding mask [B, N] into a 4D causal attention mask
        [B, 1, N, N] suitable for SDPA.

        - Positions where attention_mask == 0 (padding) are masked to -inf.
        - Upper triangle is masked (causal).
        """
        B, N = attention_mask.shape
        # Start with causal mask: lower triangle = 0, upper = -inf
        causal = torch.triu(
            torch.full((N, N), float('-inf'), dtype=dtype, device=attention_mask.device),
            diagonal=1,
        )  # [N, N]
        causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, N, N)  # [B, 1, N, N]

        # Apply padding mask: set columns for padding positions to -inf
        if (attention_mask == 0).any():
            pad_mask = attention_mask[:, None, None, :] == 0  # [B, 1, 1, N]
            causal = causal.masked_fill(pad_mask, float('-inf'))

        return causal

    def _inject_visual_features(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> tuple:
        """
        Process image through the visual encoder and inject features into
        the hidden states at image placeholder token positions.

        For Qwen3-VL, the model has a `visual` module that processes pixel
        values into image features. The visual encoder returns:
          - pooler_output: merged/projected features (dim = LLM hidden_dim)
          - deepstack_features: list of per-layer features for DeepStack injection

        Returns:
            (h, deepstack_features, visual_pos_mask) — updated hidden states,
            deepstack features list, and boolean mask of image token positions.
        """
        model = self.llm

        # Use the model's get_image_features if available (handles merger + split)
        if hasattr(model, "get_image_features"):
            try:
                vision_output = model.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True,
                )
                # pooler_output is a tuple of per-image tensors after split
                image_embeds_list = vision_output.pooler_output
                image_embeds = torch.cat(image_embeds_list, dim=0).to(
                    h.device, h.dtype,
                )  # [total_image_tokens, d]
                deepstack_features = vision_output.deepstack_features or []
            except Exception as e:
                print(f"[model] get_image_features failed: {e}")
                return h, [], None
        else:
            # Fallback: call visual encoder directly
            visual = getattr(model, "visual", None)
            if visual is None:
                visual = getattr(getattr(model, "model", None), "visual", None)
            if visual is None:
                return h, [], None

            try:
                vision_output = visual(
                    pixel_values.to(h.dtype),
                    grid_thw=image_grid_thw,
                    return_dict=True,
                )
                # Use pooler_output (merged/projected to LLM dim), not last_hidden_state
                if hasattr(vision_output, "pooler_output"):
                    image_embeds = vision_output.pooler_output
                elif hasattr(vision_output, "last_hidden_state"):
                    image_embeds = vision_output.last_hidden_state
                else:
                    return h, [], None
                image_embeds = image_embeds.to(h.device, h.dtype)
                deepstack_features = getattr(vision_output, "deepstack_features", []) or []
            except Exception as e:
                print(f"[model] Visual encoder failed: {e}")
                return h, [], None

        # Find image placeholder token ID
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.tokenizer.convert_tokens_to_ids("◣")
            if image_token_id is None or image_token_id == self.tokenizer.unk_token_id:
                return h, [], None

        # Build visual position mask [B, N] and scatter features
        visual_pos_mask = (input_ids == image_token_id)  # [B, N]
        total_img_tokens = visual_pos_mask.sum().item()

        if total_img_tokens == 0:
            return h, [], None

        # Use masked_scatter like Qwen3-VL does
        if total_img_tokens == image_embeds.shape[0]:
            mask_expanded = visual_pos_mask.unsqueeze(-1).expand_as(h)
            h = h.masked_scatter(mask_expanded, image_embeds)
        else:
            # Fallback: per-batch scatter with offset
            offset = 0
            for b in range(h.shape[0]):
                positions = visual_pos_mask[b].nonzero(as_tuple=True)[0]
                n = len(positions)
                if n > 0 and offset + n <= image_embeds.shape[0]:
                    h[b, positions] = image_embeds[offset:offset + n]
                    offset += n

        return h, deepstack_features, visual_pos_mask

    # ── Memory initialization ──

    def init_memory(self, batch_size: int, device: torch.device):
        """Initialize memory states and confidence for t=0."""
        # Cast to model dtype to avoid bfloat16/float32 mismatches
        llm_dtype = next(self.llm.parameters()).dtype
        m_states, c_states = [], []
        for sm in self.side_memories:
            m = sm.init_memory.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device=device, dtype=llm_dtype)
            c = torch.full((batch_size, self.num_mem), 0.5, device=device, dtype=llm_dtype)
            m_states.append(m)
            c_states.append(c)
        return m_states, c_states

    # ── Forward ──

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_embed_input: torch.Tensor,
        memory_states: List[torch.Tensor],
        confidence_states: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        One time step through the full memory-augmented LLM.

        Args:
            input_ids:          [B, N] — token IDs (includes image placeholder tokens)
            attention_mask:     [B, N]
            action_embed_input: [B, d] — raw action embedding (control variable)
            memory_states:      List of [B, N_m, d_mem]
            confidence_states:  List of [B, N_m]
            labels:             [B, N] — token labels for loss (optional)
            pixel_values:       [B, C, H, W] — processed image tensor (optional)
            image_grid_thw:     [B, 3] — image grid dimensions (optional)

        Returns:
            dict with: logits, new_memory, new_confidence, loss (if labels)
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # ── Embed tokens (pretrained) ──
        # For Qwen3-VL, the model's forward() handles replacing image placeholder
        # tokens with vision features. We call the full model's forward() for the
        # embedding step if pixel_values are provided.
        embed_layer = self._get_input_embeddings()
        h = embed_layer(input_ids)  # [B, N, d]

        # ── Process image features if provided ──
        # Qwen3-VL replaces image placeholder tokens with vision encoder output.
        # We use the model's get_image_features() to get projected features (via
        # the merger), and also collect deepstack_features for per-layer injection.
        deepstack_features = []
        visual_pos_mask = None
        if pixel_values is not None and self._has_visual_encoder():
            h, deepstack_features, visual_pos_mask = self._inject_visual_features(
                h, input_ids, pixel_values, image_grid_thw,
            )

        # ── Raw action embedding ──
        action_embed = self.action_embed(action_embed_input)  # [B, d]

        # ── Get decoder layers ──
        decoder_layers = self._get_decoder_layers()

        # ── Compute position embeddings if the model uses RoPE ──
        # Qwen3-VL layers require a `position_embeddings` kwarg.
        position_embeddings = self._compute_position_embeddings(
            h, attention_mask,
        )

        # ── Convert attention_mask to 4D causal mask ──
        # Qwen3-VL's SDPA attention expects a [B, 1, N, N] causal mask,
        # not a 2D [B, N] padding mask.
        attention_mask_4d = self._build_causal_attention_mask(
            attention_mask, h.dtype,
        )

        # ── Process through each layer ──
        new_memory, new_confidence = [], []
        predicted_mems, obs_feats = [], []  # for uncertainty calibration loss
        for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
            # 4a. Memory Predict (FiLM-GRU)
            m_hat, c_hat = sm.predict(
                memory_states[l], confidence_states[l], action_embed,
            )

            # 4b. Run pretrained layer (self-attn + FFN)
            layer_output = layer(
                h,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if isinstance(layer_output, tuple):
                h_layer = layer_output[0]
            else:
                h_layer = layer_output

            # 4b.5 DeepStack injection (Qwen3-VL adds visual features to
            #      early decoder layers' hidden states at image positions)
            if deepstack_features and l < len(deepstack_features):
                ds_feat = deepstack_features[l].to(h_layer.device, h_layer.dtype)
                if visual_pos_mask is not None:
                    mask_1d = visual_pos_mask  # [B, N]
                    # Add visual features at image token positions
                    for b in range(h_layer.shape[0]):
                        positions = mask_1d[b].nonzero(as_tuple=True)[0]
                        n = len(positions)
                        if n > 0 and n <= ds_feat.shape[0]:
                            h_layer[b, positions] = h_layer[b, positions] + ds_feat[:n]

            # 4c. Extract observation
            z_t = sm.extract_observation(h_layer, self.num_act_tokens)

            # 4d. Memory Correct (Kalman Filter)
            m_new, c_new = sm.correct(m_hat, c_hat, z_t)

            # 4e. Memory Inject
            h = sm.inject(h_layer, m_new)

            new_memory.append(m_new)
            new_confidence.append(c_new)
            # Save for uncertainty calibration loss
            predicted_mems.append(m_hat.detach())
            obs_feats.append(z_t.detach())

        # ── LM head (pretrained) ──
        logits = self._get_lm_head()(h)  # [B, N, vocab_size]

        result = {
            "logits": logits,
            "new_memory": new_memory,
            "new_confidence": new_confidence,
        }

        # ── Compute loss if labels provided ──
        if labels is not None:
            from .loss import compute_compact_loss
            loss, loss_dict = compute_compact_loss(
                logits=logits,
                labels=labels,
                memory_states=new_memory,
                confidence_states=new_confidence,
                config=self.config,
                predicted_memory=predicted_mems,
                observation_feat=obs_feats,
            )
            result["loss"] = loss
            result["loss_dict"] = loss_dict

        return result

    # ── Generation (for eval) ──

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_embed_input: torch.Tensor,
        memory_states: List[torch.Tensor],
        confidence_states: List[torch.Tensor],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Generate action tokens autoregressively with memory.

        Images are processed once for the prompt. During generation, each new
        token goes through all decoder layers (self-attn + FFN) and memory
        is updated. Position embeddings use correct absolute positions.
        """
        B = input_ids.shape[0]
        device = input_ids.device
        seq_len = input_ids.shape[1]

        action_embed = self.action_embed(action_embed_input)
        decoder_layers = self._get_decoder_layers()
        embed_layer = self._get_input_embeddings()

        # ── Process prompt embeddings ──
        h = embed_layer(input_ids)  # [B, N, d]

        # Inject visual features if available
        deepstack_features = []
        visual_pos_mask = None
        if pixel_values is not None and self._has_visual_encoder():
            h, deepstack_features, visual_pos_mask = self._inject_visual_features(
                h, input_ids, pixel_values, image_grid_thw,
            )

        # ── Build 4D causal mask for the prompt ──
        attn_mask_4d = self._build_causal_attention_mask(attention_mask, h.dtype)

        # ── Compute position embeddings for prompt ──
        position_embeddings = self._compute_position_embeddings(h, attention_mask)

        # ── Process prompt through all layers with memory ──
        new_memory, new_confidence = [], []
        for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
            m_hat, c_hat = sm.predict(memory_states[l], confidence_states[l], action_embed)
            layer_output = layer(h, attention_mask=attn_mask_4d,
                                position_embeddings=position_embeddings, **kwargs)
            h_layer = layer_output[0] if isinstance(layer_output, tuple) else layer_output

            # DeepStack injection
            if deepstack_features and l < len(deepstack_features):
                ds_feat = deepstack_features[l].to(h_layer.device, h_layer.dtype)
                if visual_pos_mask is not None:
                    for b in range(h_layer.shape[0]):
                        positions = visual_pos_mask[b].nonzero(as_tuple=True)[0]
                        n = len(positions)
                        if n > 0 and n <= ds_feat.shape[0]:
                            h_layer[b, positions] = h_layer[b, positions] + ds_feat[:n]

            z_t = sm.extract_observation(h_layer, self.num_act_tokens)
            m_new, c_new = sm.correct(m_hat, c_hat, z_t)
            h = sm.inject(h_layer, m_new)
            new_memory.append(m_new)
            new_confidence.append(c_new)

        # ── Generate tokens one by one ──
        generated_ids = []
        cur_pos = seq_len  # absolute position for the next token

        # Use the last position's hidden state (after memory injection) for logits
        cur_h = h[:, -1:, :]  # [B, 1, d]

        for _ in range(max_new_tokens):
            logits = self._get_lm_head()(cur_h)  # [B, 1, vocab]
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_logits.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated_ids.append(next_token)

            # ── Feed next token through all layers with memory ──
            next_h = embed_layer(next_token)  # [B, 1, d]

            # Single-token causal mask (no padding possible for 1 token)
            single_mask_4d = torch.zeros(
                (B, 1, 1, 1), dtype=h.dtype, device=device,
            )  # [B, 1, 1, 1] — attend to self

            # Position embedding for this token at absolute position cur_pos
            single_pos_ids = torch.tensor(
                [[cur_pos]], dtype=torch.long, device=device,
            )  # [B, 1]
            rotary_emb = self._find_rotary_emb()
            if rotary_emb is not None:
                cos_single, sin_single = rotary_emb(next_h, single_pos_ids)
                pos_emb_single = (cos_single.to(dtype=next_h.dtype),
                                  sin_single.to(dtype=next_h.dtype))
            else:
                pos_emb_single = None

            for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
                layer_output = layer(next_h, attention_mask=single_mask_4d,
                                    position_embeddings=pos_emb_single,
                                    **kwargs)
                next_h_layer = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                # Update memory with the new token's hidden state
                z_t = sm.extract_observation(next_h_layer, self.num_act_tokens)
                m_hat, c_hat = sm.predict(new_memory[l], new_confidence[l], action_embed)
                m_new, c_new = sm.correct(m_hat, c_hat, z_t)
                next_h = sm.inject(next_h_layer, m_new)
                new_memory[l] = m_new
                new_confidence[l] = c_new

            cur_h = next_h  # [B, 1, d] — use injected output for next logits
            cur_pos += 1

            # Stop at EOS
            if self.tokenizer.eos_token_id is not None and \
               (next_token == self.tokenizer.eos_token_id).all():
                break

        generated_ids = torch.cat(generated_ids, dim=1) if generated_ids else \
            torch.empty(B, 0, dtype=torch.long, device=device)

        return {
            "generated_ids": generated_ids,
            "new_memory": new_memory,
            "new_confidence": new_confidence,
        }

    # ── Save / Load ──

    def save_pretrained(self, save_directory: str | Path):
        """Save the full model: base LLM + side memory modules + config."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save side memory modules
        side_mem_dir = save_path / "side_memory"
        side_mem_dir.mkdir(exist_ok=True)
        torch.save(
            self.side_memories.state_dict(),
            side_mem_dir / "side_memories.pt",
        )
        torch.save(
            self.action_embed.state_dict(),
            side_mem_dir / "action_embed.pt",
        )

        # Save config
        config_dict = self.config.to_dict()
        config_dict["hidden_dim"] = self.hidden_dim
        config_dict["num_layers"] = self.num_layers
        (save_path / "compact_config.json").write_text(
            json.dumps(config_dict, indent=2, ensure_ascii=False)
        )

        # Save base LLM (or just reference if frozen)
        if self.config.freeze_base:
            # Only save a reference to the base model
            (save_path / "base_model_ref.txt").write_text(self.config.base_model_name)
        else:
            self.llm.save_pretrained(save_path / "base_model")

        # Save tokenizer and processor
        self.tokenizer.save_pretrained(save_path)
        if self.processor is not None:
            try:
                self.processor.save_pretrained(save_path)
            except Exception:
                pass

        print(f"[save] JAMEL-COMPACT model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_directory: str | Path,
                        config_override: Optional[CompactConfig] = None) -> "JAMELCompactWrapper":
        """Load a saved JAMEL-COMPACT model."""
        load_path = Path(load_directory)

        # Load config
        config_path = load_path / "compact_config.json"
        if config_path.exists():
            config_dict = json.loads(config_path.read_text())
            config = CompactConfig.from_args(**config_dict)
        else:
            config = config_override or CompactConfig()

        if config_override is not None:
            # Override specific fields
            for k, v in config_override.to_dict().items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Check for base model reference (frozen) or saved base model
        base_ref_path = load_path / "base_model_ref.txt"
        base_model_dir = load_path / "base_model"
        if base_ref_path.exists():
            config.base_model_name = base_ref_path.read_text().strip()
        elif base_model_dir.exists():
            config.base_model_name = str(base_model_dir)

        # Create model
        model = cls(config)

        # Load side memory weights
        side_mem_dir = load_path / "side_memory"
        if side_mem_dir.exists():
            sm_state = torch.load(side_mem_dir / "side_memories.pt", map_location="cpu")
            model.side_memories.load_state_dict(sm_state)
            ae_state = torch.load(side_mem_dir / "action_embed.pt", map_location="cpu")
            model.action_embed.load_state_dict(ae_state)
            print(f"[load] Side memory modules loaded from {side_mem_dir}")

        return model

    # ── Parameter counting ──

    def count_parameters(self) -> dict:
        """Return parameter counts: base, new, total."""
        base_params = sum(p.numel() for p in self.llm.parameters())
        new_params = sum(p.numel() for p in self.side_memories.parameters())
        new_params += sum(p.numel() for p in self.action_embed.parameters())
        return {
            "base": base_params,
            "new": new_params,
            "total": base_params + new_params,
            "overhead_pct": new_params / base_params * 100,
        }