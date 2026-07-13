"""
JAMEL-COMPACT vs Original JAMEL — Side-by-side Python Showcase
================================================================

This file demonstrates the architectural differences between:

  A) **Original JAMEL** — separate frozen Qwen3-VL compressor + actor LLM,
     memory tokens as prefix embeddings, no recurrent state.

  B) **JAMEL-COMPACT** — unified LLM with per-layer side memory,
     FiLM-GRU + Kalman Filter update, cross-attention read/write,
     confidence tracking, hierarchical design.

Dependencies: torch only (pip install torch)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# Shared constants & utilities
# ═══════════════════════════════════════════════════════════════════════════════

HIDDEN_DIM   = 768        # d  — LLM hidden dimension
NUM_LAYERS   = 12         # L  — number of transformer layers
NUM_ACT      = 1          # action tokens (always 1)
NUM_IMG      = 256        # image tokens (after visual encoding)
NUM_INST     = 32         # instruction tokens
SEQ_LEN      = NUM_ACT + NUM_IMG + NUM_INST  # N = 289
NUM_MEM      = 16         # N_m — memory tokens per layer
BATCH        = 2


def _layerwise_param(layer_idx: int, num_layers: int, *, shallow: float,
                     mid: float, deep: float) -> float:
    """Assign hierarchical hyperparameters by layer depth."""
    if layer_idx < num_layers // 3:
        return shallow
    elif layer_idx < 2 * num_layers // 3:
        return mid
    return deep


# ═══════════════════════════════════════════════════════════════════════════════
# A)  ORIGINAL JAMEL  —  separate compressor + prefix injection
# ═══════════════════════════════════════════════════════════════════════════════

class OriginalJAMELScreenCompressor(nn.Module):
    """
    Simplified replica of ScreenCompressor from jamel/arch/qwen3vl_compressor/.

    Takes a batch of (screenshot_image_features, action_text_embedding) pairs
    and outputs one compressed memory token per pair — the EOS hidden state
    from the final transformer layer.  In the real code this wraps a full
    Qwen3-VL; here we simulate it with a small transformer encoder.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.compressor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=4,
        )
        # EOS token embedding (learned, analogous to Qwen3-VL's EOS token)
        self.eos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, images: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images:  [B, seq_per_img, d]  — image features (already encoded)
            actions: [B, seq_per_act, d]  — action text embeddings
        Returns:
            memory: [B, 1, d] — one compressed memory token per sample
        """
        B = images.shape[0]
        # Concatenate: [image tokens, action tokens, EOS]
        eos = self.eos_embed.expand(B, -1, -1)
        seq = torch.cat([images, actions, eos], dim=1)  # [B, L_img+L_act+1, d]
        hidden = self.compressor(seq)
        # Extract EOS position → compressed memory token
        return hidden[:, -1:, :]  # [B, 1, d]


class OriginalJAMELLLM(nn.Module):
    """
    Simplified replica of MemoryAugmentedCausalLM from jamel/train/memory/modeling.py.

    A standard transformer LLM that accepts memory tokens as a PREFIX to its
    input embeddings.  Memory is *outside* the layer stack — a single batch of
    pre-computed memory tokens is injected once before layer 0.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [B, prefix_len + N, d] — memory prefix + text tokens
        Returns:
            hidden: [B, prefix_len + N, d]
        """
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=None)
        return h


class OriginalJAMEL(nn.Module):
    """
    Full original JAMEL pipeline:

        Screenshot + Action → Compressor → memory tokens [B, N_mem, d]
        Query text → Token Embedding → concat with memory → LLM → action logits

    Key characteristics:
      • Compressor is a SEPARATE frozen model (Qwen3-VL)
      • Compressor runs BEFORE the LLM, producing one token per history step
      • All memory tokens are concatenated into a flat prefix
      • No recurrent state — memory is recomputed from scratch each step
      • No confidence tracking
      • No per-layer injection
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 vocab_size: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Separate compressor (frozen during SFT)
        self.compressor = OriginalJAMELScreenCompressor(hidden_dim)

        # Dimension alignment: compressor hidden → LLM hidden
        # (Needed when compressor and LLM have different hidden sizes,
        #  e.g. Qwen3-VL-2B 2048 → Qwen3-8B 4096)
        self.aligner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Main LLM
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.llm = OriginalJAMELLLM(hidden_dim, num_layers)
        self.action_head = nn.Linear(hidden_dim, vocab_size)

    def compress_history(self, history_screenshots: List[torch.Tensor],
                         history_actions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compress all history (screenshot_i, action_i) pairs into memory tokens.

        In real JAMEL this is done by OnlineHistoryMemoryBuilder which caches
        results per step.  All history pairs are batched through the compressor.

        Args:
            history_screenshots: list of [B, N_img, d] — one per history step
            history_actions:      list of [B, N_act, d] — one per history step
        Returns:
            memory_tokens: [B, len(history), d] — all compressed history tokens
        """
        if not history_screenshots:
            B = history_screenshots  # will be 0; caller must handle
            return torch.empty(0, 0, self.hidden_dim)

        all_mems = []
        for img, act in zip(history_screenshots, history_actions):
            mem = self.compressor(img, act)          # [1, 1, d]
            mem = self.aligner(mem)                   # [1, 1, d]
            all_mems.append(mem)
        return torch.cat(all_mems, dim=1)             # [1, T, d]

    def forward(self, query_token_ids: torch.Tensor,
                memory_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_token_ids: [B, N] — current step text tokens
            memory_tokens:   [B, T, d] — compressed history (T steps)
        Returns:
            logits: [B, N_out, vocab_size] — action logits (use last position)
        """
        B, T, _ = memory_tokens.shape
        # Embed query text
        query_embeds = self.token_embed(query_token_ids)  # [B, N, d]
        # Concatenate: [memory prefix | text query]
        combined = torch.cat([memory_tokens, query_embeds], dim=1)  # [B, T+N, d]
        hidden = self.llm(combined)                                  # [B, T+N, d]
        # Action prediction from last position
        logits = self.action_head(hidden)  # [B, T+N, vocab_size]
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# B)  JAMEL-COMPACT  —  unified LLM + per-layer side memory + RNN + KF
# ═══════════════════════════════════════════════════════════════════════════════

class FiLMGRUCell(nn.Module):
    """
    FiLM-modulated GRU cell.

    The action embedding a_emb modulates the GRU state transition via
    Feature-wise Linear Modulation (FiLM):

        γ, β = MLP(a_emb)
        h_new = GRU(action_proj(a_emb), γ ⊙ h_old + β)

    This makes the action an explicit *control variable* that steers
    how memory evolves — analogous to the control input u_t in
    state-space models:  s_t = f(s_{t-1}, u_{t-1}).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        # FiLM: action → (γ, β) modulation pair
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
        )
        # Project action embedding to GRU input dimension
        self.action_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_prev: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: [B*N_m, d] — previous memory state (flattened batch)
            a_emb:  [B*N_m, d] — action embedding (expanded to all tokens)
        Returns:
            h_new:  [B*N_m, d] — predicted memory state
        """
        # 1. FiLM: action modulates the old memory to produce the GRU hidden
        gamma_beta = self.film_mlp(a_emb)            # [B*N_m, 2*d]
        gamma, beta = gamma_beta.chunk(2, dim=-1)    # [B*N_m, d], [B*N_m, d]
        h_modulated = gamma * h_prev + beta          # [B*N_m, d]
        # 2. GRU: action-encoded input, FiLM-modulated memory as hidden
        gru_input = self.action_proj(a_emb)           # [B*N_m, d]
        return self.gru(gru_input, h_modulated)       # [B*N_m, d]


class MemoryAugmentedLayer(nn.Module):
    """
    A single transformer layer augmented with side memory (JAMEL-COMPACT).

    This is the core building block.  Each layer owns its own memory bank
    and executes the full Predict→Observe→Correct→Inject cycle per time step.

    Architecture:

       ┌────────────────────────────────────────────┐
       │  H_in ──→ SelfAttn ──→ H_self ──────────────┤──→ Inject ──→ H_out
       │                │                          │
       │                └──→ Extract Z_t ──────────┤──→ Gate (KF)
       │                                           │
       │  M_prev ──→ FiLM-GRU ──→ M_hat ──────────┤──→ CrossAttn → ΔM
       │       ↑                            │       │                    │
       │  C_prev ──→ Decay ──→ C_hat ──────┤──→ K =(1-C)*gate ────────┤
       │                                   │                            │
       │                            M_new = M_hat + K ⊙ ΔM ────────────┤
       │                            C_new = C_hat + α(1-C_hat)·match ──┤
       └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, layer_idx: int, num_layers: int,
                 hidden_dim: int = HIDDEN_DIM, num_mem: int = NUM_MEM,
                 num_heads: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_mem = num_mem

        # ── Standard transformer components ──
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # ── Memory Predict: FiLM-GRU ──
        self.gru = FiLMGRUCell(hidden_dim)

        # ── Memory Update: Cross-Attention (Innovation) ──
        # Q = M_hat (each memory token asks "how should I be corrected?")
        # K, V = Z_t (observation provides the correction signal)
        # Output: [B, N_m, d] — per-token innovation
        self.mem_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        # Project CrossAttn output back to memory dimension
        self.innovation_proj = nn.Linear(hidden_dim, hidden_dim)

        # ── Kalman Gain gate (per-token) ──
        # Input: [Z_t broadcast; M_hat] per token → [B, N_m, 2d]
        # Output: [B, N_m, d] — per-token gain base
        self.k_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Memory Injection: Cross-Attention (read from memory) ──
        self.inject_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )

        # ── Hierarchical hyperparameters ──
        # Confidence decay: shallow layers forget fast (focus on UI detail);
        # deep layers forget slowly (preserve task logic).
        self.lambda_decay = _layerwise_param(
            layer_idx, num_layers, shallow=0.70, mid=0.85, deep=0.95,
        )
        # Injection weight: shallow layers inject heavily (visual detail);
        # deep layers inject lightly (preserve reasoning).
        self.inject_weight = _layerwise_param(
            layer_idx, num_layers, shallow=0.8, mid=0.5, deep=0.3,
        )
        # Learning rate α for confidence update
        self.alpha = 0.1

        # ── Learnable initial memory ──
        self.init_memory = nn.Parameter(torch.randn(num_mem, hidden_dim) * 0.02)

    def _extract_observation(self, h_self: torch.Tensor) -> torch.Tensor:
        """
        Extract observation feature from self-attention output.

        We take the image and instruction token positions (skip action token)
        and mean-pool them into a single observation vector.

        Args:
            h_self: [B, N, d]
        Returns:
            Z_t:    [B, d]
        """
        # Slice: skip action token (index 0), take img + inst
        obs_tokens = h_self[:, NUM_ACT:, :]                # [B, N_img+N_inst, d]
        return obs_tokens.mean(dim=1)                       # [B, d]

    def forward(self, h_in: torch.Tensor,
                m_prev: torch.Tensor, c_prev: torch.Tensor,
                action_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor]:
        """
        One time step of a JAMEL-COMPACT layer.

        Args:
            h_in:         [B, N, d]     — input hidden states from prev layer
            m_prev:       [B, N_m, d]   — previous memory state
            c_prev:       [B, N_m]      — previous confidence vector
            action_embed: [B, d]        — raw action embedding (x_act^{t-1})

        Returns:
            h_out: [B, N, d]     — output hidden states (to next layer)
            m_new: [B, N_m, d]   — updated memory state (to next time step)
            c_new: [B, N_m]      — updated confidence   (to next time step)
        """
        B = h_in.shape[0]

        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Self-Attention (standard transformer forward)
        # ═══════════════════════════════════════════════════════════════════
        h_normed = self.self_attn_norm(h_in)
        h_attn, _ = self.self_attn(h_normed, h_normed, h_normed)
        h_self = h_in + h_attn  # residual                           [B, N, d]

        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Memory Predict — FiLM-GRU + confidence decay (batched)
        # ═══════════════════════════════════════════════════════════════════
        # Batched GRU: reshape [B, N_m, d] → [B*N_m, d], run GRU, reshape back
        m_prev_flat = m_prev.reshape(B * self.num_mem, -1)         # [B*N_m, d]
        # Expand action_embed [B, d] → [B*N_m, d] (same action for all tokens)
        a_emb_flat = action_embed.unsqueeze(1).expand(
            -1, self.num_mem, -1,
        ).reshape(B * self.num_mem, -1)                            # [B*N_m, d]
        m_hat_flat = self.gru(m_prev_flat, a_emb_flat)             # [B*N_m, d]
        m_hat = m_hat_flat.view(B, self.num_mem, -1)               # [B, N_m, d]
        # Confidence decay (per-token, preserves granularity)
        c_hat = self.lambda_decay * c_prev                         # [B, N_m]

        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Observation Extraction
        # ═══════════════════════════════════════════════════════════════════
        z_t = self._extract_observation(h_self)                    # [B, d]
        # Expand for cross-attention: Z_t as K,V source
        z_t_expanded = z_t.unsqueeze(1)                            # [B, 1, d]

        # ═══════════════════════════════════════════════════════════════════
        # Step 4: Memory Update — Kalman Filter (per-token)
        # ═══════════════════════════════════════════════════════════════════
        # 4a. Innovation (per-token): each memory token queries the observation
        #     ΔM = Proj(CrossAttn(Q=M_hat, K=Z_t, V=Z_t))  → [B, N_m, d]
        delta_m_raw, _ = self.mem_cross_attn(
            m_hat, z_t_expanded, z_t_expanded,
        )                                                          # [B, N_m, d]
        delta_m = self.innovation_proj(delta_m_raw)                # [B, N_m, d]

        # 4b. Kalman Gain (per-token): K = σ(W[Z_t; M_hat]) ⊙ (1 - C_hat)
        #     Z_t is broadcast to all N_m tokens, concatenated with M_hat
        z_broadcast = z_t.unsqueeze(1).expand(-1, self.num_mem, -1)  # [B, N_m, d]
        concat = torch.cat([z_broadcast, m_hat], dim=-1)           # [B, N_m, 2d]
        k_base = self.k_gate(concat)                               # [B, N_m, d]
        # (1 - C_hat) per-token: low confidence → high gain → trust observation
        k_uncertainty = (1.0 - c_hat).unsqueeze(-1)                # [B, N_m, 1]
        k_gain = k_base * k_uncertainty                            # [B, N_m, d]

        # 4c. Update memory (per-token): M_new = M_hat + K ⊙ ΔM
        m_new = m_hat + k_gain * delta_m                           # [B, N_m, d]

        # 4d. Update confidence (per-token):
        #     match_i = cosine_sim(Z_t, M_hat_i)  for each token i
        #     C_new = C_hat + α · (1-C_hat) · match
        z_norm = F.normalize(z_t, dim=-1)                          # [B, d]
        m_hat_norm = F.normalize(m_hat, dim=-1)                    # [B, N_m, d]
        match = (z_norm.unsqueeze(1) * m_hat_norm).sum(dim=-1)     # [B, N_m]
        match = match.clamp(0.0, 1.0)                              # [B, N_m]
        c_new = c_hat + self.alpha * (1.0 - c_hat) * match         # [B, N_m]

        # ═══════════════════════════════════════════════════════════════════
        # Step 5: Memory Injection — read from updated memory into main stream
        # ═══════════════════════════════════════════════════════════════════
        # ΔH = CrossAttn(Q=H_self, K=M_new, V=M_new)
        delta_h, _ = self.inject_cross_attn(h_self, m_new, m_new)  # [B, N, d]
        h_tilde = h_self + self.inject_weight * delta_h             # [B, N, d]

        # ═══════════════════════════════════════════════════════════════════
        # Step 6: FFN (standard transformer)
        # ═══════════════════════════════════════════════════════════════════
        h_ffn_normed = self.ffn_norm(h_tilde)
        h_out = h_tilde + self.ffn(h_ffn_normed)                   # [B, N, d]

        return h_out, m_new, c_new


class JAMELCompact(nn.Module):
    """
    Full JAMEL-COMPACT model: unified LLM with per-layer side memory.

    Key characteristics vs Original JAMEL:

      • Compressor and Actor are the SAME LLM — no separate vision encoder
      • Each transformer layer owns its own memory bank (side memory)
      • Memory is updated via FiLM-GRU (predict) + KF cross-attention (correct)
      • Confidence vector C tracks uncertainty per memory token
      • Hierarchical: shallow layers forget fast, deep layers retain logic
      • Action embeddings control memory state transition via FiLM
      • Cross-attention reads memory back into the main stream at every layer
      • End-to-end trainable with 3-term loss (action + mem_reg + uncert_calib)

    Algorithm (per time step t, per layer l):

      1. Predict:  M̂ = FiLM-GRU(M_{t-1}, x_act^{t-1})
                   Ĉ = λ_l · C_{t-1}
      2. Observe:  H = SelfAttn(X_t)
                   Z = Extract(H[img, instruct])
      3. Correct:  ΔM = CrossAttn(Q=Z, K=M̂, V=M̂)
                   K  = σ(W[Z; pool(M̂)]) ⊙ (1-Ĉ)
                   M  = M̂ + K ⊙ ΔM
                   C  = Ĉ + α·(1-Ĉ)·cosine_sim(Z, M̂)
      4. Inject:   H̃ = H + W_inj · CrossAttn(Q=H, K=M, V=M)
                   H_out = FFN(H̃)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_LAYERS,
                 num_mem: int = NUM_MEM, vocab_size: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mem = num_mem

        # Token embeddings (shared for text and action; image comes from visual encoder)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        # Image projection: separate visual encoder → LLM hidden space
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        # Action embedding: discrete action tokens
        self.action_embed = nn.Embedding(100, hidden_dim)

        # Per-layer memory-augmented transformer layers
        self.layers = nn.ModuleList([
            MemoryAugmentedLayer(l, num_layers, hidden_dim, num_mem)
            for l in range(num_layers)
        ])

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, vocab_size)

    def _init_memory(self, batch_size: int, device: torch.device):
        """Initialize memory states and confidence for t=0."""
        m_init = self.layers[0].init_memory.unsqueeze(0).expand(
            batch_size, -1, -1,
        )                                                         # [B, N_m, d]
        c_init = torch.full(
            (batch_size, self.num_mem), 0.5, device=device,
        )                                                         # [B, N_m]
        return [m_init.clone() for _ in range(self.num_layers)], \
               [c_init.clone() for _ in range(self.num_layers)]

    def forward(self, text_token_ids: torch.Tensor,
                image_features: torch.Tensor,
                action_token_id: torch.Tensor,
                memory_states: List[torch.Tensor],
                confidence_states: List[torch.Tensor],
                ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        One time step of JAMEL-COMPACT.

        Args:
            text_token_ids:  [B, N_inst]           — instruction token IDs
            image_features:  [B, N_img, d]          — pre-encoded image features
            action_token_id: [B, 1]                 — previous action token ID
            memory_states:   List of [B, N_m, d]    — layer memories from t-1
            confidence_states: List of [B, N_m]     — layer confidences from t-1

        Returns:
            logits:           [B, vocab_size]       — action logits
            new_memory:       List of [B, N_m, d]   — updated memories for t+1
            new_confidence:   List of [B, N_m]      — updated confidences for t+1
        """
        B = text_token_ids.shape[0]

        # ── Raw action embedding (NOT from self-attention!) ──
        # This is x_act^{t-1}, the raw control input, not H[act].
        # Using raw embedding avoids information leakage from current
        # observation and keeps the Predict step a true "prior".
        action_embed = self.action_embed(action_token_id).squeeze(1)  # [B, d]

        # ── Build input sequence: [action | image | instruction] ──
        text_embed = self.token_embed(text_token_ids)                 # [B, N_inst, d]
        img_embed = self.image_proj(image_features)                  # [B, N_img, d]
        x_t = torch.cat([action_embed.unsqueeze(1), img_embed, text_embed],
                        dim=1)                                        # [B, N, d]

        # ── Feed through memory-augmented layers ──
        h = x_t
        new_memory, new_confidence = [], []
        for l, layer in enumerate(self.layers):
            h, m_new, c_new = layer(
                h, memory_states[l], confidence_states[l], action_embed,
            )
            new_memory.append(m_new)
            new_confidence.append(c_new)

        # ── Action prediction from final hidden state ──
        # Take the last (instruction) token position for action decoding
        logits = self.action_head(h[:, -1, :])                        # [B, vocab_size]
        return logits, new_memory, new_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# C)  LOSS FUNCTIONS  (JAMEL-COMPACT only)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jamel_compact_loss(
    logits: torch.Tensor,
    action_gt: torch.Tensor,
    memory_states: List[torch.Tensor],
    confidence_states: List[torch.Tensor],
    predicted_memory: List[torch.Tensor],   # M_hat per layer (before KF update)
    observation_feat: List[torch.Tensor],    # Z_t per layer
    lambda_mem: float = 0.001,
    lambda_uncert: float = 0.1,
    beta_entropy: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    JAMEL-COMPACT total loss with three components.

    Returns: (total_loss, loss_dict)
    """
    # ── 1. Action loss (Cross-Entropy) ──
    loss_action = F.cross_entropy(logits, action_gt)

    # ── 2. Memory regularization ──
    # L2 penalty on memory tokens + entropy penalty on confidence
    loss_mem_l2 = 0.0
    loss_mem_entropy = 0.0
    eps = 1e-8
    for M, C in zip(memory_states, confidence_states):
        # L2: prevent memory values from exploding
        loss_mem_l2 += M.pow(2).mean()
        # Bernoulli Entropy: prevent C from saturating at 0 or 1
        entropy = -(C * torch.log(C + eps)
                    + (1 - C) * torch.log(1 - C + eps))
        loss_mem_entropy += entropy.mean()
    L = len(memory_states)
    loss_mem = (loss_mem_l2 / L) + beta_entropy * (loss_mem_entropy / L)

    # ── 3. Uncertainty calibration ──
    # MSE between confidence C and actual observation-to-prediction match
    # Per-token: match_i = cosine_sim(Z, M_hat_i)
    loss_uncert = 0.0
    for C, M_hat, Z in zip(confidence_states, predicted_memory, observation_feat):
        # Z: [B, d], M_hat: [B, N_m, d] → match: [B, N_m]
        z_norm = F.normalize(Z, dim=-1)                        # [B, d]
        m_norm = F.normalize(M_hat, dim=-1)                    # [B, N_m, d]
        match = (z_norm.unsqueeze(1) * m_norm).sum(dim=-1).clamp(0, 1)  # [B, N_m]
        loss_uncert += F.mse_loss(C, match.detach())           # per-token MSE
    loss_uncert = loss_uncert / L

    # ── Total ──
    loss_total = loss_action + lambda_mem * loss_mem + lambda_uncert * loss_uncert

    loss_dict = {
        "total":   loss_total.item(),
        "action":  loss_action.item(),
        "mem_l2":  (loss_mem_l2 / L).item(),
        "mem_ent": (loss_mem_entropy / L).item(),
        "uncert":  loss_uncert.item(),
    }
    return loss_total, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════
# D)  DEMO:  side-by-side forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    device = torch.device("cpu")
    B = BATCH

    print("=" * 70)
    print("JAMEL-COMPACT vs Original JAMEL — Forward Pass Demo")
    print("=" * 70)

    # ── Synthetic inputs ──────────────────────────────────────────────────
    text_ids     = torch.randint(0, 500, (B, NUM_INST), device=device)
    img_feat     = torch.randn(B, NUM_IMG, HIDDEN_DIM, device=device)
    action_id    = torch.randint(0, 100, (B, 1), device=device)
    action_gt    = torch.randint(0, 500, (B,), device=device)

    # ── Original JAMEL ────────────────────────────────────────────────────
    print("\n── Original JAMEL ──")
    orig = OriginalJAMEL(HIDDEN_DIM, NUM_LAYERS).to(device)

    # Simulate 3 history steps: each is a (screenshot, action) pair
    # Batch size must match text_ids batch size (B=2)
    history_imgs = [torch.randn(B, NUM_IMG, HIDDEN_DIM) for _ in range(3)]
    history_acts = [torch.randn(B, NUM_ACT, HIDDEN_DIM) for _ in range(3)]

    # Compress history → memory prefix
    memory_prefix = orig.compress_history(history_imgs, history_acts)
    print(f"  History steps: {len(history_imgs)}")
    print(f"  Memory prefix shape: {list(memory_prefix.shape)}  # [B, T, d]")

    # Forward: memory tokens are prepended to text embeddings
    logits_orig = orig.forward(text_ids, memory_prefix)
    print(f"  Output logits:   {list(logits_orig.shape)}")
    print(f"  Architecture:    Compressor (separate, frozen) → prefix → LLM")
    print(f"  Memory location: BEFORE all layers (single injection)")
    print(f"  Memory update:   None (re-compute from scratch each step)")

    # ── JAMEL-COMPACT ─────────────────────────────────────────────────────
    print("\n── JAMEL-COMPACT ──")
    compact = JAMELCompact(HIDDEN_DIM, NUM_LAYERS, NUM_MEM).to(device)

    # Initialize memory and confidence
    mem_states, conf_states = compact._init_memory(B, device)
    print(f"  Layers:          {NUM_LAYERS}")
    print(f"  Memory tokens/layer: {NUM_MEM}")
    print(f"  Init memory:     [{B}, {NUM_MEM}, {HIDDEN_DIM}]")
    print(f"  Init confidence: [{B}, {NUM_MEM}]")

    # Forward: one time step through all layers
    logits_compact, new_mem, new_conf = compact.forward(
        text_ids, img_feat, action_id, mem_states, conf_states,
    )
    print(f"  Output logits:   {list(logits_compact.shape)}")
    print(f"  Memory updated:  {len(new_mem)} layers × [{B}, {NUM_MEM}, {HIDDEN_DIM}]")
    print(f"  Confidence upd:  {len(new_conf)} layers × [{B}, {NUM_MEM}]")

    # Show hierarchical params
    print(f"\n  Hierarchical params (sample layers):")
    for l_idx in [0, NUM_LAYERS // 3, NUM_LAYERS // 2, NUM_LAYERS - 1]:
        layer = compact.layers[l_idx]
        print(f"    Layer {l_idx:2d}:  λ_decay={layer.lambda_decay:.2f}  "
              f"W_inj={layer.inject_weight:.2f}  "
              f"({'shallow' if l_idx < NUM_LAYERS//3 else 'mid' if l_idx < 2*NUM_LAYERS//3 else 'deep'})")

    # ── Loss computation (JAMEL-COMPACT) ──────────────────────────────────
    # Collect M_hat and Z_t for uncertainty calibration
    # (In a real training loop these would come from the layer forward pass)
    pred_mem = [ms + 0.01 * torch.randn_like(ms) for ms in mem_states]  # placeholder
    obs_feat = [torch.randn(B, HIDDEN_DIM) for _ in range(NUM_LAYERS)]  # placeholder

    loss_total, loss_dict = compute_jamel_compact_loss(
        logits_compact, action_gt, new_mem, new_conf, pred_mem, obs_feat,
    )
    print(f"\n  Losses:")
    for k, v in loss_dict.items():
        print(f"    loss_{k:7s}: {v:.4f}")

    # ── Comparison summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ARCHITECTURAL COMPARISON")
    print("=" * 70)
    print(f"""
    {"Feature":<35} {"Original JAMEL":<25} {"JAMEL-COMPACT":<25}
    {"─"*35} {"─"*25} {"─"*25}
    {"Compressor location":<35} {"Separate frozen VLM":<25} {"Inside LLM layers":<25}
    {"Memory granularity":<35} {"Model-level (prefix)":<25} {"Per-layer (side memory)":<25}
    {"Memory update mechanism":<35} {"Batch re-compress":<25} {"FiLM-GRU + KF (recurrent)":<25}
    {"State tracking":<35} {"Stateless (recompute)":<25} {"Stateful (M_t → M_{t+1})":<25}
    {"Uncertainty awareness":<35} {"None":<25} {"Confidence vector C":<25}
    {"Action role":<35} {"Passive (text prefix)":<25} {"Control variable (FiLM)":<25}
    {"Read mechanism":<35} {"Prefix concatenation":<25} {"Cross-attention at each layer":<25}
    {"Hierarchical design":<35} {"No":<25} {"Yes (λ, W_inj per layer)":<25}
    {"Context cost":<35} {"O(T) grows with history":<25} {"O(N_m) fixed per layer":<25}
    {"Training":<35} {"SFT only":<25} {"SFT + RL (end-to-end)":<25}
    """)


if __name__ == "__main__":
    demo()
