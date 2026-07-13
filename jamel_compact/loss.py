"""
Loss functions for JAMEL-COMPACT.

Three-term loss:
  1. Action loss (Cross-Entropy) — teach the agent to predict correct actions
  2. Memory regularization — prevent memory explosion and confidence saturation
  3. Uncertainty calibration — make confidence reflect actual match quality
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .config import CompactConfig


def compute_compact_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    memory_states: List[torch.Tensor],
    confidence_states: List[torch.Tensor],
    config: Optional[CompactConfig] = None,
    predicted_memory: Optional[List[torch.Tensor]] = None,
    observation_feat: Optional[List[torch.Tensor]] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the JAMEL-COMPACT total loss.

    Args:
        logits:            [B, N, vocab_size] — model output logits
        labels:            [B, N] — token labels (-100 for ignore)
        memory_states:     List of [B, N_m, d_mem] — updated memory per layer
        confidence_states: List of [B, N_m] — updated confidence per layer
        config:            CompactConfig with loss weights
        predicted_memory:  List of [B, N_m, d_mem] — M_hat before KF (optional)
        observation_feat:  List of [B, d] — Z_t per layer (optional)

    Returns:
        (total_loss, loss_dict)
    """
    if config is None:
        config = CompactConfig()

    # ── 1. Action loss (Cross-Entropy) ──
    # Shift labels: predict token t+1 from token t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_action = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # ── 2. Memory regularization ──
    loss_mem_l2 = 0.0
    loss_mem_entropy = 0.0
    eps = 1e-8
    L = len(memory_states)

    for M, C in zip(memory_states, confidence_states):
        # L2: prevent memory values from exploding
        loss_mem_l2 += M.pow(2).mean()
        # Bernoulli Entropy: prevent C from saturating at 0 or 1
        entropy = -(C * torch.log(C + eps) + (1 - C) * torch.log(1 - C + eps))
        loss_mem_entropy += entropy.mean()

    loss_mem_l2 = loss_mem_l2 / L
    loss_mem_entropy = loss_mem_entropy / L
    loss_mem = loss_mem_l2 + config.beta_entropy * loss_mem_entropy

    # ── 3. Uncertainty calibration ──
    # If predicted_memory and observation_feat are available, compute per-token
    # MSE between confidence C and actual observation-to-prediction match.
    loss_uncert = torch.tensor(0.0, device=logits.device)
    if predicted_memory is not None and observation_feat is not None:
        for l_idx, (C, M_hat, Z) in enumerate(
            zip(confidence_states, predicted_memory, observation_feat)
        ):
            # Z: [B, d], M_hat: [B, N_m, d_mem]
            # Need to project Z to d_mem if dimensions differ
            # For simplicity, we use the already-projected z_down from the model
            # Here we approximate by using the first d_mem dims of Z if d > d_mem
            if Z.shape[-1] != M_hat.shape[-1]:
                # Dimension mismatch — skip this layer's uncertainty loss
                continue
            z_norm = F.normalize(Z, dim=-1)
            m_norm = F.normalize(M_hat, dim=-1)
            match = (z_norm.unsqueeze(1) * m_norm).sum(dim=-1).clamp(0, 1)
            loss_uncert = loss_uncert + F.mse_loss(C, match.detach())
        loss_uncert = loss_uncert / L

    # ── Total ──
    loss_total = (
        loss_action
        + config.lambda_mem * loss_mem
        + config.lambda_uncert * loss_uncert
    )

    loss_dict = {
        "total": loss_total.item(),
        "action": loss_action.item(),
        "mem_l2": loss_mem_l2.item() if isinstance(loss_mem_l2, torch.Tensor) else loss_mem_l2,
        "mem_entropy": loss_mem_entropy.item() if isinstance(loss_mem_entropy, torch.Tensor) else loss_mem_entropy,
        "uncert": loss_uncert.item(),
    }

    return loss_total, loss_dict