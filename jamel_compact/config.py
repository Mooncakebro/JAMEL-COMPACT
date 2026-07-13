"""
Configuration for JAMEL-COMPACT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompactConfig:
    """All hyperparameters for JAMEL-COMPACT model, training, and eval."""

    # ── Model ──
    base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    mem_dim: int = 512                # reduced memory dimension d_mem
    num_mem_tokens: int = 16           # N_m memory tokens per layer
    num_heads: int = 8                # attention heads in side memory
    freeze_base: bool = False         # freeze pretrained LLM weights
    num_act_tokens: int = 1           # action tokens in input sequence

    # ── Hierarchical hyperparameters ──
    lambda_shallow: float = 0.70
    lambda_mid: float = 0.85
    lambda_deep: float = 0.95
    inject_shallow: float = 0.8
    inject_mid: float = 0.5
    inject_deep: float = 0.3
    alpha_confidence: float = 0.1     # learning rate for confidence update

    # ── Training ──
    output_dir: str = "outputs/compact_ckpt"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    log_steps: int = 10
    save_steps: int = 500
    val_steps: int = 200

    # ── Loss weights ──
    lambda_mem: float = 0.001        # memory regularization weight
    lambda_uncert: float = 0.1       # uncertainty calibration weight
    beta_entropy: float = 0.01        # entropy regularization in mem loss

    # ── Data ──
    max_length: int = 8192
    train_file: str = ""
    val_file: str = ""
    val_ratio: float = 0.05
    image_resize: tuple = (640, 360)  # WEB_MODEL_IMAGE_SIZE

    # ── Eval ──
    eval_output: str = "outputs/compact_eval"
    max_steps: int = 50
    num_sessions: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    scalewob_root: str = "env/browser_env/scalewob-env"
    apps_mode: str = "test10"

    # ── TensorBoard ──
    tb_log_dir: str = "outputs/compact_tb"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_args(cls, **kwargs) -> "CompactConfig":
        cfg = cls()
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg