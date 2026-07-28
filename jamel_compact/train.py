"""
Training script for JAMEL-COMPACT.

Trains the model end-to-end with:
  - Action loss (coverage-weighted Cross-Entropy)
  - Observation prediction MSE and Gaussian NLL auxiliary losses
  - Memory L2 regularization
  - TensorBoard logging for all loss components and memory statistics

Usage:
    python -m jamel_compact.train \
        --train-file data/compact_train.parquet \
        --val-file data/compact_val.parquet \
        --base-model Qwen/Qwen3-VL-2B-Instruct \
        --output-dir outputs/compact_ckpt \
        --tb-log-dir outputs/compact_tb \
        --max-epochs 3
"""
from __future__ import annotations

import os
import sys

# ── Set CUDA_VISIBLE_DEVICES BEFORE importing torch ──
# If --gpu-ids is passed on the command line, we must set the env var
# before torch initializes the CUDA context. Once torch is imported and
# CUDA is initialized, changing CUDA_VISIBLE_DEVICES has no effect.
_gpu_ids_arg = ""
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpu-ids" and _i + 1 < len(sys.argv):
        _gpu_ids_arg = sys.argv[_i + 1]
        break
    if _arg.startswith("--gpu-ids="):
        _gpu_ids_arg = _arg.split("=", 1)[1]
        break
if _gpu_ids_arg:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids_arg

import argparse
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# tqdm progress bar
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None

# TensorBoard is optional — training works without it (logs to stdout only)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None

from .config import CompactConfig
from .model import JAMELCompactWrapper
from .data import CompactDataset, collate_fn, SessionChunkDataset, session_collate_fn
from .loss import compute_compact_loss

# F8: Optional memory diagnostics
try:
    import importlib.util
    if importlib.util.find_spec("scripts.probe_memory"):
        from scripts.probe_memory import log_memory_stats as _log_mem_stats
    else:
        _log_mem_stats = None
except ImportError:
    _log_mem_stats = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_action_embedding(
    action_input_ids: torch.Tensor,
    action_attention_mask: torch.Tensor,
    model,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert action token IDs into a fixed-size embedding for FiLM-GRU.
    Uses the pretrained token embedding layer + mean pooling.
    Works with both raw model and DataParallel-wrapped model.
    """
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    action_input_ids = action_input_ids.to(device)
    action_attention_mask = action_attention_mask.to(device)
    embed_layer = raw._get_input_embeddings()
    action_embeds = embed_layer(action_input_ids)  # [B, L_act, d]
    mask = action_attention_mask.unsqueeze(-1).to(action_embeds.dtype)
    return (action_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _unwrap(m):
    """Return the underlying model from DataParallel if present."""
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def _scalar(v):
    """Convert a value (tensor or float) to a Python float."""
    if isinstance(v, torch.Tensor):
        return v.mean().item()
    return float(v)


def _optimizer_group_lr(optimizer, group_name: str) -> Optional[float]:
    """Return the current learning rate for a named optimizer group."""
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return None


def _ensure_batch_dim(value):
    """Keep scalar tensors scatterable by ``DataParallel``."""
    if isinstance(value, torch.Tensor) and value.dim() == 0:
        return value.unsqueeze(0)
    return value


def _validate_and_save_best(
    model,
    val_dataloader: DataLoader,
    config: CompactConfig,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    best_val_loss: float,
) -> float:
    """Validate and immediately persist improved model weights."""
    val_loss = validate(
        model, val_dataloader, config, writer, global_step, device,
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_dir = Path(config.output_dir) / "best"
        _unwrap(model).save_pretrained(best_dir)
        print(f"  [best] New best val loss: {best_val_loss:.4f}")
    return best_val_loss


def _process_chunk_step(
    model,
    raw_model: JAMELCompactWrapper,
    config: CompactConfig,
    device: torch.device,
    step_data: dict,
    memory_states: list,
    variance_states: list,
    e_prev_list: list = None,
) -> tuple:
    """Process a single step within a session chunk.

    Handles visual feature pre-computation on the raw model and the forward
    pass through the (possibly DataParallel-wrapped) model.

    Args:
        model:           possibly DataParallel-wrapped model
        raw_model:       unwrapped model
        config:          training config
        device:          cuda device
        step_data:       dict with input_ids, attention_mask, labels,
                         action_input_ids, pixel_values, image_grid_thw,
                         sample_weights (optional)
        memory_states:   list of [B, N_m, d_mem] — carried from previous step
        variance_states: list of [B, N_m] — variance P, carried from previous step
        e_prev_list:     list of [B] or None — surprise from previous step per layer

    Returns:
        (loss, loss_dict, new_memory, new_variance, e_list)
    """
    input_ids = step_data["input_ids"].to(device)
    attention_mask = step_data["attention_mask"].to(device)
    observation_mask = step_data["observation_mask"].to(device)
    mm_token_type_ids = step_data.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)
    labels = step_data["labels"].to(device)
    action_input_ids = step_data["action_input_ids"].to(device)
    action_attention_mask = step_data["action_attention_mask"].to(device)
    pixel_values = step_data.get("pixel_values")
    if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device)
    image_grid_thw = step_data.get("image_grid_thw")
    if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.to(device)

    # Get action embedding (previous action → FiLM-GRU input)
    action_embed_input = get_action_embedding(
        action_input_ids, action_attention_mask, model, device,
    )

    # Pre-compute visual features on the raw model before DataParallel scatter
    inputs_embeds = None
    deepstack_features = None
    visual_pos_mask = None
    if pixel_values is not None and raw_model._has_visual_encoder():
        embed_layer = raw_model._get_input_embeddings()
        h_embed = embed_layer(input_ids)
        inputs_embeds, deepstack_features, visual_pos_mask = \
            raw_model._inject_visual_features(
                h_embed, input_ids, pixel_values, image_grid_thw,
            )

    # ── TBPTT-1: Detach memory states from the computation graph before
    # passing to the model forward. This prevents full BPTT (backprop
    # through time) which would retain all intermediate steps' graphs and
    # cause OOM. Instead, the FiLM-GRU learns from the per-step loss signal
    # only — memory carries forward VALUES but not GRADIENTS.
    #
    # This is truncated BPTT (TBPTT) with truncation length 1: each step's
    # loss backprops only through that step's forward pass, but the memory
    # VALUES (not gradients) carry forward across steps. This still trains
    # the recurrent dynamics because each step sees a non-trivial memory
    # state (not the zero-initialized state), so the inject/correct modules
    # learn to use evolved memory.
    #
    # When config.tbptt_detach=False, states are NOT detached — enabling
    # full BPTT for ablation experiments (much higher memory cost).
    detach = config.tbptt_detach
    mem_input = [m.detach() if detach and isinstance(m, torch.Tensor) else m
                 for m in memory_states]
    var_input = [c.detach() if detach and isinstance(c, torch.Tensor) else c
                 for c in variance_states]
    e_prev_input = [e.detach() if detach and isinstance(e, torch.Tensor) else e
                    for e in (e_prev_list or [None] * len(memory_states))]

    # Get sample weights (F7: coverage-weighted SFT)
    sample_weights = step_data.get("sample_weights")
    if sample_weights is not None and isinstance(sample_weights, torch.Tensor):
        sample_weights = sample_weights.to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        observation_mask=observation_mask,
        mm_token_type_ids=mm_token_type_ids,
        action_embed_input=action_embed_input,
        memory_states=mem_input,
        variance_states=var_input,
        labels=labels,
        pixel_values=None,
        image_grid_thw=image_grid_thw,
        inputs_embeds=inputs_embeds,
        deepstack_features=deepstack_features,
        visual_pos_mask=visual_pos_mask,
        e_prev_list=e_prev_input,
        sample_weights=sample_weights,
    )

    loss = outputs["loss"]
    if loss.dim() > 0:
        loss = loss.mean()

    def _to_scalar(v):
        if isinstance(v, torch.Tensor):
            return v.mean().item()
        return float(v)
    loss_dict = {k: _to_scalar(v) for k, v in outputs["loss_dict"].items()}

    # Detach new memory before returning — values carry forward, not gradients
    new_mem = [m.detach() if detach and isinstance(m, torch.Tensor) else m
               for m in outputs["new_memory"]]
    new_var = [c.detach() if detach and isinstance(c, torch.Tensor) else c
               for c in outputs["new_variance"]]
    new_e_list = [e.detach() if detach and isinstance(e, torch.Tensor) else e
                  for e in outputs.get("e_list", [None] * len(new_mem))]

    return loss, loss_dict, new_mem, new_var, new_e_list


def train_one_epoch(
    model: JAMELCompactWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: CompactConfig,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    epoch: int,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    val_dataloader: Optional[DataLoader] = None,
    best_val_loss: float = float("inf"),
) -> tuple[int, float, Optional[int]]:
    """Train for one epoch and track the best mid-epoch validation loss.

    Supports both single-step (chunk_size=1) and session-chunked training
    (chunk_size>1). In chunked mode, memory carries forward across steps
    within each chunk, training the recurrent dynamics of the FiLM-GRU.
    """
    model.train()
    raw_model = _unwrap(model)
    total_steps = len(dataloader)
    accum_steps = config.gradient_accumulation_steps
    use_chunking = config.chunk_size > 1
    optimizer.zero_grad()
    last_val_step = None

    # Build progress bar
    if _TQDM_AVAILABLE:
        pbar = tqdm(
            dataloader, desc=f"Epoch {epoch}", total=total_steps,
            unit="chunk" if use_chunking else "batch", leave=True,
        )
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar):
        batch_start = time.time()

        if use_chunking:
            # ── Session-chunked training ──
            # batch is a dict of lists (one entry per step in the chunk)
            chunk_size = batch["chunk_size"]

            # Initialize memory and e_prev at the start of each chunk
            memory_states, variance_states = raw_model.init_memory(1, device)
            e_prev_list = None  # None for first step in chunk

            total_chunk_loss = None
            last_loss_dict = {}
            all_loss_dicts = []

            for s in range(chunk_size):
                # sample_weights[s] indexes a 1-d tensor → produces a 0-d
                # scalar. DataParallel.scatter cannot chunk 0-d tensors, so
                # unsqueeze to [1] (B=1) to keep it 1-dimensional.
                sw = batch.get("sample_weights", [None] * chunk_size)[s]
                sw = _ensure_batch_dim(sw)

                step_data = {
                    "input_ids": batch["input_ids"][s],
                    "attention_mask": batch["attention_mask"][s],
                    "observation_mask": batch["observation_mask"][s],
                    "mm_token_type_ids": batch.get(
                        "mm_token_type_ids", [None] * chunk_size,
                    )[s],
                    "labels": batch["labels"][s],
                    "action_input_ids": batch["action_input_ids"][s],
                    "action_attention_mask": batch["action_attention_mask"][s],
                    "pixel_values": batch["pixel_values"][s],
                    "image_grid_thw": batch["image_grid_thw"][s],
                    "sample_weights": sw,
                }

                loss, loss_dict, memory_states, variance_states, e_list = \
                    _process_chunk_step(
                        model, raw_model, config, device,
                        step_data, memory_states, variance_states,
                        e_prev_list=e_prev_list,
                    )

                # Carry e_list forward as next step's e_prev_list
                e_prev_list = e_list

                # Accumulate loss across steps in the chunk
                # Weight each step equally
                step_loss = loss / chunk_size
                total_chunk_loss = (
                    step_loss
                    if total_chunk_loss is None
                    else total_chunk_loss + step_loss
                )
                last_loss_dict = loss_dict
                all_loss_dicts.append(loss_dict)

            # Average loss dict across steps
            loss_dict = {}
            for k in last_loss_dict:
                loss_dict[k] = sum(d[k] for d in all_loss_dicts) / chunk_size

            loss = total_chunk_loss / accum_steps

        else:
            # ── Single-step training (original path) ──
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            observation_mask = batch["observation_mask"].to(device)
            mm_token_type_ids = batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(device)
            labels = batch["labels"].to(device)
            action_input_ids = batch["action_input_ids"].to(device)
            action_attention_mask = batch["action_attention_mask"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = image_grid_thw.to(device)

            action_embed_input = get_action_embedding(
                action_input_ids, action_attention_mask, model, device,
            )
            B = input_ids.shape[0]
            memory_states, variance_states = raw_model.init_memory(B, device)

            # Get sample weights (F7)
            sample_weights = batch.get("sample_weights")
            if sample_weights is not None and isinstance(sample_weights, torch.Tensor):
                sample_weights = sample_weights.to(device)

            # Pre-compute visual features on the raw model before DataParallel
            inputs_embeds = None
            deepstack_features = None
            visual_pos_mask = None
            if pixel_values is not None and raw_model._has_visual_encoder():
                embed_layer = raw_model._get_input_embeddings()
                h_embed = embed_layer(input_ids)
                inputs_embeds, deepstack_features, visual_pos_mask = \
                    raw_model._inject_visual_features(
                        h_embed, input_ids, pixel_values, image_grid_thw,
                    )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                observation_mask=observation_mask,
                mm_token_type_ids=mm_token_type_ids,
                action_embed_input=action_embed_input,
                memory_states=memory_states,
                variance_states=variance_states,
                labels=labels,
                pixel_values=None,
                image_grid_thw=image_grid_thw,
                inputs_embeds=inputs_embeds,
                deepstack_features=deepstack_features,
                visual_pos_mask=visual_pos_mask,
                sample_weights=sample_weights,
            )

            loss = outputs["loss"]
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / accum_steps

            def _to_scalar(v):
                if isinstance(v, torch.Tensor):
                    return v.mean().item()
                return float(v)
            loss_dict = {k: _to_scalar(v) for k, v in outputs["loss_dict"].items()}

        # Backward
        loss.backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            memory_lr = _optimizer_group_lr(optimizer, "memory")
            base_lr = _optimizer_group_lr(optimizer, "base")
            lr = memory_lr if memory_lr is not None else optimizer.param_groups[0]["lr"]

            # ── TensorBoard logging ──
            if global_step % config.log_steps == 0:
                elapsed = time.time() - batch_start
                lr_text = f"memory_lr={lr:.2e} "
                if base_lr is not None:
                    lr_text += f"base_lr={base_lr:.2e} "

                if writer is not None:
                    writer.add_scalar("train/loss_total", loss_dict["total"], global_step)
                    writer.add_scalar("train/loss_action", loss_dict["action"], global_step)
                    writer.add_scalar("train/loss_action_unweighted", loss_dict.get("action_unweighted", loss_dict["action"]), global_step)
                    writer.add_scalar("train/loss_mem_l2", loss_dict["mem_l2"], global_step)
                    writer.add_scalar("train/loss_obs", loss_dict.get("obs", 0.0), global_step)
                    writer.add_scalar("train/loss_nll", loss_dict.get("nll", 0.0), global_step)
                    writer.add_scalar("train/learning_rate", lr, global_step)
                    writer.add_scalar("train/memory_learning_rate", lr, global_step)
                    if base_lr is not None:
                        writer.add_scalar("train/base_learning_rate", base_lr, global_step)
                    writer.add_scalar("train/step_time_s", elapsed, global_step)

                    # F8: Memory diagnostics (every 100 steps to avoid overhead)
                    if global_step % 100 == 0 and _log_mem_stats is not None:
                        try:
                            _log_mem_stats(raw_model, writer, global_step)
                        except Exception:
                            pass  # don't crash training on diagnostics

                print(
                    f"  [epoch {epoch} step {global_step}] "
                    f"loss={loss_dict['total']:.4f} "
                    f"action={loss_dict['action']:.4f} "
                    f"mem_l2={loss_dict['mem_l2']:.6f} "
                    f"obs={loss_dict.get('obs', 0.0):.4f} "
                    f"nll={loss_dict.get('nll', 0.0):.4f} "
                    f"{lr_text}"
                    f"time={elapsed:.2f}s"
                )

            # Update progress bar
            if _TQDM_AVAILABLE:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "action": f"{loss_dict['action']:.4f}",
                    "memory_lr": f"{lr:.2e}",
                    "step": global_step,
                })

            # ── LR scheduler step (per optimizer step, not per epoch) ──
            if scheduler is not None:
                scheduler.step()

            # ── Save checkpoint ──
            if global_step % config.save_steps == 0:
                ckpt_dir = Path(config.output_dir) / f"global_step_{global_step}"
                raw_model.save_pretrained(ckpt_dir)
                print(f"  [checkpoint] saved to {ckpt_dir}")

            # Validate during the epoch so a later auxiliary-loss divergence
            # cannot overwrite the best weights observed earlier.
            if (
                val_dataloader is not None
                and config.val_steps > 0
                and global_step % config.val_steps == 0
            ):
                best_val_loss = _validate_and_save_best(
                    model, val_dataloader, config, writer,
                    global_step, device, best_val_loss,
                )
                last_val_step = global_step

    # Close progress bar
    if _TQDM_AVAILABLE:
        pbar.close()

    return global_step, best_val_loss, last_val_step


def validate(
    model: JAMELCompactWrapper,
    dataloader: DataLoader,
    config: CompactConfig,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
) -> float:
    """Run validation and log to TensorBoard. Returns average loss.

    Supports both single-step and session-chunked validation.
    """
    model.eval()
    raw_model = _unwrap(model)
    use_chunking = config.chunk_size > 1
    total_loss = 0.0
    total_action_loss = 0.0
    total_mem_loss = 0.0
    total_obs_loss = 0.0
    total_nll_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        if _TQDM_AVAILABLE:
            pbar = tqdm(dataloader, desc="Validating",
                        unit="chunk" if use_chunking else "batch", leave=False)
        else:
            pbar = dataloader

        for batch in pbar:
            if use_chunking:
                # ── Session-chunked validation ──
                chunk_size = batch["chunk_size"]
                memory_states, variance_states = raw_model.init_memory(1, device)

                chunk_loss = 0.0
                chunk_action = 0.0
                chunk_mem = 0.0
                chunk_obs = 0.0
                chunk_nll = 0.0
                e_prev_list = None

                for s in range(chunk_size):
                    step_data = {
                        "input_ids": batch["input_ids"][s],
                        "attention_mask": batch["attention_mask"][s],
                        "observation_mask": batch["observation_mask"][s],
                        "mm_token_type_ids": batch.get(
                            "mm_token_type_ids", [None] * chunk_size,
                        )[s],
                        "labels": batch["labels"][s],
                        "action_input_ids": batch["action_input_ids"][s],
                        "action_attention_mask": batch["action_attention_mask"][s],
                        "pixel_values": batch["pixel_values"][s],
                        "image_grid_thw": batch["image_grid_thw"][s],
                        "sample_weights": _ensure_batch_dim(
                            batch.get("sample_weights", [None] * chunk_size)[s]
                        ),
                    }

                    loss, loss_dict, memory_states, variance_states, e_list = \
                        _process_chunk_step(
                            model, raw_model, config, device,
                            step_data, memory_states, variance_states,
                            e_prev_list=e_prev_list,
                        )
                    e_prev_list = e_list

                    chunk_loss += loss_dict["total"]
                    chunk_action += loss_dict["action"]
                    chunk_mem += loss_dict["mem_l2"]
                    chunk_obs += loss_dict.get("obs", 0.0)
                    chunk_nll += loss_dict.get("nll", 0.0)

                # Average over steps in chunk
                total_loss += chunk_loss / chunk_size
                total_action_loss += chunk_action / chunk_size
                total_mem_loss += chunk_mem / chunk_size
                total_obs_loss += chunk_obs / chunk_size
                total_nll_loss += chunk_nll / chunk_size
                num_batches += 1

                if _TQDM_AVAILABLE:
                    pbar.set_postfix({"val_loss": f"{chunk_loss / chunk_size:.4f}"})

            else:
                # ── Single-step validation ──
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                observation_mask = batch["observation_mask"].to(device)
                mm_token_type_ids = batch.get("mm_token_type_ids")
                if mm_token_type_ids is not None:
                    mm_token_type_ids = mm_token_type_ids.to(device)
                labels = batch["labels"].to(device)
                action_input_ids = batch["action_input_ids"].to(device)
                action_attention_mask = batch["action_attention_mask"].to(device)
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(device)
                image_grid_thw = batch.get("image_grid_thw")
                if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
                    image_grid_thw = image_grid_thw.to(device)

                action_embed_input = get_action_embedding(
                    action_input_ids, action_attention_mask, model, device,
                )
                B = input_ids.shape[0]
                memory_states, variance_states = raw_model.init_memory(B, device)

                # Get sample weights (F7)
                sample_weights = batch.get("sample_weights")
                if sample_weights is not None and isinstance(sample_weights, torch.Tensor):
                    sample_weights = sample_weights.to(device)

                inputs_embeds = None
                deepstack_features = None
                visual_pos_mask = None
                if pixel_values is not None and raw_model._has_visual_encoder():
                    embed_layer = raw_model._get_input_embeddings()
                    h_embed = embed_layer(input_ids)
                    inputs_embeds, deepstack_features, visual_pos_mask = \
                        raw_model._inject_visual_features(
                            h_embed, input_ids, pixel_values, image_grid_thw,
                        )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    observation_mask=observation_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    action_embed_input=action_embed_input,
                    memory_states=memory_states,
                    variance_states=variance_states,
                    labels=labels,
                    pixel_values=None,
                    image_grid_thw=image_grid_thw,
                    inputs_embeds=inputs_embeds,
                    deepstack_features=deepstack_features,
                    visual_pos_mask=visual_pos_mask,
                    sample_weights=sample_weights,
                )

                ld = outputs["loss_dict"]
                total_loss += _scalar(ld["total"])
                total_action_loss += _scalar(ld["action"])
                total_mem_loss += _scalar(ld["mem_l2"])
                total_obs_loss += _scalar(ld.get("obs", 0.0))
                total_nll_loss += _scalar(ld.get("nll", 0.0))
                num_batches += 1

                if _TQDM_AVAILABLE:
                    pbar.set_postfix({"val_loss": f"{_scalar(ld['total']):.4f}"})

        if _TQDM_AVAILABLE:
            pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    avg_action = total_action_loss / max(num_batches, 1)
    avg_mem = total_mem_loss / max(num_batches, 1)
    avg_obs = total_obs_loss / max(num_batches, 1)
    avg_nll = total_nll_loss / max(num_batches, 1)

    if writer is not None:
        writer.add_scalar("val/loss_total", avg_loss, global_step)
        writer.add_scalar("val/loss_action", avg_action, global_step)
        writer.add_scalar("val/loss_mem_l2", avg_mem, global_step)
        writer.add_scalar("val/loss_obs", avg_obs, global_step)
        writer.add_scalar("val/loss_nll", avg_nll, global_step)

    print(
        f"  [val step {global_step}] "
        f"loss={avg_loss:.4f} "
        f"action={avg_action:.4f} "
        f"mem_l2={avg_mem:.6f} "
        f"obs={avg_obs:.4f} "
        f"nll={avg_nll:.4f}"
    )

    model.train()
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="JAMEL-COMPACT Training")
    parser.add_argument("--train-file", required=True, help="Train parquet file")
    parser.add_argument("--val-file", required=True, help="Val parquet file")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Pretrained base model name or path")
    parser.add_argument("--output-dir", default="outputs/compact_ckpt")
    parser.add_argument("--tb-log-dir", default="outputs/compact_tb")
    parser.add_argument("--mem-dim", type=int, default=512)
    parser.add_argument("--num-mem-tokens", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Per-device batch size in single-step mode. Recurrent chunked "
             "training (--chunk-size > 1) always processes one chunk at a "
             "time; use --grad-accum to increase its effective batch size.",
    )
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Base-model learning rate when --train-base is enabled.")
    parser.add_argument("--memory-lr", type=float, default=5e-6,
                        help="Learning rate for side-memory and action modules.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--lambda-obs", type=float, default=0.01,
                        help="Weight for observation-prediction loss.")
    parser.add_argument("--lambda-nll", type=float, default=0.01,
                        help="Weight for Gaussian NLL loss.")
    base_training_group = parser.add_mutually_exclusive_group()
    base_training_group.add_argument(
        "--freeze-base", dest="freeze_base", action="store_true",
        help="Freeze pretrained LLM weights (default).",
    )
    base_training_group.add_argument(
        "--train-base", dest="freeze_base", action="store_false",
        help="Also fine-tune the pretrained LLM using --lr.",
    )
    parser.set_defaults(freeze_base=True)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2'). "
                             "Empty = all available GPUs. "
                             "For single-GPU training, specify one ID (e.g. '0').")
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument(
        "--model-parallel", dest="model_parallel", action="store_true",
        help="Shard a frozen base model across all visible GPUs.",
    )
    parallel_group.add_argument(
        "--data-parallel", dest="model_parallel", action="store_false",
        help="Force legacy batch-splitting DataParallel.",
    )
    parser.set_defaults(model_parallel=None)
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Session-chunked training: number of consecutive "
                             "steps per chunk. 1 = single-step (original). "
                             ">1 = multi-step, carries memory forward across "
                             "steps to train FiLM-GRU recurrent dynamics.")
    parser.add_argument("--coverage-weight-eta", type=float, default=0.0,
                        help="F7: Coverage-weighted SFT. 0 = off; >0 upweights "
                             "samples with high coverage delta (novelty).")
    args = parser.parse_args()

    if args.chunk_size > 1 and args.batch_size != 1:
        print(
            f"[train] WARNING: --batch-size {args.batch_size} is incompatible "
            "with recurrent session chunks; overriding it to 1. Increase "
            "--grad-accum for a larger effective batch."
        )
        args.batch_size = 1

    # ── GPU selection ──
    # CUDA_VISIBLE_DEVICES was already set before torch import (see top of file).
    # Here we just report what's visible.
    if args.gpu_ids:
        print(f"[train] Requested GPUs: {args.gpu_ids} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    else:
        print("[train] GPU_IDS not set — using all visible GPUs")

    # Clear CUDA cache after changing visible devices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.model_parallel is None:
        args.model_parallel = bool(
            num_gpus > 1
            and args.freeze_base
            and args.chunk_size > 1
            and args.batch_size == 1
        )
        if args.model_parallel:
            print(
                "[train] Auto-enabled model parallelism for frozen, "
                "chunked B=1 multi-GPU training."
            )
    if args.model_parallel and num_gpus < 2:
        parser.error("--model-parallel requires at least two visible GPUs")
    if args.model_parallel and not args.freeze_base:
        parser.error(
            "--model-parallel uses Transformers/Accelerate device-map "
            "sharding and only supports a frozen base model. Set "
            "FREEZE_BASE=1, or use an FSDP/ZeRO-3 training path to fine-tune "
            "the sharded base model."
        )
    use_data_parallel = num_gpus > 1 and not args.model_parallel

    parallel_mode = (
        "MODEL_PARALLEL" if args.model_parallel
        else "DATA_PARALLEL" if use_data_parallel
        else "SINGLE_DEVICE"
    )
    print(f"[train] device={device}, GPUs visible={num_gpus}, mode={parallel_mode}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

    # ── Build config ──
    config = CompactConfig.from_args(
        base_model_name=args.base_model,
        mem_dim=args.mem_dim,
        num_mem_tokens=args.num_mem_tokens,
        output_dir=args.output_dir,
        tb_log_dir=args.tb_log_dir,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        memory_learning_rate=args.memory_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        val_steps=args.val_steps,
        lambda_obs=args.lambda_obs,
        lambda_nll=args.lambda_nll,
        freeze_base=args.freeze_base,
        model_parallel=args.model_parallel,
        gradient_checkpointing=not args.no_grad_checkpoint,
        bf16=args.bf16,
        seed=args.seed,
        chunk_size=args.chunk_size,
        coverage_weight_eta=args.coverage_weight_eta,
    )

    # ── Build model ──
    print(f"[train] Loading base model: {config.base_model_name}")
    model = JAMELCompactWrapper(config)
    if config.model_parallel:
        device = model.input_device
    else:
        model = model.to(device)

    # Wrap with DataParallel for multi-GPU training
    if use_data_parallel:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"[train] DataParallel active on {num_gpus} GPUs: {device_ids}")

    raw_model = _unwrap(model)
    param_info = raw_model.count_parameters()
    print(f"[train] Base params:   {param_info['base'] / 1e9:.2f}B")
    print(f"[train] New params:    {param_info['new'] / 1e6:.1f}M")
    print(f"[train] Total:         {param_info['total'] / 1e9:.2f}B")
    print(f"[train] Overhead:      {param_info['overhead_pct']:.1f}%")

    # ── Build dataset ──
    use_chunking = config.chunk_size > 1
    print(f"[train] Loading data: {args.train_file}")
    print(f"[train] Chunk size: {config.chunk_size} "
          f"({'session-chunked' if use_chunking else 'single-step'})")
    if use_chunking and use_data_parallel and config.per_device_batch_size == 1:
        print(
            "[train] WARNING: DataParallel cannot split chunked B=1 inputs; "
            "only the first GPU will compute. Use --model-parallel for large "
            "frozen models."
        )

    train_dataset = CompactDataset(
        parquet_files=args.train_file,
        tokenizer=raw_model.tokenizer,
        processor=raw_model.processor,
        max_length=config.max_length,
        image_resize=config.image_resize,
        coverage_weight_eta=config.coverage_weight_eta,
    )
    val_dataset = CompactDataset(
        parquet_files=args.val_file,
        tokenizer=raw_model.tokenizer,
        processor=raw_model.processor,
        max_length=config.max_length,
        image_resize=config.image_resize,
        coverage_weight_eta=config.coverage_weight_eta,
    )

    if use_chunking:
        # Wrap with SessionChunkDataset for multi-step training
        train_dataset = SessionChunkDataset(train_dataset, chunk_size=config.chunk_size,
                                             coverage_weight_eta=config.coverage_weight_eta)
        val_dataset = SessionChunkDataset(val_dataset, chunk_size=config.chunk_size,
                                           coverage_weight_eta=config.coverage_weight_eta)
        chunk_batch_size = 1
        print(f"[train] Chunked mode: batch_size={chunk_batch_size} "
              f"× {config.chunk_size} steps/chunk")
        pad_token_id = raw_model.tokenizer.pad_token_id or 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=chunk_batch_size,
            shuffle=True,
            collate_fn=lambda b: session_collate_fn(b, pad_token_id),
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=chunk_batch_size,
            shuffle=False,
            collate_fn=lambda b: session_collate_fn(b, pad_token_id),
            num_workers=2,
        )
    else:
        # Single-step mode (original)
        effective_batch = config.per_device_batch_size * max(num_gpus, 1)
        print(f"[train] Per-GPU batch size: {config.per_device_batch_size}"
              f" × {max(num_gpus, 1)} GPUs = effective batch {effective_batch}")
        pad_token_id = raw_model.tokenizer.pad_token_id or 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_token_id),
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_token_id),
            num_workers=2,
        )

    # ── Optimizer ──
    # Keep the new recurrent-memory system on a lower learning rate than the
    # pretrained backbone. When the base is frozen, only the memory group is
    # present; --train-base explicitly opts into full-model fine-tuning.
    memory_params = [
        p for p in (
            list(raw_model.side_memories.parameters())
            + list(raw_model.action_embed.parameters())
        )
        if p.requires_grad
    ]
    base_params = [p for p in raw_model.llm.parameters() if p.requires_grad]
    optimizer_groups = [
        {
            "name": "memory",
            "params": memory_params,
            "lr": config.memory_learning_rate,
        }
    ]
    if base_params:
        optimizer_groups.append({
            "name": "base",
            "params": base_params,
            "lr": config.learning_rate,
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=config.weight_decay,
    )
    print(f"[train] Memory learning rate: {config.memory_learning_rate:.2e}")
    if base_params:
        print(f"[train] Base learning rate:   {config.learning_rate:.2e}")
    else:
        print("[train] Base model frozen")

    # ── LR scheduler (cosine with warmup) ──
    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── TensorBoard (optional) ──
    if _TB_AVAILABLE:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
        print(f"[train] TensorBoard logging to {config.tb_log_dir}")
        print(f"  Run: tensorboard --logdir {config.tb_log_dir}")
        for k, v in config.to_dict().items():
            writer.add_text("config", f"{k}: {v}")
    else:
        writer = None
        print("[train] TensorBoard not available (pip install tensorboard)")
        print(f"[train] Logs will go to stdout only")

    # ── Training loop ──
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")

        global_step, best_val_loss, last_val_step = train_one_epoch(
            model, train_loader, optimizer, config, writer,
            global_step, device, epoch,
            scheduler=scheduler,
            val_dataloader=val_loader,
            best_val_loss=best_val_loss,
        )

        # Always validate the completed epoch unless the same weights were
        # already validated exactly at the final optimizer step.
        if last_val_step != global_step:
            best_val_loss = _validate_and_save_best(
                model, val_loader, config, writer,
                global_step, device, best_val_loss,
            )

    # ── Save final model ──
    final_dir = Path(config.output_dir) / "final"
    raw_model.save_pretrained(final_dir)
    print(f"\n[train] Final model saved to {final_dir}")

    if writer is not None:
        writer.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()
