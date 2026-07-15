"""
Training script for JAMEL-COMPACT.

Trains the model end-to-end with:
  - Action loss (Cross-Entropy)
  - Memory regularization (L2 + entropy)
  - Uncertainty calibration (MSE)
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_action_embedding(action_input_ids: torch.Tensor, model,
                          device: torch.device) -> torch.Tensor:
    """
    Convert action token IDs into a fixed-size embedding for FiLM-GRU.
    Uses the pretrained token embedding layer + mean pooling.
    Works with both raw model and DataParallel-wrapped model.
    """
    raw = model.module if isinstance(model, torch.nn.DataParallel) else model
    action_input_ids = action_input_ids.to(device)
    embed_layer = raw._get_input_embeddings()
    action_embeds = embed_layer(action_input_ids)  # [B, L_act, d]
    # Mean pool over action tokens
    return action_embeds.mean(dim=1)  # [B, d]


def _unwrap(m):
    """Return the underlying model from DataParallel if present."""
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def _scalar(v):
    """Convert a value (tensor or float) to a Python float."""
    if isinstance(v, torch.Tensor):
        return v.mean().item()
    return float(v)


def _process_chunk_step(
    model,
    raw_model: JAMELCompactWrapper,
    config: CompactConfig,
    device: torch.device,
    step_data: dict,
    memory_states: list,
    confidence_states: list,
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
                         action_input_ids, pixel_values, image_grid_thw
        memory_states:   list of [B, N_m, d_mem] — carried from previous step
        confidence_states: list of [B, N_m] — carried from previous step

    Returns:
        (loss, loss_dict, new_memory, new_confidence)
    """
    input_ids = step_data["input_ids"].to(device)
    attention_mask = step_data["attention_mask"].to(device)
    labels = step_data["labels"].to(device)
    action_input_ids = step_data["action_input_ids"].to(device)
    pixel_values = step_data.get("pixel_values")
    if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device)
    image_grid_thw = step_data.get("image_grid_thw")
    if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = image_grid_thw.to(device)

    # Get action embedding (previous action → FiLM-GRU input)
    action_embed_input = get_action_embedding(action_input_ids, model, device)

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

    # ── Detach memory states from the computation graph before passing to
    # the model forward. This prevents BPTT (backprop through time) which
    # would require retaining all intermediate steps' graphs and cause OOM.
    # Instead, the FiLM-GRU learns from the per-step loss signal only —
    # memory carries forward VALUES but not GRADIENTS.
    #
    # This is equivalent to truncated BPTT (TBPTT) with truncation length 1:
    # each step's loss backprops only through that step's forward pass,
    # but the memory VALUES (not gradients) carry forward across steps.
    # This still trains the recurrent dynamics because each step sees a
    # non-trivial memory state (not the zero-initialized state), so the
    # inject/correct modules learn to use evolved memory.
    mem_input = [m.detach() if isinstance(m, torch.Tensor) else m
                 for m in memory_states]
    conf_input = [c.detach() if isinstance(c, torch.Tensor) else c
                  for c in confidence_states]

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_embed_input=action_embed_input,
        memory_states=mem_input,
        confidence_states=conf_input,
        labels=labels,
        pixel_values=None,
        image_grid_thw=None,
        inputs_embeds=inputs_embeds,
        deepstack_features=deepstack_features,
        visual_pos_mask=visual_pos_mask,
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
    new_mem = [m.detach() if isinstance(m, torch.Tensor) else m
               for m in outputs["new_memory"]]
    new_conf = [c.detach() if isinstance(c, torch.Tensor) else c
                for c in outputs["new_confidence"]]

    return loss, loss_dict, new_mem, new_conf


def train_one_epoch(
    model: JAMELCompactWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: CompactConfig,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    epoch: int,
) -> int:
    """Train for one epoch. Returns updated global_step.

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

            # Initialize memory at the start of each chunk
            memory_states, confidence_states = raw_model.init_memory(1, device)

            total_chunk_loss = torch.tensor(0.0, device=device, requires_grad=False)
            last_loss_dict = {}
            all_loss_dicts = []

            for s in range(chunk_size):
                step_data = {
                    "input_ids": batch["input_ids"][s],
                    "attention_mask": batch["attention_mask"][s],
                    "labels": batch["labels"][s],
                    "action_input_ids": batch["action_input_ids"][s],
                    "pixel_values": batch["pixel_values"][s],
                    "image_grid_thw": batch["image_grid_thw"][s],
                }

                loss, loss_dict, memory_states, confidence_states = \
                    _process_chunk_step(
                        model, raw_model, config, device,
                        step_data, memory_states, confidence_states,
                    )

                # Accumulate loss across steps in the chunk
                # Weight each step equally
                total_chunk_loss = total_chunk_loss + loss / chunk_size
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
            labels = batch["labels"].to(device)
            action_input_ids = batch["action_input_ids"].to(device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                pixel_values = pixel_values.to(device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
                image_grid_thw = image_grid_thw.to(device)

            action_embed_input = get_action_embedding(action_input_ids, model, device)
            B = input_ids.shape[0]
            memory_states, confidence_states = raw_model.init_memory(B, device)

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
                action_embed_input=action_embed_input,
                memory_states=memory_states,
                confidence_states=confidence_states,
                labels=labels,
                pixel_values=None,
                image_grid_thw=None,
                inputs_embeds=inputs_embeds,
                deepstack_features=deepstack_features,
                visual_pos_mask=visual_pos_mask,
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

            # ── TensorBoard logging ──
            if global_step % config.log_steps == 0:
                elapsed = time.time() - batch_start
                lr = optimizer.param_groups[0]["lr"]

                if writer is not None:
                    writer.add_scalar("train/loss_total", loss_dict["total"], global_step)
                    writer.add_scalar("train/loss_action", loss_dict["action"], global_step)
                    writer.add_scalar("train/loss_mem_l2", loss_dict["mem_l2"], global_step)
                    writer.add_scalar("train/loss_mem_entropy", loss_dict["mem_entropy"], global_step)
                    writer.add_scalar("train/loss_uncert", loss_dict["uncert"], global_step)
                    writer.add_scalar("train/learning_rate", lr, global_step)
                    writer.add_scalar("train/step_time_s", elapsed, global_step)

                print(
                    f"  [epoch {epoch} step {global_step}] "
                    f"loss={loss_dict['total']:.4f} "
                    f"action={loss_dict['action']:.4f} "
                    f"mem_l2={loss_dict['mem_l2']:.6f} "
                    f"uncert={loss_dict['uncert']:.4f} "
                    f"lr={lr:.2e} "
                    f"time={elapsed:.2f}s"
                )

            # Update progress bar
            if _TQDM_AVAILABLE:
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "action": f"{loss_dict['action']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "step": global_step,
                })

            # ── Save checkpoint ──
            if global_step % config.save_steps == 0:
                ckpt_dir = Path(config.output_dir) / f"global_step_{global_step}"
                raw_model.save_pretrained(ckpt_dir)
                print(f"  [checkpoint] saved to {ckpt_dir}")

    # Close progress bar
    if _TQDM_AVAILABLE:
        pbar.close()

    return global_step


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
    total_uncert_loss = 0.0
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
                memory_states, confidence_states = raw_model.init_memory(1, device)

                chunk_loss = 0.0
                chunk_action = 0.0
                chunk_mem = 0.0
                chunk_uncert = 0.0

                for s in range(chunk_size):
                    step_data = {
                        "input_ids": batch["input_ids"][s],
                        "attention_mask": batch["attention_mask"][s],
                        "labels": batch["labels"][s],
                        "action_input_ids": batch["action_input_ids"][s],
                        "pixel_values": batch["pixel_values"][s],
                        "image_grid_thw": batch["image_grid_thw"][s],
                    }

                    loss, loss_dict, memory_states, confidence_states = \
                        _process_chunk_step(
                            model, raw_model, config, device,
                            step_data, memory_states, confidence_states,
                        )

                    chunk_loss += loss_dict["total"]
                    chunk_action += loss_dict["action"]
                    chunk_mem += loss_dict["mem_l2"]
                    chunk_uncert += loss_dict["uncert"]

                # Average over steps in chunk
                total_loss += chunk_loss / chunk_size
                total_action_loss += chunk_action / chunk_size
                total_mem_loss += chunk_mem / chunk_size
                total_uncert_loss += chunk_uncert / chunk_size
                num_batches += 1

                if _TQDM_AVAILABLE:
                    pbar.set_postfix({"val_loss": f"{chunk_loss / chunk_size:.4f}"})

            else:
                # ── Single-step validation ──
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                action_input_ids = batch["action_input_ids"].to(device)
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(device)
                image_grid_thw = batch.get("image_grid_thw")
                if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
                    image_grid_thw = image_grid_thw.to(device)

                action_embed_input = get_action_embedding(action_input_ids, model, device)
                B = input_ids.shape[0]
                memory_states, confidence_states = raw_model.init_memory(B, device)

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
                    action_embed_input=action_embed_input,
                    memory_states=memory_states,
                    confidence_states=confidence_states,
                    labels=labels,
                    pixel_values=None,
                    image_grid_thw=None,
                    inputs_embeds=inputs_embeds,
                    deepstack_features=deepstack_features,
                    visual_pos_mask=visual_pos_mask,
                )

                ld = outputs["loss_dict"]
                total_loss += _scalar(ld["total"])
                total_action_loss += _scalar(ld["action"])
                total_mem_loss += _scalar(ld["mem_l2"])
                total_uncert_loss += _scalar(ld["uncert"])
                num_batches += 1

                if _TQDM_AVAILABLE:
                    pbar.set_postfix({"val_loss": f"{_scalar(ld['total']):.4f}"})

        if _TQDM_AVAILABLE:
            pbar.close()

    avg_loss = total_loss / max(num_batches, 1)
    avg_action = total_action_loss / max(num_batches, 1)
    avg_mem = total_mem_loss / max(num_batches, 1)
    avg_uncert = total_uncert_loss / max(num_batches, 1)

    if writer is not None:
        writer.add_scalar("val/loss_total", avg_loss, global_step)
        writer.add_scalar("val/loss_action", avg_action, global_step)
        writer.add_scalar("val/loss_mem_l2", avg_mem, global_step)
        writer.add_scalar("val/loss_uncert", avg_uncert, global_step)

    print(
        f"  [val step {global_step}] "
        f"loss={avg_loss:.4f} "
        f"action={avg_action:.4f} "
        f"mem_l2={avg_mem:.6f} "
        f"uncert={avg_uncert:.4f}"
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze pretrained LLM weights")
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs to use (e.g. '0,1,2'). "
                             "Empty = all available GPUs. "
                             "For single-GPU training, specify one ID (e.g. '0').")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Session-chunked training: number of consecutive "
                             "steps per chunk. 1 = single-step (original). "
                             ">1 = multi-step, carries memory forward across "
                             "steps to train FiLM-GRU recurrent dynamics.")
    args = parser.parse_args()

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
    use_data_parallel = num_gpus > 1

    print(f"[train] device={device}, GPUs visible={num_gpus}, "
          f"DataParallel={'YES' if use_data_parallel else 'NO'}")
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
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        val_steps=args.val_steps,
        freeze_base=args.freeze_base,
        gradient_checkpointing=not args.no_grad_checkpoint,
        bf16=args.bf16,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    # ── Build model ──
    print(f"[train] Loading base model: {config.base_model_name}")
    model = JAMELCompactWrapper(config).to(device)

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

    train_dataset = CompactDataset(
        parquet_files=args.train_file,
        tokenizer=raw_model.tokenizer,
        processor=raw_model.processor,
        max_length=config.max_length,
        image_resize=config.image_resize,
    )
    val_dataset = CompactDataset(
        parquet_files=args.val_file,
        tokenizer=raw_model.tokenizer,
        processor=raw_model.processor,
        max_length=config.max_length,
        image_resize=config.image_resize,
    )

    if use_chunking:
        # Wrap with SessionChunkDataset for multi-step training
        train_dataset = SessionChunkDataset(train_dataset, chunk_size=config.chunk_size)
        val_dataset = SessionChunkDataset(val_dataset, chunk_size=config.chunk_size)
        # In chunked mode, each "batch" is one chunk (B=1, multiple steps)
        # batch_size=1 because each chunk already contains chunk_size steps
        chunk_batch_size = config.per_device_batch_size
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
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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

        global_step = train_one_epoch(
            model, train_loader, optimizer, config, writer,
            global_step, device, epoch,
        )
        scheduler.step()

        # Validation
        if (epoch + 1) * len(train_loader) // config.gradient_accumulation_steps >= config.val_steps:
            val_loss = validate(model, val_loader, config, writer, global_step, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = Path(config.output_dir) / "best"
                raw_model.save_pretrained(best_dir)
                print(f"  [best] New best val loss: {best_val_loss:.4f}")

    # ── Save final model ──
    final_dir = Path(config.output_dir) / "final"
    raw_model.save_pretrained(final_dir)
    print(f"\n[train] Final model saved to {final_dir}")

    if writer is not None:
        writer.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()