"""
Baseline SFT training: pure Qwen3-VL (no side memory).

Trains the pretrained Qwen3-VL model with standard supervised fine-tuning
on the same data used by JAMEL-COMPACT.  No memory modules, no chunking,
no action embeddings — just plain next-token prediction loss (cross-entropy)
on the response tokens.

This baseline lets us measure how much JAMEL-COMPACT's side memory contributes
above and beyond simple SFT of the same base model on the same data.

Usage:
    python -m jamel_compact.baseline_train \
        --train-file data/compact_sft_data/compact_train.parquet \
        --val-file data/compact_sft_data/compact_val.parquet \
        --base-model Qwen/Qwen3-VL-2B-Instruct \
        --output-dir outputs/baseline_ckpt \
        --tb-log-dir outputs/baseline_tb \
        --max-epochs 2 \
        --gpu-ids 0,1,2,3
"""
from __future__ import annotations

import os
import sys

# ── Set CUDA_VISIBLE_DEVICES BEFORE importing torch ──
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

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None

from .data import CompactDataset, collate_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap(model):
    """Get the underlying model from DataParallel wrapper."""
    while hasattr(model, "module"):
        model = model.module
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    config,
    writer,
    global_step: int,
    device: torch.device,
    epoch: int,
    accum_steps: int,
    log_steps: int,
    max_grad_norm: float,
    raw_model,
    config_save_steps: int = 0,
    config_val_steps: int = 0,
    config_output_dir: str = "",
    dataloader_val=None,
    processor=None,
    best_val_loss_tracker=None,
    scheduler=None,
):
    """Standard SFT training epoch — pure cross-entropy loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    if _TQDM_AVAILABLE:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar):
        batch_start = time.time()

        # ── Move data to device ──
        # NOTE: this block is identical to validate() — keep in sync.
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.to(device)

        # ── Pre-compute visual features on raw model before DataParallel ──
        # Same trick as JAMEL-COMPACT: process images on the main device to
        # avoid visual encoder issues under DataParallel scatter.
        # Qwen3-VL nests the visual tower under .model.visual, so check both.
        inputs_embeds = None
        has_visual = (
            hasattr(raw_model, "visual")
            or (hasattr(raw_model, "model") and hasattr(raw_model.model, "visual"))
        )
        if pixel_values is not None and has_visual:
            embed_layer = raw_model.get_input_embeddings()
            h_embed = embed_layer(input_ids)
            inputs_embeds = _inject_visual_features(
                raw_model, h_embed, input_ids, pixel_values, image_grid_thw,
            )

        # ── Forward pass ──
        if inputs_embeds is not None:
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        loss = loss / accum_steps

        # ── Backward ──
        loss.backward()

        total_loss += loss.item() * accum_steps
        num_batches += 1

        # ── Gradient accumulation ──
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if scheduler is not None:
                scheduler.step()

            # ── Logging ──
            if global_step % log_steps == 0:
                elapsed = time.time() - batch_start
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = total_loss / num_batches

                if _TB_AVAILABLE and writer is not None:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                msg = (f"  step {global_step}  loss={avg_loss:.4f}  "
                       f"lr={lr:.2e}  time={elapsed:.1f}s")
                if _TQDM_AVAILABLE:
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                else:
                    print(msg)

            # ── Mid-epoch checkpoint save ──
            if config_save_steps > 0 and global_step % config_save_steps == 0:
                ckpt_dir = Path(config_output_dir) / f"global_step_{global_step}"
                raw_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                if processor is not None:
                    try:
                        processor.save_pretrained(ckpt_dir)
                    except Exception:
                        pass
                print(f"  [checkpoint] Saved to {ckpt_dir}")

            # ── Mid-epoch validation ──
            if config_val_steps > 0 and global_step % config_val_steps == 0:
                val_loss = validate(model, dataloader_val, device, raw_model)
                if _TB_AVAILABLE and writer is not None:
                    writer.add_scalar("val/loss", val_loss, global_step)
                if val_loss < best_val_loss_tracker[0]:
                    best_val_loss_tracker[0] = val_loss
                    best_dir = Path(config_output_dir) / "best"
                    raw_model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    if processor is not None:
                        try:
                            processor.save_pretrained(best_dir)
                        except Exception:
                            pass
                    print(f"  [best] New best val loss: {val_loss:.4f}")

        # ── Periodic cache cleanup ──
        if step % 50 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(num_batches, 1)
    print(f"  [epoch {epoch}] avg loss: {avg_loss:.4f}")
    return global_step


@torch.inference_mode()
def validate(model, dataloader, device, raw_model):
    """Validation — pure CE loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    if _TQDM_AVAILABLE:
        pbar = tqdm(dataloader, desc="Validation")
    else:
        pbar = dataloader

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.to(device)

        inputs_embeds = None
        has_visual = (
            hasattr(raw_model, "visual")
            or (hasattr(raw_model, "model") and hasattr(raw_model.model, "visual"))
        )
        if pixel_values is not None and has_visual:
            embed_layer = raw_model.get_input_embeddings()
            h_embed = embed_layer(input_ids)
            inputs_embeds = _inject_visual_features(
                raw_model, h_embed, input_ids, pixel_values, image_grid_thw,
            )

        if inputs_embeds is not None:
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        loss = outputs.loss
        if loss.dim() > 0:
            loss = loss.mean()
        total_loss += loss.item()
        num_batches += 1

        if _TQDM_AVAILABLE:
            pbar.set_postfix({"val_loss": f"{total_loss / num_batches:.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    print(f"  [val] avg loss: {avg_loss:.4f}")
    return avg_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Visual feature injection (simplified — no DeepStack, no side memory)
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_visual_features(model, h, input_ids, pixel_values, image_grid_thw):
    """
    Process image through the visual encoder and inject features into the
    hidden states at image placeholder token positions.

    This is the same logic as JAMELCompactWrapper._inject_visual_features,
    but operating on the raw Qwen3-VL model without side memory.
    """
    # Flatten batch dims
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
    if image_grid_thw is not None and image_grid_thw.dim() == 3:
        image_grid_thw = image_grid_thw.reshape(-1, 3)

    # Use the model's get_image_features if available
    if hasattr(model, "get_image_features"):
        try:
            vision_output = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds_list = vision_output.pooler_output
            image_embeds = torch.cat(image_embeds_list, dim=0).to(h.device, h.dtype)
        except Exception as e:
            print(f"[baseline] get_image_features failed: {e}")
            return None
    else:
        visual = getattr(model, "visual", None)
        if visual is None:
            visual = getattr(getattr(model, "model", None), "visual", None)
        if visual is None:
            return None
        try:
            vision_output = visual(
                pixel_values.to(h.dtype),
                grid_thw=image_grid_thw,
                return_dict=True,
            )
            if hasattr(vision_output, "pooler_output"):
                image_embeds = vision_output.pooler_output
            elif hasattr(vision_output, "last_hidden_state"):
                image_embeds = vision_output.last_hidden_state
            else:
                return None
            image_embeds = image_embeds.to(h.device, h.dtype)
        except Exception as e:
            print(f"[baseline] Visual encoder failed: {e}")
            return None

    # Find image token ID
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        tokenizer = getattr(model, "_tokenizer", None)
        if tokenizer is not None:
            image_token_id = tokenizer.convert_tokens_to_ids("◣")
            if image_token_id is None or image_token_id == tokenizer.unk_token_id:
                return None
        else:
            return None

    # Scatter features at image token positions
    visual_pos_mask = (input_ids == image_token_id)
    total_img_tokens = visual_pos_mask.sum().item()
    if total_img_tokens == 0:
        return None

    if total_img_tokens == image_embeds.shape[0]:
        mask_expanded = visual_pos_mask.unsqueeze(-1).expand_as(h)
        h = h.masked_scatter(mask_expanded, image_embeds)
    else:
        offset = 0
        for b in range(h.shape[0]):
            positions = visual_pos_mask[b].nonzero(as_tuple=True)[0]
            n = len(positions)
            if n > 0 and offset + n <= image_embeds.shape[0]:
                h[b, positions] = image_embeds[offset:offset + n]
                offset += n

    return h


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Baseline SFT Training (pure Qwen3-VL)")
    parser.add_argument("--train-file", required=True, help="Train parquet file")
    parser.add_argument("--val-file", required=True, help="Val parquet file")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Pretrained base model name or path")
    parser.add_argument("--output-dir", default="outputs/baseline_ckpt")
    parser.add_argument("--tb-log-dir", default="outputs/baseline_tb")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--val-steps", type=int, default=200)
    parser.add_argument("--no-grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gpu-ids", type=str, default="",
                        help="Comma-separated GPU IDs (e.g. '0,1,2'). Empty = all.")
    parser.add_argument("--image-resize-w", type=int, default=640)
    parser.add_argument("--image-resize-h", type=int, default=360)
    args = parser.parse_args()

    # ── GPU selection ──
    if args.gpu_ids:
        print(f"[baseline] Requested GPUs: {args.gpu_ids} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    else:
        print("[baseline] GPU_IDS not set — using all visible GPUs")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = num_gpus > 1

    print(f"[baseline] device={device}, GPUs visible={num_gpus}, "
          f"DataParallel={'YES' if use_data_parallel else 'NO'}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB)")

    # ── Load model ──
    print(f"[baseline] Loading base model: {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[baseline] AutoModelForImageTextToText failed ({e}), "
              f"trying AutoModelForCausalLM...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    except Exception:
        processor = None

    # Store tokenizer reference for visual feature injection
    model._tokenizer = tokenizer

    # Gradient checkpointing (saves memory)
    if not args.no_grad_checkpoint:
        model.gradient_checkpointing_enable()
        # use_cache MUST be False when gradient checkpointing is enabled
        if hasattr(model, 'config'):
            model.config.use_cache = False
        print("[baseline] Gradient checkpointing enabled")

    model = model.to(device)

    # Wrap with DataParallel for multi-GPU
    if use_data_parallel:
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"[baseline] DataParallel active on {num_gpus} GPUs: {device_ids}")

    raw_model = _unwrap(model)

    # Report params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[baseline] Total params:     {total_params / 1e9:.2f}B")
    print(f"[baseline] Trainable params: {trainable_params / 1e9:.2f}B")

    # ── Build dataset ──
    print(f"[baseline] Loading data: {args.train_file}")
    image_resize = (args.image_resize_w, args.image_resize_h)

    train_dataset = CompactDataset(
        parquet_files=args.train_file,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        image_resize=image_resize,
    )
    val_dataset = CompactDataset(
        parquet_files=args.val_file,
        tokenizer=tokenizer,
        processor=processor,
        max_length=args.max_length,
        image_resize=image_resize,
    )

    # DataLoader — standard collate (no chunking for baseline)
    effective_batch = args.batch_size * max(num_gpus, 1)
    print(f"[baseline] Per-GPU batch size: {args.batch_size}"
          f" × {max(num_gpus, 1)} GPUs = effective batch {effective_batch}")
    pad_token_id = tokenizer.pad_token_id or 0
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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── LR scheduler (cosine with warmup) ──
    total_steps = len(train_loader) * args.max_epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── TensorBoard ──
    if _TB_AVAILABLE:
        writer = SummaryWriter(log_dir=args.tb_log_dir)
        print(f"[baseline] TensorBoard logging to {args.tb_log_dir}")
        print(f"  Run: tensorboard --logdir {args.tb_log_dir}")
    else:
        writer = None
        print("[baseline] TensorBoard not available (pip install tensorboard)")

    # ── Training loop ──
    global_step = 0
    best_val_loss = float('inf')
    # Mutable holder so train_one_epoch can update best_val_loss mid-epoch
    best_val_loss_tracker = [best_val_loss]

    for epoch in range(args.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print(f"{'='*60}")

        global_step = train_one_epoch(
            model, train_loader, optimizer, None, writer,
            global_step, device, epoch,
            accum_steps=args.grad_accum,
            log_steps=args.log_steps,
            max_grad_norm=args.max_grad_norm,
            raw_model=raw_model,
            config_save_steps=args.save_steps,
            config_val_steps=args.val_steps,
            config_output_dir=args.output_dir,
            dataloader_val=val_loader,
            processor=processor,
            best_val_loss_tracker=best_val_loss_tracker,
            scheduler=scheduler,
        )
        # Sync back from the mutable tracker
        best_val_loss = best_val_loss_tracker[0]
        # scheduler is stepped per optimizer-step inside train_one_epoch

        # Validation (end of epoch)
        val_loss = validate(model, val_loader, device, raw_model)
        if _TB_AVAILABLE and writer is not None:
            writer.add_scalar("val/loss", val_loss, global_step)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = Path(args.output_dir) / "best"
            raw_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            if processor is not None:
                try:
                    processor.save_pretrained(best_dir)
                except Exception:
                    pass
            print(f"  [best] New best val loss: {best_val_loss:.4f}")

        # Save checkpoint per epoch
        ckpt_dir = Path(args.output_dir) / f"epoch{epoch}"
        raw_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        if processor is not None:
            try:
                processor.save_pretrained(ckpt_dir)
            except Exception:
                pass
        print(f"  [checkpoint] Saved to {ckpt_dir}")

    # ── Save final model ──
    final_dir = Path(args.output_dir) / "final"
    raw_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if processor is not None:
        try:
            processor.save_pretrained(final_dir)
        except Exception:
            pass
    print(f"\n[baseline] Final model saved to {final_dir}")

    if writer is not None:
        writer.close()
    print("[baseline] Done.")


if __name__ == "__main__":
    main()
