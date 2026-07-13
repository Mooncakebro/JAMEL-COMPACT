"""
Data preparation for JAMEL-COMPACT.

Converts augmented browser-session parquet into a dataset suitable for
training the JAMEL-COMPACT model.  Unlike the original JAMEL which
pre-compresses screenshots into memory tokens offline, COMPACT learns
memory online during the forward pass — so data prep is simpler:

  • Each row = one step in a session
  • Input: prompt text + screenshot image + previous action
  • Label: the action the agent took (response text)
  • Memory is NOT pre-computed; it evolves during training

The dataset yields:
  - input_ids:      tokenized prompt (text + <image> placeholder)
  - attention_mask
  - pixel_values:   screenshot image (processed by the VLM processor)
  - action_input:   embedding of the previous action (for FiLM-GRU)
  - labels:         tokenized response (action)
  - session_id, step_idx: for session-level memory continuity
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _decode_png(image_bytes: bytes | None) -> Image.Image | None:
    if image_bytes is None:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None


def _get_screenshot(row) -> bytes | None:
    v = row.get("before_screenshot")
    if v is not None:
        return v
    return row.get("screenshot")


def _build_action_embedding_input(action_str: str, tokenizer) -> torch.Tensor:
    """
    Tokenize the action string and return its embedding-ready representation.
    For COMPACT, we use the token IDs of the action text as the control variable.
    The model's action_embed layer will project these into the hidden dimension.
    """
    # Simple: tokenize action text and take mean of token embeddings
    # The actual projection happens in the model's action_embed layer
    tokens = tokenizer.encode(action_str, add_special_tokens=False, max_length=32, truncation=True)
    if not tokens:
        tokens = [tokenizer.pad_token_id or 0]
    return torch.tensor(tokens, dtype=torch.long)


class CompactDataset(Dataset):
    """
    Dataset for JAMEL-COMPACT training.

    Each item contains:
      - input_ids:      [N] — tokenized prompt (with <image> placeholder)
      - attention_mask: [N]
      - pixel_values:   processed image tensor (or None)
      - action_input:   [d] — action embedding input (previous action)
      - labels:         [N] — tokenized full sequence (prompt masked with -100)
      - session_id:     str
      - step_idx:       int

    Memory continuity is handled by the training loop, which groups samples
    by session_id and passes memory states forward.
    """

    def __init__(
        self,
        parquet_files: str | List[str],
        tokenizer,
        processor=None,
        max_length: int = 8192,
        image_resize: tuple = (640, 360),
        prompt_key: str = "prompt",
        response_key: str = "response",
        image_key: str = "current_image_png_bytes",
        action_key: str = "action",
    ):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_resize = image_resize
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.image_key = image_key
        self.action_key = action_key

        self._read_files()
        self._validate_and_filter()

    def _read_files(self):
        frames = [pd.read_parquet(p) for p in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)

    def _validate_and_filter(self):
        """Filter out samples that exceed max_length after tokenization."""
        total = len(self.dataframe)
        valid_indices = []

        for i in range(total):
            row = self.dataframe.iloc[i]
            prompt = str(row.get(self.prompt_key, ""))
            response = str(row.get(self.response_key, ""))

            try:
                prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                response_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
            except Exception:
                continue

            # Estimate total (prompt + response + chat template overhead)
            estimated = prompt_tokens + response_tokens + 200
            if estimated <= self.max_length:
                valid_indices.append(i)

        self._valid_indices = valid_indices
        discarded = total - len(valid_indices)
        if discarded > 0:
            print(f"[data] Discarded {discarded}/{total} samples exceeding max_length={self.max_length}")
        print(f"[data] {len(valid_indices)}/{total} samples retained")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, index: int) -> dict:
        row_idx = self._valid_indices[index]
        row = self.dataframe.iloc[row_idx]

        prompt = str(row.get(self.prompt_key, ""))
        response = str(row.get(self.response_key, ""))
        action = str(row.get(self.action_key, ""))

        # ── Build chat messages ──
        has_image = "<image>" in prompt
        image = None
        if has_image:
            raw = row.get(self.image_key)
            if raw is not None:
                image = _decode_png(raw)
                if image is not None and image.size != self.image_resize:
                    image = image.resize(self.image_resize, Image.BILINEAR)

        # Split prompt on <image> tag for multimodal content
        segments = prompt.split("<image>")
        content = []
        for idx, seg in enumerate(segments):
            if seg:
                content.append({"type": "text", "text": seg})
            if idx < len(segments) - 1:
                content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]

        # ── Tokenize ──
        if self.processor is not None and has_image and image is not None:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            full_messages = messages + [{"role": "assistant", "content": [{"type": "text", "text": response}]}]
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_inputs = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
            full_inputs = self.processor(text=[full_text], images=[image], return_tensors="pt")
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            full_messages = messages + [{"role": "assistant", "content": response}]
            full_text = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False,
            )
            prompt_inputs = {"input_ids": self.tokenizer.encode(prompt_text, return_tensors="pt")}
            full_inputs = {"input_ids": self.tokenizer.encode(full_text, return_tensors="pt")}

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))[0]
        prompt_length = prompt_inputs["input_ids"].shape[-1]

        # ── Truncate if needed ──
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # ── Labels: mask prompt with -100 ──
        labels = input_ids.clone()
        labels[:prompt_length] = -100

        # ── Action embedding input ──
        action_tokens = self.tokenizer.encode(
            action, add_special_tokens=False, max_length=32, truncation=True,
        )
        if not action_tokens:
            action_tokens = [self.tokenizer.pad_token_id or 0]
        action_input_ids = torch.tensor(action_tokens, dtype=torch.long)

        # ── Extract pixel values if available ──
        pixel_values = None
        if "pixel_values" in full_inputs:
            pixel_values = full_inputs["pixel_values"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "action_input_ids": action_input_ids,
            "pixel_values": pixel_values,
            "session_id": str(row.get("session_id", "")),
            "step_idx": int(row.get("step_idx", 0)),
            "target_app": str(row.get("target_app", "")),
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate function for CompactDataset. Pads sequences to max length in batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    max_action_len = max(item["action_input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_masks = []
    labels = []
    action_inputs = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
        ]))
        attention_masks.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))

        act_len = item["action_input_ids"].shape[0]
        act_pad = max_action_len - act_len
        action_inputs.append(torch.cat([
            item["action_input_ids"],
            torch.full((act_pad,), pad_token_id, dtype=torch.long),
        ]))

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
        "action_input_ids": torch.stack(action_inputs),
        "session_ids": [item["session_id"] for item in batch],
        "step_indices": [item["step_idx"] for item in batch],
    }

    # Pixel values (may be None for text-only)
    pixel_values = [item.get("pixel_values") for item in batch]
    if all(pv is not None for pv in pixel_values):
        result["pixel_values"] = torch.stack(pixel_values)

    return result


def prepare_compact_dataset(
    input_files: str | list[str],
    output_dir: str,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[str, str]:
    """
    Prepare SFT data for JAMEL-COMPACT training.

    Unlike original JAMEL, no memory compression is needed — the model
    learns memory online.  We just split into train/val parquet files.

    Args:
        input_files: augmented session-schema parquet file(s)
        output_dir:  directory to write train/val parquet
        val_ratio:   fraction of data for validation
        seed:        random seed for shuffling

    Returns:
        (train_path, val_path)
    """
    if not isinstance(input_files, list):
        input_files = [input_files]

    frames = [pd.read_parquet(p) for p in input_files]
    df = pd.concat(frames, ignore_index=True)

    print(f"[data] Loaded {len(df)} rows from {len(input_files)} files")
    print(f"  sessions: {df.get('session_id', pd.Series()).nunique() if 'session_id' in df.columns else 'N/A'}")
    print(f"  apps:     {df.get('target_app', pd.Series()).nunique() if 'target_app' in df.columns else 'N/A'}")

    # Shuffle and split
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if val_ratio <= 0:
        train_df = df.reset_index(drop=True)
        val_df = train_df.iloc[:1].copy()
    else:
        split_idx = max(1, int(len(df) * (1.0 - val_ratio)))
        train_df = df.iloc[indices[:split_idx]].reset_index(drop=True)
        val_df = df.iloc[indices[split_idx:]].reset_index(drop=True)
        if len(val_df) == 0:
            val_df = train_df.iloc[:1].copy()

    train_path = output_path / "compact_train.parquet"
    val_path = output_path / "compact_val.parquet"
    train_df.to_parquet(train_path, row_group_size=4000)
    val_df.to_parquet(val_path, row_group_size=4000)

    print(f"[data] Train: {train_path} ({len(train_df)} rows)")
    print(f"[data] Val:   {val_path} ({len(val_df)} rows)")

    return str(train_path), str(val_path)