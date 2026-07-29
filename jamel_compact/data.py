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
from PIL import Image

# torch is only needed for CompactDataset and collate_fn, not for
# prepare_compact_dataset.  Import lazily to allow data prep without torch.
try:
    import torch
    from torch.utils.data import Dataset, Sampler
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    Dataset = object  # fallback base class
    Sampler = object


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
        image_key: str = "before_screenshot",  # matches ExplorerSFT-ReAct parquet
        action_key: str = "action",
        coverage_weight_eta: float = 0.0,
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
        self._coverage_weight_eta = coverage_weight_eta  # F7: coverage-weighted SFT

        self._read_files()
        self._validate_and_filter()

    def _read_files(self):
        # Only load columns needed for training to save memory
        needed_cols = [self.prompt_key, self.response_key, self.image_key,
                      self.action_key, "session_id", "episode_idx", "step_idx",
                      "session_step_idx", "target_app", "coverage_delta_score"]
        frames = []
        for p in self.parquet_files:
            try:
                df_p = pd.read_parquet(p, columns=needed_cols)
            except (ValueError, KeyError):
                df_p = pd.read_parquet(p)
            frames.append(df_p)
        self.dataframe = pd.concat(frames, ignore_index=True)
        self._build_previous_actions()

    def _build_previous_actions(self):
        """Map each row to the action taken at the preceding session step."""
        self._previous_actions = {}
        if "session_id" not in self.dataframe.columns:
            return

        order_col = (
            "session_step_idx"
            if "session_step_idx" in self.dataframe.columns
            else "step_idx"
        )
        for _, group in self.dataframe.groupby("session_id", sort=False):
            ordered = group.sort_values(order_col, kind="stable")
            previous_action = "noop()"
            for row_idx, row in ordered.iterrows():
                self._previous_actions[row_idx] = previous_action
                current_action = row.get(self.action_key, "")
                previous_action = "" if pd.isna(current_action) else str(current_action)

    def _validate_and_filter(self):
        """Filter out samples that exceed max_length.

        Uses fast char-based estimation (chars/3 ≈ tokens for English) instead
        of tokenizing every row, which would be very slow for large datasets.
        Also reads coverage_delta_score for F7 coverage-weighted SFT.
        """
        total = len(self.dataframe)
        valid_indices = []

        # Char-based estimation: ~3 chars per token for English text
        # Image tokens: ~1175 for Qwen2.5-VL at 640×360
        CHAT_TEMPLATE_OVERHEAD = 200
        IMAGE_TOKEN_OVERHEAD = 1175

        # Check if coverage_delta_score column exists (F7)
        self._has_coverage = "coverage_delta_score" in self.dataframe.columns

        for i in range(total):
            row = self.dataframe.iloc[i]
            prompt = str(row.get(self.prompt_key, ""))
            response = str(row.get(self.response_key, ""))

            prompt_chars = len(prompt)
            response_chars = len(response)
            has_image = "<image>" in prompt

            # Estimate tokens: chars/3 + image overhead + template overhead
            estimated = (prompt_chars + response_chars) // 3 + CHAT_TEMPLATE_OVERHEAD
            if has_image:
                estimated += IMAGE_TOKEN_OVERHEAD

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
        action = self._previous_actions.get(row_idx, "")

        # ── Build chat messages ──
        # Always load the screenshot if available, even if prompt has no <image> tag.
        # The ExplorerSFT-ReAct dataset's prompt column may not contain <image>,
        # but the before_screenshot column has the image bytes.
        raw_screenshot = row.get(self.image_key)
        image = None
        if raw_screenshot is not None:
            image = _decode_png(raw_screenshot)
            if image is not None and image.size != self.image_resize:
                image = image.resize(self.image_resize, Image.BILINEAR)

        has_image = image is not None

        # Build content: if prompt has <image> tag, split on it; otherwise
        # append the image at the end before the response instruction.
        if "<image>" in prompt:
            segments = prompt.split("<image>")
            content = []
            for idx, seg in enumerate(segments):
                if seg:
                    content.append({"type": "text", "text": seg})
                if idx < len(segments) - 1:
                    content.append({"type": "image"})
        elif has_image:
            # No <image> tag in prompt — append image at the end
            content = [{"type": "text", "text": prompt}, {"type": "image"}]
        else:
            content = [{"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": content}]

        # ── Tokenize ──
        if self.processor is not None and has_image:
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
        mm_token_type_ids = full_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[0]
        prompt_length = prompt_inputs["input_ids"].shape[-1]

        # ── Truncate if needed ──
        # F2: Left-truncate the PROMPT (keep the full response), matching
        # eval.py which also truncates from the left.  Right-truncation
        # (v1) can cut the response — the supervised signal — losing labels.
        full_len = input_ids.shape[0]
        if full_len > self.max_length:
            excess = full_len - self.max_length
            # Only truncate from the prompt portion; if the response alone
            # exceeds max_length, the sample should be dropped.
            if excess >= prompt_length:
                # Response alone doesn't fit — mark for dropping
                return None
            # Left-truncate prompt by `excess` tokens, keep all response tokens
            input_ids = input_ids[excess:]
            attention_mask = attention_mask[excess:]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[excess:]
            prompt_length = prompt_length - excess
            # F2: If the surviving prompt is shorter than 64 tokens, drop
            # the sample — too little context for meaningful memory updates.
            if prompt_length < 64:
                return None

        # ── Labels: mask prompt with -100 ──
        # Clamp prompt_length to max_length - 1 so that at least 1 token
        # has a valid label (otherwise all-−100 → CrossEntropyLoss returns
        # NaN, which corrupts all model weights in one step).
        effective_prompt_len = min(prompt_length, input_ids.shape[0] - 1)
        # F2: Assert at least one non-(-100) label exists
        labels = input_ids.clone()
        labels[:effective_prompt_len] = -100
        assert (labels != -100).any(), \
            f"All labels are -100 after truncation (prompt_len={prompt_length}, " \
            f"seq_len={input_ids.shape[0]})"

        # Only the prompt/current observation exists at inference time. The
        # response tokens are teacher-forced targets and must not enter the
        # memory observation pool.
        observation_mask = torch.zeros_like(attention_mask)
        observation_mask[:effective_prompt_len] = 1

        # ── Action embedding input ──
        action_tokens = self.tokenizer.encode(
            action, add_special_tokens=False, max_length=32, truncation=True,
        )
        has_previous_action = bool(action_tokens)
        if not action_tokens:
            action_tokens = [self.tokenizer.pad_token_id or 0]
        action_input_ids = torch.tensor(action_tokens, dtype=torch.long)
        action_attention_mask = torch.full(
            action_input_ids.shape,
            1 if has_previous_action else 0,
            dtype=torch.long,
        )

        # ── Extract pixel values if available ──
        # Qwen3-VL processor returns pixel_values as [num_patches, patch_dim]
        # (2D, no batch dim) and image_grid_thw as [num_images, 3] (2D).
        # We keep the full tensors — do NOT index [0] (that would take only
        # the first patch, not the first image).
        pixel_values = None
        image_grid_thw = None
        if "pixel_values" in full_inputs:
            pixel_values = full_inputs["pixel_values"]
            # Squeeze batch dim if processor returned 3D [1, num_patches, dim]
            if pixel_values.dim() == 3 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values.squeeze(0)
        if "image_grid_thw" in full_inputs:
            image_grid_thw = full_inputs["image_grid_thw"]
            # Squeeze batch dim if processor returned 3D [1, num_images, 3]
            if image_grid_thw.dim() == 3 and image_grid_thw.shape[0] == 1:
                image_grid_thw = image_grid_thw.squeeze(0)

        # ── F7: Coverage-weighted sample weight ──
        sample_weight = 1.0
        if self._has_coverage:
            cov_delta = float(row.get("coverage_delta_score", 0) or 0)
            sample_weight = 1.0 + self._coverage_weight_eta * max(cov_delta, 0.0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "observation_mask": observation_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "labels": labels,
            "action_input_ids": action_input_ids,
            "action_attention_mask": action_attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "session_id": str(row.get("session_id", "")),
            "step_idx": int(row.get("step_idx", 0)),
            "target_app": str(row.get("target_app", "")),
            "sample_weight": sample_weight,
        }


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate function for CompactDataset. Pads sequences to max length in batch.

    F2: Filters out None samples (dropped during truncation) and logs drop count.
    F7: Carries sample_weight for coverage-weighted SFT.
    """
    # F2: Filter out None samples (response didn't fit in max_length)
    valid_batch = [item for item in batch if item is not None]
    dropped = len(batch) - len(valid_batch)
    if dropped > 0:
        # Don't print here (would spam logs); caller can check batch size
        pass
    if not valid_batch:
        # Return an empty batch marker — caller should skip
        return None
    batch = valid_batch

    max_len = max(item["input_ids"].shape[0] for item in batch)
    max_action_len = max(item["action_input_ids"].shape[0] for item in batch)

    input_ids = []
    attention_masks = []
    observation_masks = []
    mm_token_type_ids_list = []
    labels = []
    action_inputs = []
    action_attention_masks = []
    sample_weights = []

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
        observation_masks.append(torch.cat([
            item["observation_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        mm_ids = item.get("mm_token_type_ids")
        if mm_ids is not None:
            mm_ids = torch.cat([mm_ids, torch.zeros(pad_len, dtype=torch.long)])
        mm_token_type_ids_list.append(mm_ids)
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
        action_attention_masks.append(torch.cat([
            item["action_attention_mask"],
            torch.zeros(act_pad, dtype=torch.long),
        ]))
        sample_weights.append(item.get("sample_weight", 1.0))

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "observation_mask": torch.stack(observation_masks),
        "labels": torch.stack(labels),
        "action_input_ids": torch.stack(action_inputs),
        "action_attention_mask": torch.stack(action_attention_masks),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "session_ids": [item["session_id"] for item in batch],
        "step_indices": [item["step_idx"] for item in batch],
    }
    if all(ids is not None for ids in mm_token_type_ids_list):
        result["mm_token_type_ids"] = torch.stack(mm_token_type_ids_list)

    # Pixel values: concatenate all patches along dim 0.
    # Each sample is [num_patches, patch_dim] (2D). Concatenating gives
    # [total_patches, patch_dim] which is what the visual encoder expects.
    pixel_values = [item.get("pixel_values") for item in batch]
    if all(pv is not None for pv in pixel_values):
        try:
            result["pixel_values"] = torch.cat(pixel_values, dim=0)
        except RuntimeError:
            # Different shapes — return as list
            result["pixel_values"] = pixel_values

    # image_grid_thw: concatenate along dim 0.
    # Each sample is [num_images, 3] (e.g. [1, 3]). Concatenating gives
    # [total_images, 3] which is what the visual encoder expects.
    grid_thws = [item.get("image_grid_thw") for item in batch]
    if all(g is not None for g in grid_thws):
        try:
            result["image_grid_thw"] = torch.cat(grid_thws, dim=0)
        except RuntimeError:
            result["image_grid_thw"] = grid_thws

    return result


def session_collate_fn(batch, pad_token_id: int = 0) -> dict:
    """Collate a chunk of consecutive steps from one session.

    Each step in the chunk is a dict from CompactDataset.__getitem__.
    Steps are processed sequentially with batch size one, so each step keeps
    its own token length instead of being padded to the longest step.

    Returns a dict where each value is a list of per-step tensors (for
    sequence-level processing in the training loop).

    Note: Recurrent chunk training requires DataLoader(batch_size=1). The
    DataLoader wraps the list-of-dicts from SessionChunkDataset.__getitem__
    in another list, so we receive [[dict, dict, ...]].
    """
    if not batch:
        return None

    if isinstance(batch[0], dict):
        chunk = batch
    elif len(batch) == 1 and isinstance(batch[0], list):
        chunk = batch[0]
    else:
        raise ValueError(
            "Session-chunked training requires DataLoader batch_size=1; "
            "use gradient accumulation to increase the effective batch size."
        )

    # F2: Filter out None samples (dropped during truncation)
    chunk = [item for item in chunk if item is not None]
    if not chunk:
        return None

    input_ids_list = []
    attention_masks_list = []
    observation_masks_list = []
    mm_token_type_ids_list = []
    labels_list = []
    action_inputs_list = []
    action_attention_masks_list = []
    pixel_values_list = []
    grid_thws_list = []
    sample_weights_list = []
    step_valids = []

    for item in chunk:
        input_ids_list.append(item["input_ids"])
        attention_masks_list.append(item["attention_mask"])
        observation_masks_list.append(item["observation_mask"])
        mm_token_type_ids_list.append(item.get("mm_token_type_ids"))
        labels_list.append(item["labels"])
        action_inputs_list.append(item["action_input_ids"])
        action_attention_masks_list.append(item["action_attention_mask"])

        pv = item.get("pixel_values")
        pixel_values_list.append(pv)
        gt = item.get("image_grid_thw")
        grid_thws_list.append(gt)
        sample_weights_list.append(item.get("sample_weight", 1.0))
        step_valids.append(bool(item.get("_chunk_step_valid", True)))

    # Stack per-step tensors: each becomes [1, N] (single sample per step)
    # so the training loop can process one step at a time with B=1
    result = {
        "input_ids": [ids.unsqueeze(0) for ids in input_ids_list],
        "attention_mask": [am.unsqueeze(0) for am in attention_masks_list],
        "observation_mask": [om.unsqueeze(0) for om in observation_masks_list],
        "mm_token_type_ids": [
            ids.unsqueeze(0) if ids is not None else None
            for ids in mm_token_type_ids_list
        ],
        "labels": [lab.unsqueeze(0) for lab in labels_list],
        "action_input_ids": [ai.unsqueeze(0) for ai in action_inputs_list],
        "action_attention_mask": [am.unsqueeze(0) for am in action_attention_masks_list],
        "pixel_values": pixel_values_list,  # list of [num_patches, dim] or None
        "image_grid_thw": grid_thws_list,    # list of [num_images, 3] or None
        "sample_weights": torch.tensor(sample_weights_list, dtype=torch.float32),
        "session_ids": [item["session_id"] for item in chunk],
        "step_indices": [item["step_idx"] for item in chunk],
        "step_valids": step_valids,
        "chunk_size": len(chunk),
    }

    return result


class SessionChunkDataset(Dataset):
    """
    Dataset that groups consecutive steps from the same session into chunks.

    This enables sequence-level training where the side memory (FiLM-GRU +
    Kalman filter) evolves across multiple steps, training the recurrent
    dynamics that the model needs at inference time.

    Each item is a list of consecutive step dicts from the same session.
    The chunk_size parameter controls how many steps per chunk.
    """

    def __init__(
        self,
        base_dataset_or_files,
        chunk_size: int = 4,
        tokenizer=None,
        processor=None,
        max_length: int = 8192,
        image_resize: tuple = (640, 360),
        prompt_key: str = "prompt",
        response_key: str = "response",
        image_key: str = "before_screenshot",
        action_key: str = "action",
        coverage_weight_eta: float = 0.0,
        pad_to_chunk_size: bool = False,
        pad_dropped_steps: bool = False,
    ):
        self.chunk_size = chunk_size
        self.pad_to_chunk_size = pad_to_chunk_size
        self.pad_dropped_steps = pad_dropped_steps
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.image_key = image_key
        self.action_key = action_key

        # Accept either an already-built CompactDataset (avoids re-reading
        # parquet files) or parquet file paths to build a new one.
        if isinstance(base_dataset_or_files, CompactDataset):
            self._base = base_dataset_or_files
        else:
            parquet_files = base_dataset_or_files
            if not isinstance(parquet_files, list):
                parquet_files = [parquet_files]
            self.parquet_files = list(parquet_files)
            self.tokenizer = tokenizer
            self.processor = processor
            self.max_length = max_length
            self.image_resize = image_resize
            self._base = CompactDataset(
                parquet_files=parquet_files,
                tokenizer=tokenizer,
                processor=processor,
                max_length=max_length,
                image_resize=image_resize,
                prompt_key=prompt_key,
                response_key=response_key,
                image_key=image_key,
                action_key=action_key,
            )
        # F7: propagate coverage weight eta to the base dataset
        self._base._coverage_weight_eta = coverage_weight_eta
        self.dataframe = self._base.dataframe
        self._build_chunks()

    def _build_chunks(self):
        """Group consecutive steps from the same session into chunks."""
        df = self.dataframe
        valid_indices = self._base._valid_indices

        # Group by session_id. session_step_idx is monotonic across reset-local
        # episodes; step_idx restarts at every reset and must not be used here.
        session_groups = {}
        for dataset_idx, row_idx in enumerate(valid_indices):
            row = df.iloc[row_idx]
            sid = str(row.get("session_id", ""))
            order_idx = int(row.get("session_step_idx", row.get("step_idx", 0)))
            if sid not in session_groups:
                session_groups[sid] = []
            # Store CompactDataset positions, not raw dataframe indices.
            session_groups[sid].append((order_idx, dataset_idx))

        # Sort by the continuous session index and split at missing steps so a
        # dropped sample cannot silently bridge unrelated observations.
        self._chunks = []
        for sid, steps in session_groups.items():
            steps.sort(key=lambda x: x[0])
            runs = []
            run = []
            previous_order = None
            for order_idx, row_idx in steps:
                if previous_order is not None and order_idx != previous_order + 1:
                    if run:
                        runs.append(run)
                    run = []
                run.append((order_idx, row_idx))
                previous_order = order_idx
            if run:
                runs.append(run)

            for run in runs:
                for i in range(0, len(run), self.chunk_size):
                    chunk = run[i:i + self.chunk_size]
                    if chunk:
                        self._chunks.append([idx for _, idx in chunk])

        # Shuffle chunk order (DataLoader will also shuffle, but this helps
        # when shuffle=True samples are drawn)
        random.shuffle(self._chunks)

        total_steps = sum(len(c) for c in self._chunks)
        print(f"[data] SessionChunkDataset: {len(self._chunks)} chunks "
              f"(chunk_size={self.chunk_size}), {total_steps} total steps, "
              f"avg {total_steps / max(len(self._chunks), 1):.1f} steps/chunk")

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> list[dict]:
        """Return a list of step dicts for one chunk."""
        chunk_indices = self._chunks[index]
        items = []
        for item_index in chunk_indices:
            item = self._base[item_index]
            if item is not None:
                item = dict(item)
                item["_chunk_step_valid"] = True
                items.append(item)

        target_length = None
        if self.pad_to_chunk_size:
            target_length = self.chunk_size
        elif self.pad_dropped_steps:
            target_length = len(chunk_indices)

        if target_length is not None:
            if not items:
                raise RuntimeError(
                    "Every sample in a distributed session chunk was dropped "
                    "during tokenization. Reduce max_length filtering or "
                    "remove the invalid session."
                )
            while len(items) < target_length:
                padded_item = dict(items[-1])
                padded_item["_chunk_step_valid"] = False
                items.append(padded_item)

        return items


class SynchronizedChunkDistributedSampler(Sampler):
    """Assign equal-length recurrent chunks to all ranks per iteration."""

    def __init__(
        self,
        dataset: SessionChunkDataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _groups(self) -> list[list[int]]:
        buckets = {}
        for index, chunk in enumerate(self.dataset._chunks):
            buckets.setdefault(len(chunk), []).append(index)

        rng = random.Random(self.seed + self.epoch)
        groups = []
        for chunk_length in sorted(buckets):
            indices = list(buckets[chunk_length])
            if self.shuffle:
                rng.shuffle(indices)
            group_count = (
                len(indices) + self.num_replicas - 1
            ) // self.num_replicas
            padded_count = group_count * self.num_replicas
            padded_indices = [
                indices[position % len(indices)]
                for position in range(padded_count)
            ]
            groups.extend([
                padded_indices[start:start + self.num_replicas]
                for start in range(0, padded_count, self.num_replicas)
            ])

        if self.shuffle:
            rng.shuffle(groups)
        return groups

    def __iter__(self):
        return iter(group[self.rank] for group in self._groups())

    def __len__(self) -> int:
        bucket_counts = {}
        for chunk in self.dataset._chunks:
            bucket_counts[len(chunk)] = bucket_counts.get(len(chunk), 0) + 1
        return sum(
            (count + self.num_replicas - 1) // self.num_replicas
            for count in bucket_counts.values()
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def discover_parquet_files(input_path: str | list[str]) -> list[str]:
    """
    Discover all trajectory.parquet files from the input path.

    Supports three input formats:
      1. A single parquet file:  "data/weibo/trajectory.parquet"
      2. A list of parquet files: ["data/weibo/trajectory.parquet", ...]
      3. A directory containing app subdirectories (ExplorerSFT-ReAct layout):
         "data/ExplorerSFT-ReAct_Dataset/data/react-vision"
         → discovers data/ExplorerSFT-ReAct_Dataset/data/react-vision/*/trajectory.parquet

    The ExplorerSFT-ReAct dataset has this structure:
      data/
      ├── react-text/
      │   ├── weibo/
      │   │   ├── trajectory.parquet   (150 rows)
      │   │   └── summary.json
      │   ├── alibaba/
      │   │   ├── trajectory.parquet
      │   │   └── summary.json
      │   └── ... (80 apps)
      └── react-vision/
          ├── weibo/
          │   ├── trajectory.parquet   (150 rows, includes screenshots)
          │   └── summary.json
          └── ... (80 apps)

    Each trajectory.parquet has ~150 rows (one per step in a 150-step session).
    Total: 80 apps × 150 steps × 2 variants = ~24,000 rows.
    """
    if isinstance(input_path, list):
        # Already a list of files
        return [f for f in input_path if f.endswith(".parquet")]

    p = Path(input_path)
    if p.is_file() and p.suffix == ".parquet":
        return [str(p)]

    if p.is_dir():
        # Directory: discover all trajectory.parquet under app subdirectories
        parquets = sorted(p.glob("*/trajectory.parquet"))
        if not parquets:
            # Try deeper: variant/app/trajectory.parquet
            parquets = sorted(p.glob("*/*/trajectory.parquet"))
        if not parquets:
            # Try any .parquet
            parquets = sorted(p.glob("**/*.parquet"))
        return [str(f) for f in parquets]

    return [str(p)]


def _apply_prompt_rebuild_inplace(df_chunk, source_path, chunk_size: int) -> None:
    """
    Rebuild ``prompt`` and strip ``<think>`` from ``response`` for every row
    in a chunk DataFrame, IN-PLACE.  Uses the canonical ``build_web_prompt``
    from ``jamel.train.memory.web_prompt``.

    This matches what original JAMEL's ``prepare_sft_dataset.py`` does.

    All imports are done inside this function to avoid triggering heavy
    dependencies (gymnasium, torch) during import time.
    """
    # ── Inline imports to avoid gymnasium dependency chain ──
    import re as _re

    # --- strip_think (inline from web_prompt.py) ---
    _STRIP_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)

    # --- extract_axtree_from_observation_str (inline from web_prompt.py) ---
    def _extract_axtree(obs_str: str) -> str:
        marker = "Current Observation:"
        idx = obs_str.rfind(marker)
        if idx == -1:
            return obs_str.strip()
        return obs_str[idx + len(marker):].strip()

    # --- prune_axtree (inline from jamel/core/env/web/axtree_utils.py) ---
    # Keeps all lines with interactive bids + structural ancestors.
    # Falls back to interactive-element-only if still too long.
    _IE_PATTERN = _re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")
    _STRUCTURAL = {
        "heading", "labeltext", "banner", "navigation", "main", "region",
        "complementary", "contentinfo", "search", "article", "section",
        "list", "listitem", "table", "rowgroup", "row", "gridcell",
        "menu", "menubar", "menuitem", "toolbar", "tablist", "tab",
        "tabpanel", "dialog", "alert", "form", "group",
    }

    def _prune_axtree(text: str, max_chars: int = 8000) -> str:
        if not text or len(text) <= max_chars:
            return text
        lines = text.split("\n")
        n = len(lines)
        if n <= 1:
            return text
        parsed = []
        for line in lines:
            stripped = line.lstrip("\t")
            indent = len(line) - len(stripped)
            m = _IE_PATTERN.search(stripped)
            parsed.append({
                "indent": indent, "text": stripped,
                "has_bid": m is not None,
                "role": m.group(2).lower() if m else None,
            })
        keep = [False] * n
        if n > 0:
            keep[0] = True
        for i, p in enumerate(parsed):
            if not p["has_bid"]:
                continue
            keep[i] = True
            target_indent = p["indent"] - 1
            for j in range(i - 1, -1, -1):
                if parsed[j]["indent"] == target_indent:
                    role = parsed[j]["role"]
                    if role and role in _STRUCTURAL:
                        keep[j] = True
                    elif parsed[j]["indent"] <= 1:
                        keep[j] = True
                    target_indent -= 1
                    if target_indent < 0:
                        break
        result = "\n".join(lines[i] for i in range(n) if keep[i])
        if len(result) <= max_chars:
            return result
        # Fallback: interactive lines only
        bid_result = "\n".join(lines[i] for i in range(n) if keep[i] and parsed[i]["has_bid"])
        return bid_result[:max_chars]

    # --- build_web_prompt (inline from web_prompt.py) ---
    _WEB_ACTION_SPACE = """\
noop(wait_ms: float = 1000)
send_msg_to_user(text: str)
report_infeasible(reason: str)
scroll(delta_x: float, delta_y: float)
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
hover(bid: str)
press(bid: str, key_comb: str)
focus(bid: str)
clear(bid: str)
drag_and_drop(from_bid: str, to_bid: str)
upload_file(bid: str, file: str | list[str])
tab_close()
tab_focus(index: int)
new_tab()
go_back()
go_forward()
goto(url: str)
reset()"""

    _WEB_PROMPT_TEMPLATE = """\
You are an autonomous browser exploration agent.

This is session step {step_idx}.
Your goal is to explore the target app and maximize novel JavaScript execution coverage.

Target app: {target_app}
Start URL: {start_url}

Browser action space:
{action_space}

Current open pages URLs:
{open_urls}

Current Observation:
{pruned_axtree}

Current valid interactive element ids:
{element_ids}

The current webpage screenshot is:
<image>

Respond with exactly:
<action>one action</action>

The <action> content must be one single BrowserGym action call. If the action
uses a bid, use one exact id from the valid interactive element list. Never
invent bids and never combine two actions in one response."""

    _BID_LINE_RE = _re.compile(r"\[(\d+)\]\s+([A-Za-z]+)\s+'([^']*)'")

    def _format_open_urls(urls: object) -> str:
        if isinstance(urls, str):
            s = urls.strip()
            if s.startswith("(") and s.endswith(")"):
                return s
            return repr((s,))
        try:
            items = tuple(str(u) for u in urls)
            return repr(items)
        except TypeError:
            return repr((str(urls),))

    def _extract_element_ids(pruned: str, limit: int = 200) -> str:
        seen = set()
        lines = []
        for raw in pruned.splitlines():
            m = _BID_LINE_RE.search(raw)
            if not m:
                continue
            bid, role, label = m.groups()
            if bid in seen:
                continue
            seen.add(bid)
            lines.append(f"- {bid}: {role.lower()} {label.strip()!r}")
            if len(lines) >= limit:
                lines.append(f"- ... {limit}+ ids omitted")
                break
        return "\n".join(lines) if lines else "(none)"

    def _build_prompt(*, step_idx, target_app, start_url, open_urls,
                      pruned_axtree, element_ids=None):
        if element_ids is None:
            element_ids = _extract_element_ids(pruned_axtree)
        return _WEB_PROMPT_TEMPLATE.format(
            step_idx=int(step_idx),
            target_app=str(target_app),
            start_url=str(start_url),
            action_space=_WEB_ACTION_SPACE,
            open_urls=_format_open_urls(open_urls),
            pruned_axtree=pruned_axtree,
            element_ids=element_ids,
        )

    # ── Apply to each row ──
    for i in range(chunk_size):
        row = df_chunk.iloc[i]
        try:
            obs_str = str(row.get("before_observation_str", "") or "")
            axtree_raw = _extract_axtree(obs_str)
            pruned = _prune_axtree(axtree_raw, max_chars=8000)
            target_app = str(row.get("target_app", ""))
            start_url = str(row.get("start_url", row.get("target_url", "")))
            step_idx = int(row.get("session_step_idx", row.get("step_idx", 0)))
            open_urls = row.get("before_open_pages_urls", (start_url,))

            new_prompt = _build_prompt(
                step_idx=step_idx, target_app=target_app,
                start_url=start_url, open_urls=open_urls,
                pruned_axtree=pruned,
            )
            new_response = _STRIP_THINK_RE.sub(
                "", str(row.get("response", "") or ""),
            ).strip()

            df_chunk.iat[i, df_chunk.columns.get_loc("prompt")] = new_prompt
            df_chunk.iat[i, df_chunk.columns.get_loc("response")] = new_response
        except Exception as e:
            app_hint = Path(source_path).parent.name
            print(f"  [warn] Prompt rebuild failed for {app_hint} row {i}: {e}")


def prepare_compact_dataset(
    input_files: str | list[str],
    output_dir: str,
    val_ratio: float = 0.05,
    seed: int = 42,
    variant: str | None = None,
    apps: list[str] | None = None,
    rebuild_prompts: bool = True,
) -> tuple[str, str]:
    """
    Prepare SFT data for JAMEL-COMPACT training.

    Unlike original JAMEL, no memory compression is needed — the model
    learns memory online. We concatenate the data and split complete sessions
    into train/validation parquet files so recurrent chunks never cross the
    train/validation boundary.

    Args:
        input_files: can be:
                      - A single parquet file path
                      - A list of parquet file paths
                      - A directory path (e.g. ".../react-vision" or
                        ".../ExplorerSFT-ReAct_Dataset/data")
                      The directory case auto-discovers all
                      trajectory.parquet files under app subdirectories.
        output_dir:  directory to write train/val parquet
        val_ratio:   fraction of sessions for validation (at least one session)
        seed:        random seed for shuffling
        variant:     if input_files is a root containing both "react-text"
                     and "react-vision", filter to one variant.
                     Set to None to use both.
        apps:        optional list of app names to filter (e.g. ["weibo", "alibaba"])
        rebuild_prompts: if True (default), rebuild the ``prompt`` column from
                     atomic columns (``before_observation_str``, etc.) using
                     ``build_web_prompt()`` and strip ``<think>`` from
                     ``response``. This ensures training/eval prompt format
                     consistency, exactly as original JAMEL does.

    Returns:
        (train_path, val_path)
    """
    # ── Discover parquet files ──
    parquet_files = discover_parquet_files(input_files)

    # ── Filter by variant if specified ──
    if variant is not None:
        parquet_files = [f for f in parquet_files if f"/{variant}/" in f or
                         f"\\{variant}\\" in f]

    # ── Filter by app names if specified ──
    if apps is not None:
        app_set = set(apps)
        parquet_files = [f for f in parquet_files
                         if any(f"/{app}/" in f or f"\\{app}\\" in f
                                for app in app_set)]

    if not parquet_files:
        raise ValueError(f"No parquet files found in: {input_files}")

    print(f"[data] Discovered {len(parquet_files)} parquet files")
    for f in parquet_files[:5]:
        print(f"  {f}")
    if len(parquet_files) > 5:
        print(f"  ... and {len(parquet_files) - 5} more")

    # ── Two-phase loading to avoid OOM ──
    import gc

    essential_cols = [
        "session_id", "episode_idx", "step_idx", "session_step_idx",
        "target_app", "start_url", "target_url",
        "before_observation_str", "before_open_pages_urls",
        "before_screenshot", "response", "action", "reward",
        "prompt", "think", "raw_content", "parsed_content",
        "coverage_delta_score", "coverage_previous_score", "coverage_current_score",
    ]

    # ── Phase 1: Read metadata ONLY (no screenshots) to build shuffle indices ──
    no_screenshot_cols = [c for c in essential_cols if c != "before_screenshot"]
    frames_meta = []
    for p in parquet_files:
        try:
            df_meta_p = pd.read_parquet(p, columns=no_screenshot_cols)
        except (ValueError, KeyError):
            df_meta_p = pd.read_parquet(p)
        frames_meta.append(df_meta_p)
    df_meta = pd.concat(frames_meta, ignore_index=True)
    del frames_meta
    gc.collect()

    total_rows = len(df_meta)
    print(f"[data] Total: {total_rows} rows | "
          f"sessions={df_meta['session_id'].nunique() if 'session_id' in df_meta.columns else 'N/A'} | "
          f"apps={df_meta['target_app'].nunique() if 'target_app' in df_meta.columns else 'N/A'}")

    # ── Session-level shuffle and split ──
    rng = random.Random(seed)
    session_to_indices = {}
    for row_idx, session_id in enumerate(df_meta["session_id"].astype(str)):
        session_to_indices.setdefault(session_id, []).append(row_idx)
    session_ids = list(session_to_indices)
    rng.shuffle(session_ids)

    if len(session_ids) < 2:
        raise ValueError(
            "Session-level splitting requires at least two distinct sessions"
        )
    val_session_count = max(1, int(round(len(session_ids) * max(val_ratio, 0.0))))
    val_session_count = min(val_session_count, len(session_ids) - 1)
    val_sessions = set(session_ids[:val_session_count])
    val_indices = {
        row_idx
        for session_id in val_sessions
        for row_idx in session_to_indices[session_id]
    }
    all_indices = set(range(len(df_meta)))
    train_indices = all_indices - val_indices

    del df_meta
    gc.collect()

    # ── Phase 2: Write train/val parquet using pyarrow ParquetWriter ──
    # Avoids pd.concat OOM by writing each chunk's filtered rows directly.
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "compact_train.parquet"
    val_path = output_path / "compact_val.parquet"
    global_idx = 0
    train_writer = None
    val_writer = None

    try:
        for p in parquet_files:
            try:
                df_chunk = pd.read_parquet(p, columns=essential_cols)
            except (ValueError, KeyError):
                df_chunk = pd.read_parquet(p)
            chunk_len = len(df_chunk)

            chunk_global_indices = list(range(global_idx, global_idx + chunk_len))
            train_local = [j for j in range(chunk_len) if chunk_global_indices[j] in train_indices]
            val_local = [j for j in range(chunk_len) if chunk_global_indices[j] in val_indices]

            if train_local:
                train_chunk = df_chunk.iloc[train_local]
                if rebuild_prompts:
                    _apply_prompt_rebuild_inplace(train_chunk, p, len(train_chunk))
                train_table = pa.Table.from_pandas(train_chunk)
                if train_writer is None:
                    train_writer = pq.ParquetWriter(train_path, train_table.schema)
                train_writer.write_table(train_table)

            if val_local:
                val_chunk = df_chunk.iloc[val_local]
                if rebuild_prompts:
                    _apply_prompt_rebuild_inplace(val_chunk, p, len(val_chunk))
                val_table = pa.Table.from_pandas(val_chunk)
                if val_writer is None:
                    val_writer = pq.ParquetWriter(val_path, val_table.schema)
                val_writer.write_table(val_table)

            global_idx += chunk_len
            del df_chunk
            if global_idx % 3000 == 0:
                gc.collect()
                print(f"  ... processed {global_idx}/{total_rows} rows")
    finally:
        # Close writers and release pyarrow C++ resources to avoid
        # "terminate called without an active exception" on exit.
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()
        del train_writer, val_writer
        # Force pyarrow to release internal memory pool
        try:
            pa.default_memory_pool().release_unused()
        except Exception:
            pass

        del train_indices, val_indices, session_to_indices, session_ids, total_rows
    gc.collect()

    print(f"[data] Train: {train_path}")
    print(f"[data] Val:   {val_path}")
    # Quick count
    train_rows = len(pd.read_parquet(train_path, columns=["action"]))
    val_rows = len(pd.read_parquet(val_path, columns=["action"]))
    print(f"[data] Train rows: {train_rows}, Val rows: {val_rows}")

    return str(train_path), str(val_path)
