from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


def _to_numeric_array(value, *, dtype):
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(dtype, copy=False)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return np.asarray(value, dtype=dtype)


class MemoryTokenSFTDataset(Dataset):
    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "response")
        self.memory_tokens_key = config.get("memory_tokens_key", "memory_tokens")
        self.memory_attention_mask_key = config.get(
            "memory_attention_mask_key",
            "memory_attention_mask",
        )
        self.memory_max_items = config.get("memory_max_items", None)
        self.use_shm = config.get("use_shm", False)

        self._download()
        self._read_files()

    def _download(self) -> None:
        for idx, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[idx] = copy_to_local(
                parquet_file,
                verbose=True,
                use_shm=self.use_shm,
            )

    def _read_files(self) -> None:
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _pad_memory(self, memory_tokens: torch.Tensor, memory_attention_mask: torch.Tensor):
        if memory_tokens.ndim == 1:
            memory_tokens = memory_tokens.unsqueeze(0)
        if memory_attention_mask.ndim == 0:
            memory_attention_mask = memory_attention_mask.unsqueeze(0)

        max_items = self.memory_max_items or memory_tokens.shape[0]
        hidden_size = memory_tokens.shape[-1]
        padded_tokens = torch.zeros((max_items, hidden_size), dtype=memory_tokens.dtype)
        padded_mask = torch.zeros((max_items,), dtype=torch.long)
        valid_count = min(max_items, memory_tokens.shape[0])
        padded_tokens[:valid_count] = memory_tokens[:valid_count]
        padded_mask[:valid_count] = memory_attention_mask[:valid_count].to(torch.long)
        return padded_tokens, padded_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        prompt = row[self.prompt_key]
        response = row[self.response_key]

        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = self.tokenizer.apply_chat_template(
            prompt_chat,
            add_generation_prompt=True,
            tokenize=False,
        )
        response_chat_str = str(response) + self.tokenizer.eos_token

        prompt_ids_output = self.tokenizer(
            prompt_chat_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        response_ids_output = self.tokenizer(
            response_chat_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        response_ids = response_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            pad_token_id = self.tokenizer.pad_token_id
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)]
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)]
            )
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            else:
                raise ValueError(
                    f"{sequence_length=} is larger than {self.max_length=} with truncation={self.truncation!r}."
                )

        position_ids = compute_position_id_with_mask(attention_mask)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        memory_tokens = torch.tensor(
            _to_numeric_array(row[self.memory_tokens_key], dtype=np.float32),
            dtype=torch.float32,
        )
        if self.memory_attention_mask_key in row and row[self.memory_attention_mask_key] is not None:
            memory_attention_mask = torch.tensor(
                _to_numeric_array(row[self.memory_attention_mask_key], dtype=np.int64),
                dtype=torch.long,
            )
        else:
            memory_attention_mask = torch.ones(memory_tokens.shape[0], dtype=torch.long)
        memory_tokens, memory_attention_mask = self._pad_memory(
            memory_tokens,
            memory_attention_mask,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "memory_tokens": memory_tokens,
            "memory_attention_mask": memory_attention_mask,
        }
