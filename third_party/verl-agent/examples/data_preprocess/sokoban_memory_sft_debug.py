import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig


def stable_seed(*parts: Any) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    for part in parts:
        hasher.update(str(part).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)


def clean_prompt(prompt: str) -> str:
    return prompt.replace("<image>\n", "").replace("<image>", "").strip()


def build_memory_tokens(
    row: pd.Series,
    *,
    hidden_size: int,
    memory_items: int,
    scale: float,
) -> list[list[float]]:
    seed = stable_seed(
        row.get("trajectory_id", ""),
        row.get("step_id", ""),
        row.get("prompt", ""),
        row.get("response", ""),
    )
    rng = np.random.default_rng(seed)
    tokens = rng.normal(loc=0.0, scale=scale, size=(memory_items, hidden_size)).astype(np.float32)
    return tokens.tolist()


def convert_split(
    input_path: Path,
    output_path: Path,
    *,
    hidden_size: int,
    memory_items: int,
    memory_scale: float,
    limit: int | None,
) -> int:
    dataframe = pd.read_parquet(input_path)
    if limit is not None:
        dataframe = dataframe.head(limit)

    rows = []
    for _, row in dataframe.iterrows():
        rows.append(
            {
                "data_source": row.get("data_source", "sokoban_rule_sft"),
                "trajectory_id": row.get("trajectory_id", ""),
                "step_id": int(row.get("step_id", 0)),
                "prompt": clean_prompt(str(row["prompt"])),
                "response": str(row["response"]),
                "memory_tokens": build_memory_tokens(
                    row,
                    hidden_size=hidden_size,
                    memory_items=memory_items,
                    scale=memory_scale,
                ),
                "memory_attention_mask": [1] * memory_items,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def infer_vocab_size(tokenizer_source: Path) -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    size = max(int(tokenizer.vocab_size), len(tokenizer))

    config = AutoConfig.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    for candidate in (
        getattr(config, "vocab_size", None),
        getattr(text_config, "vocab_size", None),
    ):
        if candidate is not None:
            size = max(size, int(candidate))

    # Keep the embedding table friendly to tensor-core tile sizes.
    return ((size + 63) // 64) * 64


def build_tiny_model(
    model_dir: Path,
    *,
    tokenizer_source: Path,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    model_max_length: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_max_length

    vocab_size = infer_vocab_size(tokenizer_source)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny Sokoban memory-token SFT debug dataset.")
    parser.add_argument("--source-dir", default="data/sokoban_rule_sft")
    parser.add_argument("--output-dir", default="/tmp/sokoban_memory_sft_debug")
    parser.add_argument("--memory-hidden-size", type=int, default=128)
    parser.add_argument("--memory-items", type=int, default=4)
    parser.add_argument("--memory-scale", type=float, default=0.02)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--build-tiny-model", action="store_true")
    parser.add_argument("--tiny-model-dir", default=None)
    parser.add_argument(
        "--tokenizer-source",
        default="/mnt/nvme0/tsz/modelscope_cache/models/Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument("--tiny-hidden-size", type=int, default=128)
    parser.add_argument("--tiny-num-layers", type=int, default=4)
    parser.add_argument("--tiny-num-attention-heads", type=int, default=4)
    parser.add_argument("--model-max-length", type=int, default=1024)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    train_rows = convert_split(
        source_dir / "train.parquet",
        output_dir / "train.parquet",
        hidden_size=args.memory_hidden_size,
        memory_items=args.memory_items,
        memory_scale=args.memory_scale,
        limit=args.limit_train,
    )
    val_rows = convert_split(
        source_dir / "val.parquet",
        output_dir / "val.parquet",
        hidden_size=args.memory_hidden_size,
        memory_items=args.memory_items,
        memory_scale=args.memory_scale,
        limit=args.limit_val,
    )

    tiny_model_dir = Path(args.tiny_model_dir) if args.tiny_model_dir else output_dir / "tiny_qwen_tokenizer_llama"
    if args.build_tiny_model:
        build_tiny_model(
            tiny_model_dir,
            tokenizer_source=Path(args.tokenizer_source),
            hidden_size=args.tiny_hidden_size,
            num_layers=args.tiny_num_layers,
            num_attention_heads=args.tiny_num_attention_heads,
            model_max_length=args.model_max_length,
        )

    summary = {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_file": str((output_dir / "train.parquet").resolve()),
        "val_file": str((output_dir / "val.parquet").resolve()),
        "memory_hidden_size": args.memory_hidden_size,
        "memory_items": args.memory_items,
        "tiny_model_dir": str(tiny_model_dir.resolve()) if args.build_tiny_model else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
