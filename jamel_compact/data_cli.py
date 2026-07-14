"""
CLI for JAMEL-COMPACT data preparation.

Usage:
    python -m jamel_compact.data_cli \
        --input /path/to/react-vision \
        --output-dir data/compact_sft_data \
        --val-ratio 0.05 \
        --variant react-vision \
        --apps weibo,alibaba
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path so we can import jamel_compact.data without triggering
# the package __init__ (which imports torch via model.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jamel_compact.data import prepare_compact_dataset


def main():
    parser = argparse.ArgumentParser(description="JAMEL-COMPACT Data Preparation")
    parser.add_argument(
        "--input", required=True,
        help="Input parquet file, list of files, or directory containing app subdirs",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for train/val parquet",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variant", default=None,
        help="Filter to one variant: 'react-text' or 'react-vision' (default: auto)",
    )
    parser.add_argument(
        "--apps", default=None,
        help="Comma-separated app names to filter (default: all apps)",
    )
    parser.add_argument(
        "--no-rebuild-prompts", action="store_true",
        help="Keep upstream prompt/response columns as-is. "
             "Default: rebuild prompts from atomic columns via build_web_prompt() "
             "and strip <think> from response (matches original JAMEL behavior).",
    )
    args = parser.parse_args()

    apps_list = None
    if args.apps:
        apps_list = [a.strip() for a in args.apps.split(",") if a.strip()]

    train_path, val_path = prepare_compact_dataset(
        input_files=args.input,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        variant=args.variant,
        apps=apps_list,
        rebuild_prompts=not args.no_rebuild_prompts,
    )
    print(f"\nTrain: {train_path}")
    print(f"Val:   {val_path}")


if __name__ == "__main__":
    main()