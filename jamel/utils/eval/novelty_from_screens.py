#!/usr/bin/env python3
"""
Compute novelty rewards from screenshot sequences using CLIP image embeddings.

Novelty per step = 1 - max cosine similarity to any previous step.
Outputs:
  - novelty_table.csv
  - novelty_table.md
  - novelty_curve.png
  - cumulative_novelty_curve.png
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute novelty from screenshots.")
    parser.add_argument(
        "--input-dir",
        default="exploration_data/archive/0316_2",
        help="Directory containing exploration runs with images/step_*.png.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/novelty_0316_2",
        help="Directory to write tables and plots.",
    )
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name for CLIP image encoder.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for embedding model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model files from local cache (no downloads).",
    )
    return parser.parse_args()


@dataclass
class ImageStep:
    run_name: str
    run_step: int
    path: str


def list_image_steps(root_dir: str) -> List[ImageStep]:
    steps: List[ImageStep] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input dir not found: {root_dir}")

    run_names = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    step_re = re.compile(r"step_(\d+)\.png$", re.IGNORECASE)

    for run_name in run_names:
        images_dir = os.path.join(root_dir, run_name, "images")
        if not os.path.isdir(images_dir):
            continue
        image_files = []
        for fname in os.listdir(images_dir):
            match = step_re.match(fname)
            if not match:
                continue
            image_files.append((int(match.group(1)), fname))
        for step_num, fname in sorted(image_files, key=lambda x: x[0]):
            steps.append(
                ImageStep(
                    run_name=run_name,
                    run_step=step_num,
                    path=os.path.join(images_dir, fname),
                )
            )
    return steps


def load_clip(model_name: str, device: str, local_only: bool):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name, local_files_only=local_only)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_only)
    model.eval()
    model.to(device)
    return model, processor, device


def embed_image(model, processor, device: str, image_path: str):
    import torch
    from PIL import Image
    from torch.nn import functional as F

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = F.normalize(features, p=2, dim=-1)
    return features.squeeze(0)


def compute_novelty(
    steps: List[ImageStep],
    model_name: str,
    device: str,
    local_only: bool,
) -> List[dict]:
    import torch

    model, processor, device = load_clip(model_name, device, local_only)
    history: List[torch.Tensor] = []

    rows = []
    cumulative = 0.0
    for idx, step in enumerate(steps, start=1):
        embedding = embed_image(model, processor, device, step.path)
        if history:
            history_mat = torch.stack(history)
            similarities = torch.matmul(history_mat, embedding)
            max_similarity = similarities.max().item()
            novelty = 1.0 - max_similarity
        else:
            max_similarity = 0.0
            novelty = 0.0
        cumulative += novelty
        history.append(embedding)
        rows.append(
            {
                "global_step": idx,
                "run_name": step.run_name,
                "run_step": step.run_step,
                "image_path": step.path,
                "max_similarity": round(max_similarity, 6),
                "novelty": round(novelty, 6),
                "cumulative_novelty": round(cumulative, 6),
            }
        )
    return rows


def write_csv(rows: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(rows: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def draw_line_plot(
    values: List[float],
    out_path: str,
    title: str,
    y_label: str,
    run_boundaries: Optional[List[int]] = None,
) -> None:
    import matplotlib.pyplot as plt

    if not values:
        raise RuntimeError("No values to plot.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.color": "#dcdcdc",
            "grid.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(values) + 1), values, color="#1f4aa8", linewidth=2)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("step", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, which="major", axis="both")

    if run_boundaries:
        for boundary in run_boundaries:
            ax.axvline(boundary + 1, color="#bbbbbb", linestyle="--", linewidth=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def compute_run_boundaries(rows: List[dict]) -> List[int]:
    boundaries = []
    last_run = None
    for idx, row in enumerate(rows):
        run_name = row["run_name"]
        if last_run is None:
            last_run = run_name
            continue
        if run_name != last_run:
            boundaries.append(idx - 1)
            last_run = run_name
    return boundaries


def main() -> None:
    args = parse_args()
    steps = list_image_steps(args.input_dir)
    if not steps:
        raise RuntimeError("No images found under input dir.")

    rows = compute_novelty(
        steps=steps,
        model_name=args.model_name,
        device=args.device,
        local_only=args.local_files_only,
    )

    output_dir = args.output_dir
    csv_path = os.path.join(output_dir, "novelty_table.csv")
    md_path = os.path.join(output_dir, "novelty_table.md")
    write_csv(rows, csv_path)
    write_markdown_table(rows, md_path)

    novelty_values = [row["novelty"] for row in rows]
    cumulative_values = [row["cumulative_novelty"] for row in rows]
    boundaries = compute_run_boundaries(rows)

    draw_line_plot(
        novelty_values,
        os.path.join(output_dir, "novelty_curve.pdf"),
        title="Novelty Reward Over Time",
        y_label="novelty",
        run_boundaries=boundaries,
    )
    draw_line_plot(
        cumulative_values,
        os.path.join(output_dir, "cumulative_novelty_curve.pdf"),
        title="Cumulative Novelty Over Time",
        y_label="cumulative_novelty",
        run_boundaries=boundaries,
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {os.path.join(output_dir, 'novelty_curve.pdf')}")
    print(f"Wrote {os.path.join(output_dir, 'cumulative_novelty_curve.pdf')}")


if __name__ == "__main__":
    main()
