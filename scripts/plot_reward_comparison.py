#!/usr/bin/env python3
"""Plot accumulated reward curves for one COMPACT/Baseline pair."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {
    "compact": "#0072B2",
    "baseline": "#D55E00",
}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_trajectory(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Trajectory is empty: {path}")
    if "reward" not in frame:
        raise ValueError(f"Trajectory has no reward column: {path}")
    frame["reward"] = pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0)
    if "cumulative_reward" in frame:
        frame["accumulated_reward"] = pd.to_numeric(
            frame["cumulative_reward"], errors="coerce"
        ).fillna(frame["reward"].cumsum())
    else:
        frame["accumulated_reward"] = frame["reward"].cumsum()
    return frame


def _decorate(axis) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_comparison(
    compact_path: Path,
    baseline_path: Path,
    output: Path,
    compact_label: str,
    baseline_label: str,
    width: float,
    height: float,
) -> None:
    import matplotlib.pyplot as plt

    compact = _load_trajectory(compact_path)
    baseline = _load_trajectory(baseline_path)
    maximum_reward = max(
        float(compact["accumulated_reward"].max()),
        float(baseline["accumulated_reward"].max()),
    )
    x_max = max(int(compact["step"].max()), int(baseline["step"].max()))

    _style()
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    for frame, label, color in (
        (compact, compact_label, COLORS["compact"]),
        (baseline, baseline_label, COLORS["baseline"]),
    ):
        steps = frame["step"].to_numpy(dtype=float)
        accumulated = frame["accumulated_reward"].to_numpy(dtype=float)
        rewarded = frame["reward"].to_numpy(dtype=float) > 0
        axis.step(
            steps,
            accumulated,
            where="post",
            color=color,
            linewidth=2.2,
            label=label,
        )
        axis.scatter(
            steps[rewarded],
            accumulated[rewarded],
            color=color,
            edgecolors="white",
            linewidths=0.45,
            s=24,
            zorder=3,
        )

    axis.set_xlabel("Evaluation step")
    axis.set_ylabel("Accumulated reward")
    axis.set_xlim(0, max(1, x_max))
    axis.set_ylim(0, max(1.0, maximum_reward) * 1.12)
    axis.set_xticks(np.arange(0, x_max + 1, 10))
    axis.legend(loc="upper left", ncol=2, handlelength=2.8, columnspacing=1.6)
    _decorate(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-label", default="COMPACT")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--width", type=float, default=8.4)
    parser.add_argument("--height", type=float, default=2.7)
    args = parser.parse_args()
    plot_comparison(
        args.compact_trajectory,
        args.baseline_trajectory,
        args.output,
        args.compact_label,
        args.baseline_label,
        args.width,
        args.height,
    )
    print(f"Saved reward comparison to {args.output}")


if __name__ == "__main__":
    main()
