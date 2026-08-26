"""Create the recommended COMPACT success/failure uncertainty figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLORS = ("#0072B2", "#D55E00")
MARKERS = ("o", "s")


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _parse_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value.get("layers"):
        raise ValueError("Trajectory row has no uncertainty diagnostics")
    return value


def _mean(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.nanmean(array))


def _load_case(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    step_records: list[dict[str, float | int | str]] = []
    layer_records: list[dict[str, float | int | str]] = []

    for row_index, row in trajectory.iterrows():
        payload = _parse_payload(row["uncertainty_diagnostics"])
        layers = payload["layers"]
        final_layer = layers[-1]
        step_records.append({
            "case": label,
            "step": int(row["step"]),
            "previous_reward": np.nan if row_index == 0 else float(trajectory.iloc[row_index - 1]["reward"] or 0.0),
            "surprise": _mean(final_layer["surprise"]),
            "p_hat": _mean(final_layer["p_hat"]),
            "k": _mean(final_layer["k"]),
            "write_norm": _mean(final_layer["write_norm"]),
        })
        for layer_index, layer in enumerate(layers):
            layer_records.append({
                "case": label,
                "step": int(row["step"]),
                "layer": layer_index,
                "surprise": _mean(layer["surprise"]),
                "k": _mean(layer["k"]),
                "write_norm": _mean(layer["write_norm"]),
                "injection_ratio": _mean(layer["injection_ratio"]),
            })

    return pd.DataFrame(step_records), pd.DataFrame(layer_records)


def _decorate(axis: Any) -> None:
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_cross_layer(layer_frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    metrics = (
        ("surprise", "Surprise $e$", False),
        ("k", "Kalman gain $K$", False),
        ("write_norm", r"Effective memory write $\|K\odot\Delta M\|$", False),
        ("injection_ratio", "Injection / hidden norm", True),
    )
    summary = layer_frame.groupby(["case", "layer"], as_index=False).mean(numeric_only=True)
    max_layer = int(summary["layer"].max())
    third = (max_layer + 1) / 3

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for axis, (metric, ylabel, use_log) in zip(axes.flat, metrics):
        axis.axvspan(-0.5, third - 0.5, color="#E8F1F8", alpha=0.65, zorder=0)
        axis.axvspan(third - 0.5, 2 * third - 0.5, color="#F2F2F2", alpha=0.65, zorder=0)
        axis.axvspan(2 * third - 0.5, max_layer + 0.5, color="#FBEBDD", alpha=0.65, zorder=0)
        for case_index, (case, group) in enumerate(summary.groupby("case", sort=False)):
            axis.plot(
                group["layer"],
                group[metric],
                color=COLORS[case_index],
                marker=MARKERS[case_index],
                markevery=3,
                markersize=3.2,
                linewidth=1.7,
                label=case,
            )
        axis.set_xlabel("Transformer layer")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.5, max_layer + 0.5)
        axis.set_xticks([0, 4, 9, 14, 19, 24, max_layer])
        if use_log:
            axis.set_yscale("log")
        _decorate(axis)

    axes[0, 0].legend(loc="upper left")
    axes[0, 0].text(0.02, 0.04, "shallow", transform=axes[0, 0].transAxes, color="#4C78A8", fontsize=8)
    axes[0, 0].text(0.39, 0.04, "middle", transform=axes[0, 0].transAxes, color="#666666", fontsize=8)
    axes[0, 0].text(0.76, 0.04, "deep", transform=axes[0, 0].transAxes, color="#C46A2B", fontsize=8)
    fig.suptitle("Cross-Layer Uncertainty Profiles", fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_lagged(step_frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    _style()
    metrics = (
        ("surprise", "Surprise $e$"),
        ("p_hat", "Predicted variance $\\hat P$"),
        ("k", "Kalman gain $K$"),
        ("write_norm", r"Effective memory write $\|K\odot\Delta M\|$"),
    )
    frame = step_frame.copy()
    random = np.random.default_rng(7)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for case_index, (case, group) in enumerate(frame.groupby("case", sort=False)):
            group = group[group["previous_reward"].notna()]
            x = group["previous_reward"].to_numpy(dtype=float)
            y = group[metric].to_numpy(dtype=float)
            offset = -0.08 if case_index == 0 else 0.08
            jitter = random.uniform(-0.025, 0.025, size=len(group))
            axis.scatter(
                x + offset + jitter,
                y,
                s=20,
                alpha=0.42,
                color=COLORS[case_index],
                marker=MARKERS[case_index],
                edgecolors="white",
                linewidths=0.35,
                label=case,
            )
            for reward_value in (0.0, 1.0):
                values = y[x == reward_value]
                if not len(values):
                    continue
                mean = float(np.mean(values))
                sem = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                axis.errorbar(
                    reward_value + offset,
                    mean,
                    yerr=sem,
                    color=COLORS[case_index],
                    marker=MARKERS[case_index],
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    capsize=3,
                    linewidth=1.4,
                    zorder=4,
                )
        axis.set_xlabel(r"Previous action reward $r_{t-1}$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.28, 1.28)
        axis.set_xticks([0, 1], ["0 (no reward)", "1 (reward)"])
        _decorate(axis)

    axes[0, 0].legend(loc="best")
    fig.suptitle("Lagged Reward–Uncertainty Relationship", fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-trajectory", type=Path, required=True)
    parser.add_argument("--failure-trajectory", type=Path, required=True)
    parser.add_argument("--success-label", default="Vipshop success")
    parser.add_argument("--failure-label", default="Youku failure")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    success_steps, success_layers = _load_case(args.success_trajectory, args.success_label)
    failure_steps, failure_layers = _load_case(args.failure_trajectory, args.failure_label)
    steps = pd.concat([success_steps, failure_steps], ignore_index=True)
    layers = pd.concat([success_layers, failure_layers], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_cross_layer(layers, args.output_dir / "cross_layer_profiles.pdf")
    _plot_lagged(steps, args.output_dir / "lagged_reward_uncertainty.pdf")
    steps.to_csv(args.output_dir / "lagged_reward_uncertainty_data.csv", index=False)
    layers.to_csv(args.output_dir / "cross_layer_profiles_data.csv", index=False)
    print(f"Saved recommended comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
