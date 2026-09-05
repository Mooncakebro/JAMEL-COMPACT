"""Publication-style plots for the 4B COMPACT uncertainty analysis.

Examples:
    python scripts/plot_4b_uncertainty_v3.py layer-average \
        --compact-dir outputs/compact_v2_4b_debug_eval_1th \
        --compact-dir outputs/compact_v2_4b_debug_eval_2th \
        --compact-dir outputs/compact_v2_4b_final_debug_eval_1th \
        --compact-dir outputs/compact_v2_4b_final_debug_eval_2th \
        --output outputs/compact_v2_4b_visualization_v3/layer_average.pdf

    python scripts/plot_4b_uncertainty_v3.py reward-pair \
        --compact-trajectory outputs/compact_v2_4b_final_debug_eval_1th/vipshop/session0/trajectory_vipshop_20260901_092851.parquet \
        --baseline-trajectory outputs/baseline_4b_debug_eval_1th/vipshop/session0/trajectory_vipshop_20260901_172004.parquet \
        --output-dir outputs/compact_v2_4b_visualization_v3/vipshop_pair

    python scripts/plot_4b_uncertainty_v3.py slot-heatmaps \
        --success-trajectory outputs/compact_v2_4b_final_debug_eval_1th/vipshop/session0/trajectory_vipshop_20260901_092851.parquet \
        --failure-trajectory outputs/compact_v2_4b_final_debug_eval_2th/youku/session0/trajectory_youku_20260826_161023.parquet \
        --output outputs/compact_v2_4b_visualization_v3/slot_dynamics.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


COLORS = {
    "predicted_error": "#0072B2",
    "gain": "#D55E00",
    "write": "#009E73",
    "injection": "#CC79A7",
    "p_hat": "#0072B2",
    "r": "#E69F00",
    "k": "#D55E00",
    "reward": "#009E73",
}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
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
        }
    )


def _decorate(axis: Any) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _parse_diagnostics(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _array_mean(value: Any) -> float:
    values = np.asarray(value, dtype=float)
    return float(np.nanmean(values))


def _trajectory_paths(directories: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        if directory.is_file():
            paths.append(directory)
        else:
            paths.extend(directory.glob("*/session*/trajectory_*.parquet"))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No trajectory parquet files were found")
    return unique


def _load_layer_rows(paths: Iterable[Path]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for path in paths:
        trajectory = pd.read_parquet(path).sort_values("step")
        for row in trajectory.itertuples(index=False):
            payload = _parse_diagnostics(getattr(row, "uncertainty_diagnostics", None))
            if not payload or not payload.get("layers"):
                continue
            for layer_index, layer in enumerate(payload["layers"]):
                records.append(
                    {
                        "trajectory": str(path),
                        "app": path.parent.parent.name,
                        "step": int(getattr(row, "step")),
                        "layer": layer_index,
                        "predicted_error": _array_mean(layer["surprise"]),
                        "gain": _array_mean(layer["k"]),
                        "write": _array_mean(layer["write_norm"]),
                        "injection": _array_mean(layer["injection_ratio"]),
                    }
                )
    if not records:
        raise ValueError("No uncertainty diagnostics were found in the supplied trajectories")
    return pd.DataFrame(records)


def _plot_layer_average(frame: pd.DataFrame, output: Path, model_label: str) -> None:
    import matplotlib.pyplot as plt

    summary = frame.groupby("layer", as_index=False).agg(
        predicted_error=("predicted_error", "mean"),
        gain=("gain", "mean"),
        write=("write", "mean"),
        logged_steps=("step", "count"),
    )
    _style()
    metrics = (
        ("predicted_error", "Predicted observation error $e$"),
        ("gain", "Kalman gain $K$"),
        ("write", r"Effective memory write $\|K\odot\Delta M\|$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    max_layer = int(summary.layer.max())
    for axis, (column, label) in zip(axes, metrics):
        axis.plot(
            summary.layer,
            summary[column],
            color=COLORS[column],
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            markevery=max(1, len(summary) // 12),
        )
        axis.set_xlabel("Transformer layer")
        axis.set_ylabel(label)
        axis.set_xlim(-0.5, max_layer + 0.5)
        axis.set_xticks(np.unique(np.linspace(0, max_layer, 7, dtype=int)))
        _decorate(axis)
    fig.suptitle(
        f"{model_label}: Mean Uncertainty by Layer Across All Logged Tasks",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _trajectory_summary(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    actions = frame["action"].astype(str)
    summary = {
        "app": path.parent.parent.name,
        "trajectory": str(path),
        "steps": len(frame),
        "total_reward": float(frame["reward"].sum()),
        "unique_actions": int(actions.nunique()),
        "repeated_actions": int(actions.duplicated().sum()),
        "reward_steps": [int(step) for step in frame.loc[frame["reward"] > 0, "step"]],
    }
    return frame, summary


def _plot_reward_trajectory(frame: pd.DataFrame, summary: dict[str, Any], label: str, output: Path, y_max: float) -> None:
    import matplotlib.pyplot as plt

    _style()
    x = frame["step"].to_numpy()
    cumulative = frame["cumulative_reward"].to_numpy(dtype=float)
    rewarded = frame["reward"].to_numpy(dtype=float) > 0
    fig, axis = plt.subplots(figsize=(5.2, 2.85), constrained_layout=True)
    axis.step(x, cumulative, where="post", color=COLORS["reward"], linewidth=2.0)
    axis.scatter(
        x[rewarded],
        cumulative[rewarded],
        color=COLORS["reward"],
        edgecolors="white",
        linewidths=0.4,
        s=22,
        zorder=3,
    )
    axis.set_xlabel("Evaluation step")
    axis.set_ylabel("Cumulative reward")
    axis.set_xlim(0, max(1, int(x.max())))
    axis.set_ylim(0, max(y_max, 1.0) * 1.08)
    axis.set_title(
        f"{label} — {summary['app']}\n"
        f"final reward={summary['total_reward']:.0f}, "
        f"unique actions={summary['unique_actions']}/{summary['steps']}",
        fontsize=10,
        fontweight="bold",
    )
    _decorate(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_reward_pair(compact_path: Path, baseline_path: Path, output_dir: Path) -> None:
    compact_frame, compact_summary = _trajectory_summary(compact_path)
    baseline_frame, baseline_summary = _trajectory_summary(baseline_path)
    y_max = max(compact_summary["total_reward"], baseline_summary["total_reward"])
    _plot_reward_trajectory(
        compact_frame,
        compact_summary,
        "COMPACT-4B",
        output_dir / f"{compact_summary['app']}_compact_reward.pdf",
        y_max,
    )
    _plot_reward_trajectory(
        baseline_frame,
        baseline_summary,
        "Baseline-4B",
        output_dir / f"{baseline_summary['app']}_baseline_reward.pdf",
        y_max,
    )
    pd.DataFrame([compact_summary, baseline_summary]).to_csv(
        output_dir / f"{compact_summary['app']}_reward_pair_summary.csv", index=False
    )


def _load_slot_matrices(path: Path, layer_start: int, layer_end: int) -> dict[str, np.ndarray]:
    trajectory = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    matrices: dict[str, list[np.ndarray]] = {"p_hat": [], "r": [], "k": []}
    for row in trajectory.itertuples(index=False):
        payload = _parse_diagnostics(getattr(row, "uncertainty_diagnostics", None))
        if not payload or not payload.get("layers"):
            continue
        selected = payload["layers"][layer_start:layer_end]
        values: dict[str, list[np.ndarray]] = {"p_hat": [], "r": [], "k": []}
        for layer in selected:
            for key in values:
                values[key].append(np.asarray(layer[key], dtype=float).reshape(-1))
        for key, numbers in values.items():
            matrices[key].append(np.nanmean(np.stack(numbers), axis=0))
    if not matrices["p_hat"]:
        raise ValueError(f"No diagnostics found for layer range {layer_start}:{layer_end} in {path}")
    return {key: np.stack(values) for key, values in matrices.items()}


def _plot_slot_heatmaps(success_path: Path, failure_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    success_frame, success_summary = _trajectory_summary(success_path)
    failure_frame, failure_summary = _trajectory_summary(failure_path)
    success_layers = len(_parse_diagnostics(success_frame.iloc[0]["uncertainty_diagnostics"])["layers"])
    failure_layers = len(_parse_diagnostics(failure_frame.iloc[0]["uncertainty_diagnostics"])["layers"])
    if success_layers != failure_layers:
        raise ValueError("Success and failure trajectories must have the same number of layers")
    boundaries = np.linspace(0, success_layers, 4, dtype=int)
    _style()
    fig = plt.figure(figsize=(12.6, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(4, 6, height_ratios=[0.75, 1.6, 1.6, 1.6])
    cases = ((success_path, success_frame, success_summary, 0), (failure_path, failure_frame, failure_summary, 3))
    metric_keys = (("p_hat", r"$\hat P$"), ("r", "$R$"), ("k", "$K$"))
    for path, trajectory, summary, column in cases:
        axis = fig.add_subplot(grid[0, column:column + 3])
        x = trajectory["step"].to_numpy()
        axis.step(x, trajectory["cumulative_reward"], where="post", color=COLORS["reward"], linewidth=1.8)
        axis.scatter(
            x[trajectory["reward"].to_numpy() > 0],
            trajectory.loc[trajectory["reward"] > 0, "cumulative_reward"],
            color=COLORS["reward"],
            edgecolors="white",
            linewidths=0.35,
            s=16,
            zorder=3,
        )
        axis.set_title(
            f"{('Success' if column == 0 else 'Failure')}: {summary['app']}\n"
            f"final reward={summary['total_reward']:.0f}, repeated actions={summary['repeated_actions']}",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_ylabel("Reward")
        axis.set_xlabel("Evaluation step")
        axis.set_ylim(0, max(1.0, summary["total_reward"]) * 1.08)
        _decorate(axis)
        for row_index, (layer_start, layer_end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
            matrices = _load_slot_matrices(path, layer_start, layer_end)
            counterpart_path = failure_path if column == 0 else success_path
            counterpart = _load_slot_matrices(counterpart_path, layer_start, layer_end)
            for metric_index, (key, label) in enumerate(metric_keys):
                metric_axis = fig.add_subplot(grid[row_index, column + metric_index])
                matrix = matrices[key]
                combined = np.concatenate([matrix, counterpart[key]])
                vmin = float(np.nanmin(combined))
                vmax = float(np.nanmax(combined))
                image = metric_axis.imshow(
                    matrix.T,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax if vmax > vmin else vmin + 1e-6,
                )
                metric_axis.set_title(label)
                metric_axis.set_xlabel("Evaluation step")
                metric_axis.set_ylabel(
                    f"{('Shallow' if row_index == 1 else 'Middle' if row_index == 2 else 'Deep')}\nslot"
                )
                metric_axis.set_yticks(np.arange(matrix.shape[1]))
                metric_axis.set_xticks(np.unique(np.linspace(0, matrix.shape[0] - 1, 4, dtype=int)))
                _decorate(metric_axis)
                fig.colorbar(image, ax=metric_axis, fraction=0.045, pad=0.03)
    fig.suptitle(
        "COMPACT-4B Memory Slot Heatmaps: Success versus Failure\n"
        "Each heatmap retains all 16 slots and averages only across the selected layer band",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    layer_parser = subparsers.add_parser("layer-average")
    layer_parser.add_argument("--compact-dir", action="append", type=Path, required=True)
    layer_parser.add_argument("--model-label", default="COMPACT")
    layer_parser.add_argument("--output", type=Path, required=True)
    layer_parser.add_argument("--csv-output", type=Path)

    pair_parser = subparsers.add_parser("reward-pair")
    pair_parser.add_argument("--compact-trajectory", type=Path, required=True)
    pair_parser.add_argument("--baseline-trajectory", type=Path, required=True)
    pair_parser.add_argument("--output-dir", type=Path, required=True)

    slot_parser = subparsers.add_parser("slot-heatmaps", aliases=["slot-dynamics"])
    slot_parser.add_argument("--success-trajectory", type=Path, required=True)
    slot_parser.add_argument("--failure-trajectory", type=Path, required=True)
    slot_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "layer-average":
        frame = _load_layer_rows(_trajectory_paths(args.compact_dir))
        _plot_layer_average(frame, args.output, args.model_label)
        if args.csv_output:
            summary = frame.groupby("layer", as_index=False).agg(
                predicted_observation_error=("predicted_error", "mean"),
                kalman_gain=("gain", "mean"),
                effective_memory_write=("write", "mean"),
                logged_steps=("step", "count"),
                trajectories=("trajectory", "nunique"),
            )
            args.csv_output.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.csv_output, index=False)
        print(f"Saved layer-average plot to {args.output} from {frame['trajectory'].nunique()} trajectories")
    elif args.command == "reward-pair":
        _plot_reward_pair(args.compact_trajectory, args.baseline_trajectory, args.output_dir)
        print(f"Saved COMPACT/Baseline reward plots to {args.output_dir}")
    else:
        _plot_slot_heatmaps(args.success_trajectory, args.failure_trajectory, args.output)
        print(f"Saved slot-heatmaps plot to {args.output}")


if __name__ == "__main__":
    main()
