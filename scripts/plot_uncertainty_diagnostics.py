"""Plot COMPACT uncertainty dynamics from saved eval trajectories.

Examples:
    python scripts/plot_uncertainty_diagnostics.py \
        --trajectory outputs/compact_eval/app/session0/trajectory_*.parquet \
        --output outputs/compact_eval/app/session0/uncertainty_dynamics.pdf

    python scripts/plot_uncertainty_diagnostics.py \
        --eval-dir outputs/compact_eval --app weibo --session 0 \
        --mode calibration --output outputs/weibo_calibration.pdf

    python scripts/plot_uncertainty_diagnostics.py \
        --eval-dir outputs/compact_eval --mode event --output outputs/event_alignment.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _find_trajectories(
    trajectory: str | None,
    eval_dir: Path | None,
    app: str | None,
    session: int | None,
) -> list[Path]:
    if trajectory:
        paths = [Path(item) for item in sorted(glob.glob(trajectory))]
        if not paths and not any(c in trajectory for c in "*?["):
            paths = [Path(trajectory)]
    elif eval_dir is not None:
        root = eval_dir
        if app:
            root = root / app
        if session is not None:
            root = root / f"session{session}"
        paths = sorted(root.glob("**/trajectory_*.parquet"))
        paths += sorted(root.glob("**/trajectory_*.jsonl"))
    else:
        raise ValueError("Provide --trajectory or --eval-dir")
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError("No trajectory parquet/jsonl files found")
    return paths


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_diagnostics(value: Any) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _first_batch(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def _diagnostic_rows(paths: list[Path], layer: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        for row in _load_rows(path):
            payload = _parse_diagnostics(row.get("uncertainty_diagnostics"))
            if not payload:
                continue
            layers = payload.get("layers") or []
            if not layers:
                continue
            layer_index = layer if layer >= 0 else len(layers) + layer
            if not 0 <= layer_index < len(layers):
                continue
            data = layers[layer_index] or {}
            record: dict[str, Any] = {
                "source": str(path),
                "app": row.get("app", path.parent.parent.name),
                "session": row.get("session_idx", path.parent.name.replace("session", "")),
                "step": int(row.get("step", len(records))),
                "layer": int(data.get("layer", layer_index)),
                "reward": float(row.get("reward", 0.0) or 0.0),
                "delta_score": float(row.get("delta_score", 0.0) or 0.0),
                "current_score": float(row.get("current_score", 0.0) or 0.0),
                "cumulative_reward": float(row.get("cumulative_reward", 0.0) or 0.0),
                "gain_mode": payload.get("gain_mode", "learned"),
            }
            for key, value in data.items():
                values = _first_batch(value)
                if values.size:
                    record[f"{key}_mean"] = float(values.mean())
                    record[f"{key}_max"] = float(values.max())
                    record[f"{key}_p90"] = float(np.percentile(values, 90))
            records.append(record)
    if not records:
        raise ValueError(
            "No uncertainty diagnostics found. Re-run eval with --save-uncertainty "
            "or use GAIN_MODE=fixed/zero/one."
        )
    return pd.DataFrame(records).sort_values(["source", "step"]).reset_index(drop=True)


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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
    })


def _save_dynamics(frame: pd.DataFrame, output: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True, constrained_layout=True)
    x = frame["step"].to_numpy()
    axes[0].plot(
        x,
        frame["cumulative_reward"],
        color="#2ca02c",
        lw=1.8,
        drawstyle="steps-post",
        label="cumulative reward",
    )
    rewarded = frame["reward"] > 0
    axes[0].scatter(
        x[rewarded],
        frame.loc[rewarded, "cumulative_reward"],
        color="#2ca02c",
        edgecolors="white",
        linewidths=0.4,
        s=20,
        zorder=3,
        label="rewarded action",
    )
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="upper left", ncol=2)

    axes[1].plot(x, frame.get("surprise_mean", np.nan), color="#d62728", lw=1.6, label="surprise $e$")
    axes[1].plot(
        x,
        frame.get("surprise_inflation_mean", np.nan),
        color="#9467bd",
        lw=1.2,
        label=r"variance inflation $\gamma_e\,\mathrm{clip}(e_{t-1})$",
    )
    axes[1].set_ylabel("Prediction error")
    axes[1].legend(loc="upper left", ncol=2)

    axes[2].plot(x, frame.get("p_hat_mean", np.nan), color="#ff7f0e", lw=1.5, label="$\\hat P$")
    axes[2].plot(x, frame.get("p_new_mean", np.nan), color="#17becf", lw=1.5, label="posterior $P$")
    axes[2].plot(x, frame.get("r_mean", np.nan), color="#7f7f7f", lw=1.2, label="$R$")
    axes[2].set_ylabel("Uncertainty")
    axes[2].legend(loc="upper left", ncol=3)

    axes[3].plot(x, frame.get("k_mean", np.nan), color="#e377c2", lw=1.8, label="$K$")
    axes[3].plot(x, frame.get("write_norm_mean", np.nan), color="#8c564b", lw=1.3, label="$\\|K\\odot\\Delta M\\|$")
    axes[3].plot(x, frame.get("innovation_norm_mean", np.nan), color="#bcbd22", lw=1.1, label="$\\|\\Delta M\\|$")
    axes[3].set_ylabel("Memory update")
    axes[3].set_xlabel("Evaluation step")
    axes[3].legend(loc="upper left", ncol=3)

    for axis in axes:
        axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.8)
    if title:
        fig.suptitle(title, y=1.01, fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _save_calibration(frame: pd.DataFrame, output: Path, title: str | None) -> None:
    import matplotlib.pyplot as plt

    _style()
    x = frame["r_mean"].to_numpy()
    y = frame["surprise_mean"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        raise ValueError("At least two valid R/surprise points are required")
    edges = np.quantile(x, np.linspace(0, 1, min(11, len(x) + 1)))
    edges = np.unique(edges)
    if len(edges) < 2:
        center = float(x[0])
        width = max(0.5, abs(center) * 0.05)
        edges = np.array([center - width, center + width])
    bins = np.digitize(x, edges[1:-1], right=True)
    means_x, means_y, counts = [], [], []
    for index in range(len(edges) - 1):
        selected = bins == index
        if selected.any():
            means_x.append(x[selected].mean())
            means_y.append(y[selected].mean())
            counts.append(selected.sum())
    fig, axis = plt.subplots(figsize=(4.5, 3.8), constrained_layout=True)
    axis.scatter(x, y, s=10, alpha=0.18, color="#777777", rasterized=True, label="steps")
    axis.plot(means_x, means_y, marker="o", ms=4, lw=1.8, color="#1f77b4", label="binned mean")
    lo = min(min(means_x), min(means_y))
    hi = max(max(means_x), max(means_y))
    axis.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#222222", label="$R=e$")
    log_x = np.log1p(x)
    log_y = np.log1p(y)
    correlation = (
        float(np.corrcoef(log_x, log_y)[0, 1])
        if np.std(log_x) > 0 and np.std(log_y) > 0
        else float("nan")
    )
    axis.set_xlabel("Predicted observation noise $R$")
    axis.set_ylabel("Observed surprise $e$")
    axis.set_title(title or f"Uncertainty calibration (log correlation={correlation:.3f})")
    axis.grid(color="#dddddd", lw=0.5, alpha=0.8)
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _save_event_alignment(frame: pd.DataFrame, output: Path, title: str | None, window: int) -> None:
    import matplotlib.pyplot as plt

    _style()
    groups = []
    for source, group in frame.groupby("source", sort=False):
        group = group.sort_values("step").reset_index(drop=True)
        event_indices = group.index[group["delta_score"] > 0].tolist()
        for event_index in event_indices:
            event = {}
            for offset in range(-window, window + 1):
                index = event_index + offset
                if 0 <= index < len(group):
                    event[offset] = group.iloc[index]
            groups.append(event)
    if not groups:
        raise ValueError("No positive coverage-delta events found")

    metrics = [
        ("surprise_mean", "surprise $e$"),
        ("k_mean", "$K$"),
        ("write_norm_mean", "effective write"),
        ("p_new_mean", "posterior $P_t$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), constrained_layout=True)
    offsets = np.arange(-window, window + 1)
    for axis, (metric, label) in zip(axes.flat, metrics):
        values = []
        for offset in offsets:
            point_values = [event[offset][metric] for event in groups if offset in event and metric in event]
            values.append(point_values)
        means = np.array([np.nanmean(value) if value else np.nan for value in values])
        errors = np.array([
            np.nanstd(value) / max(1, np.sqrt(len(value))) if value else np.nan
            for value in values
        ])
        axis.plot(offsets, means, color="#1f77b4", lw=1.8)
        axis.fill_between(offsets, means - errors, means + errors, color="#1f77b4", alpha=0.18)
        axis.axvline(0, color="#222222", ls="--", lw=0.8)
        axis.set_title(label)
        axis.set_xlabel("Steps from coverage-gain event")
        axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.8)
    if title:
        fig.suptitle(title, y=1.02, fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _slot_matrices(paths: list[Path], layer: int, fields: list[str]) -> dict[str, np.ndarray]:
    matrices: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for path in paths:
        for row in _load_rows(path):
            payload = _parse_diagnostics(row.get("uncertainty_diagnostics"))
            if not payload:
                continue
            layers = payload.get("layers") or []
            layer_index = layer if layer >= 0 else len(layers) + layer
            if not 0 <= layer_index < len(layers):
                continue
            data = layers[layer_index] or {}
            for field in fields:
                if field in data:
                    values = _first_batch(data[field])
                    matrices[field].append(values)
    return {
        field: np.stack(values) if values else np.empty((0, 0))
        for field, values in matrices.items()
    }


def _save_heatmap(paths: list[Path], output: Path, layer: int, title: str | None) -> None:
    import matplotlib.pyplot as plt

    _style()
    fields = ["k", "p_hat", "r", "write_norm"]
    labels = [
        "Kalman gain $K$",
        "Predicted variance $\\hat P$",
        "Observation noise $R$",
        r"Effective memory write $\|K\odot\Delta M\|$",
    ]
    matrices = _slot_matrices(paths, layer, fields)
    if not any(matrix.size for matrix in matrices.values()):
        raise ValueError("No per-slot uncertainty arrays found")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    for axis, field, label in zip(axes.flat, fields, labels):
        matrix = matrices[field]
        if not matrix.size:
            axis.set_visible(False)
            continue
        image = axis.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel("Evaluation step")
        axis.set_ylabel("Memory slot")
        axis.set_yticks(np.arange(matrix.shape[1]))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, y=1.02, fontsize=11, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", help="One trajectory parquet/jsonl, optionally with a glob")
    parser.add_argument("--eval-dir", type=Path, help="Evaluation output root to search")
    parser.add_argument("--app", help="Restrict --eval-dir search to one app")
    parser.add_argument("--session", type=int, help="Restrict --eval-dir search to one session")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index; -1 is the final layer")
    parser.add_argument("--mode", choices=["dynamics", "calibration", "event", "heatmap"], default="dynamics")
    parser.add_argument("--event-window", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    paths = _find_trajectories(args.trajectory, args.eval_dir, args.app, args.session)
    frame = _diagnostic_rows(paths, args.layer)
    layer_title = f"Layer {int(frame['layer'].iloc[0])}"
    titled_layer = f"{args.title} — {layer_title}" if args.title else layer_title
    if args.mode == "dynamics":
        _save_dynamics(frame, args.output, titled_layer)
    elif args.mode == "calibration":
        _save_calibration(frame, args.output, args.title)
    elif args.mode == "event":
        _save_event_alignment(frame, args.output, args.title, args.event_window)
    else:
        _save_heatmap(paths, args.output, args.layer, titled_layer)
    print(f"Saved {args.mode} plot to {args.output} using {len(frame)} diagnostic steps")


if __name__ == "__main__":
    main()
