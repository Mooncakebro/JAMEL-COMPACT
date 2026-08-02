#!/usr/bin/env python3
"""Draw a CVPR-style per-app maximum-reward comparison for two models.

Example:
    python scripts/plot_two_model_per_app_max.py \
        --base-dir-1 outputs --prefix-1 eval_baseline_4b \
        --name-1 "Baseline (4B)" \
        --base-dir-2 outputs --prefix-2 eval_compact_4b \
        --name-2 "COMPACT (4B)" \
        --output docs/compact_v2/per_app_max_4b.pdf

Each model can point at a different result root.  The script uses the maximum
valid cumulative reward observed for every app across all matching result
folders and sessions.  By default it plots only apps available for both models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate_eval_results import (
    collect_rewards_by_app,
    compute_statistics,
    find_matching_folders,
)


MODEL_1_COLOR = "#8DA0CB"
MODEL_2_COLOR = "#FC8D62"


def _configure_plot_style() -> None:
    """Use the same compact, publication-oriented style as the slide figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _parse_apps(value: str | None) -> list[str] | None:
    if value is None:
        return None
    apps = [app.strip() for app in value.split(",") if app.strip()]
    if not apps:
        raise argparse.ArgumentTypeError("--apps must contain at least one app name")
    if len(set(apps)) != len(apps):
        raise argparse.ArgumentTypeError("--apps cannot contain duplicate app names")
    return apps


def _load_model_maxima(base_dir: str, prefix: str) -> tuple[dict[str, float], list[str], int]:
    folders = find_matching_folders(base_dir, prefix)
    if not folders:
        raise ValueError(
            f"No result folders beginning with '{prefix}' were found under '{base_dir}'."
        )

    rewards_by_app, app_order, loaded_folders, skipped_entries = collect_rewards_by_app(folders)
    if loaded_folders == 0:
        raise ValueError(
            f"Found {len(folders)} matching folder(s) for prefix '{prefix}', but none had readable results."
        )
    if not app_order:
        raise ValueError(
            f"No valid, non-failed rewards were found for prefix '{prefix}'."
        )

    if skipped_entries:
        print(f"[{prefix}] Skipped {skipped_entries} invalid or failed result entry(s).")
    statistics = compute_statistics(rewards_by_app, app_order)
    return {app: float(statistics[app]["max"]) for app in app_order}, app_order, loaded_folders


def _select_apps(
    model_1_order: Sequence[str],
    model_1_maxima: dict[str, float],
    model_2_maxima: dict[str, float],
    requested_apps: list[str] | None,
) -> list[str]:
    common_apps = set(model_1_maxima) & set(model_2_maxima)
    if requested_apps is not None:
        unavailable = [app for app in requested_apps if app not in common_apps]
        if unavailable:
            raise ValueError(
                "Requested app(s) missing valid rewards for one or both models: "
                + ", ".join(unavailable)
            )
        return requested_apps

    apps = [app for app in model_1_order if app in common_apps]
    missing_from_model_1 = sorted(set(model_2_maxima) - set(model_1_maxima))
    missing_from_model_2 = [app for app in model_1_order if app not in model_2_maxima]
    if missing_from_model_1 or missing_from_model_2:
        details = []
        if missing_from_model_1:
            details.append(f"only model 2: {', '.join(missing_from_model_1)}")
        if missing_from_model_2:
            details.append(f"only model 1: {', '.join(missing_from_model_2)}")
        print("[plot] Omitting apps without results for both models (" + "; ".join(details) + ").")
    if not apps:
        raise ValueError("The two models have no apps with valid rewards in common.")
    return apps


def _sort_apps(
    apps: list[str],
    model_1_maxima: dict[str, float],
    model_2_maxima: dict[str, float],
    sort_by: str,
) -> list[str]:
    if sort_by == "input":
        return apps
    if sort_by == "name":
        return sorted(apps)
    if sort_by == "model-1":
        return sorted(apps, key=lambda app: model_1_maxima[app], reverse=True)
    if sort_by == "model-2":
        return sorted(apps, key=lambda app: model_2_maxima[app], reverse=True)
    return sorted(
        apps,
        key=lambda app: model_2_maxima[app] - model_1_maxima[app],
        reverse=True,
    )


def _plot_maxima(
    apps: Sequence[str],
    model_1_maxima: dict[str, float],
    model_2_maxima: dict[str, float],
    model_1_name: str,
    model_2_name: str,
    output_path: Path,
    title: str | None,
    show_values: bool,
) -> None:
    model_1_values = np.asarray([model_1_maxima[app] for app in apps], dtype=float)
    model_2_values = np.asarray([model_2_maxima[app] for app in apps], dtype=float)
    positions = np.arange(len(apps))
    bar_width = 0.36

    figure_width = max(7.5, 0.78 * len(apps))
    fig, axis = plt.subplots(figsize=(figure_width, 3.55))
    bars_1 = axis.bar(
        positions - bar_width / 2,
        model_1_values,
        bar_width,
        color=MODEL_1_COLOR,
        hatch="///",
        edgecolor="#555555",
        linewidth=0.55,
        label=model_1_name,
        zorder=3,
    )
    bars_2 = axis.bar(
        positions + bar_width / 2,
        model_2_values,
        bar_width,
        color=MODEL_2_COLOR,
        edgecolor="white",
        linewidth=0.55,
        label=model_2_name,
        zorder=3,
    )

    if show_values:
        maximum_value = max(float(model_1_values.max()), float(model_2_values.max()))
        label_offset = max(0.02 * maximum_value, 0.05)
        for bars in (bars_1, bars_2):
            for bar in bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + label_offset,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="#222222",
                )

    all_values = np.concatenate((model_1_values, model_2_values))
    minimum_value = float(all_values.min())
    maximum_value = float(all_values.max())
    if minimum_value >= 0:
        lower_limit = 0.0
    else:
        lower_limit = minimum_value - max(0.08 * (maximum_value - minimum_value), 0.5)
    upper_padding = max(0.12 * (maximum_value - lower_limit), 0.5)
    if show_values:
        upper_padding += max(0.04 * maximum_value, 0.1)

    axis.set_ylim(lower_limit, maximum_value + upper_padding)
    axis.set_xticks(positions)
    axis.set_xticklabels(apps, rotation=30, ha="right")
    axis.set_ylabel("Maximum Cumulative Reward", labelpad=7)
    if title:
        axis.set_title(title, fontweight="bold", pad=10)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#666666")
    axis.spines["bottom"].set_color("#666666")
    axis.tick_params(axis="both", length=3, color="#666666")
    axis.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-app maximum cumulative rewards for two evaluated models."
    )
    parser.add_argument("--base-dir-1", required=True, help="Result root for model 1.")
    parser.add_argument("--prefix-1", required=True, help="Result-folder prefix for model 1.")
    parser.add_argument(
        "--name-1",
        default="Model 1",
        help="Display name for model 1 in the legend (independent of its prefix).",
    )
    parser.add_argument("--base-dir-2", required=True, help="Result root for model 2.")
    parser.add_argument("--prefix-2", required=True, help="Result-folder prefix for model 2.")
    parser.add_argument(
        "--name-2",
        default="Model 2",
        help="Display name for model 2 in the legend (independent of its prefix).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("per_app_max_comparison.pdf"),
        help="Figure path (.pdf, .png, etc.; default: per_app_max_comparison.pdf).",
    )
    parser.add_argument(
        "--title",
        default="Per-App Maximum Reward",
        help="Plot title; pass an empty string to omit it.",
    )
    parser.add_argument(
        "--apps",
        type=_parse_apps,
        default=None,
        help="Optional comma-separated app order; each app must exist for both models.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("input", "name", "model-1", "model-2", "delta"),
        default="input",
        help="App ordering when --apps is not provided (default: input).",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Annotate each bar with its maximum reward.",
    )
    args = parser.parse_args()

    _configure_plot_style()
    try:
        model_1_maxima, model_1_order, model_1_folders = _load_model_maxima(
            args.base_dir_1, args.prefix_1
        )
        model_2_maxima, _, model_2_folders = _load_model_maxima(
            args.base_dir_2, args.prefix_2
        )
        apps = _select_apps(model_1_order, model_1_maxima, model_2_maxima, args.apps)
        apps = _sort_apps(apps, model_1_maxima, model_2_maxima, args.sort_by)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    _plot_maxima(
        apps,
        model_1_maxima,
        model_2_maxima,
        args.name_1,
        args.name_2,
        args.output,
        args.title or None,
        args.show_values,
    )
    print(f"[plot] Model 1: {args.name_1} ({model_1_folders} result folder(s), prefix={args.prefix_1})")
    print(f"[plot] Model 2: {args.name_2} ({model_2_folders} result folder(s), prefix={args.prefix_2})")
    print(f"[plot] Compared {len(apps)} app(s): {', '.join(apps)}")
    print(f"[plot] Saved figure to {args.output}")


if __name__ == "__main__":
    main()
