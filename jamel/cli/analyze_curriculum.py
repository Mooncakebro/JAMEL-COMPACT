"""Analyze reward and last_action_error trends across stage iterations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime
    pd = None  # type: ignore[assignment]


ITERATION_DIR_PATTERN = re.compile(r"iteration_(\d+)$")
TREND_EPSILON = 1e-9


@dataclass
class IterationMetrics:
    iteration: int
    trajectory_count: int
    step_count: int
    total_reward: float
    mean_reward: float
    positive_reward_rate: float
    last_action_error_rate: float
    reward_delta: float | None
    error_rate_delta: float | None


@dataclass
class TrendAnalysis:
    metric: str
    expected_direction: str
    start_value: float
    end_value: float
    net_change: float
    slope_per_iteration: float
    improving_transitions: int
    total_transitions: int
    monotonic: bool
    verdict: str


@dataclass
class SkippedIteration:
    iteration: int
    path: str
    reason: str


def _require_pandas() -> Any:
    if pd is None:
        raise SystemExit(
            "This command requires pandas/pyarrow. Install project dependencies first."
        )
    return pd


def _iter_iteration_dirs(stage_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for child in stage_dir.iterdir():
        if not child.is_dir():
            continue
        match = ITERATION_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        items.append((int(match.group(1)), child))
    return sorted(items, key=lambda item: item[0])


def _is_last_action_error(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False

    text = str(value).strip()
    if not text:
        return False
    return "Success." not in text


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _linear_slope(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0

    xs = list(range(len(values)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _analyze_metric_trend(
    metric: str,
    values: list[float],
    expected_direction: str,
) -> TrendAnalysis:
    if not values:
        raise ValueError(f"No values available for metric: {metric}")

    sign = 1.0 if expected_direction == "increase" else -1.0
    transitions = [sign * (curr - prev) for prev, curr in zip(values, values[1:])]
    monotonic = all(delta >= -TREND_EPSILON for delta in transitions)
    improving_transitions = sum(delta > TREND_EPSILON for delta in transitions)
    total_transitions = len(transitions)
    slope = _linear_slope(values)
    directed_slope = sign * slope
    net_change = values[-1] - values[0]
    directed_net_change = sign * net_change

    if total_transitions == 0:
        verdict = "insufficient_iterations"
    elif monotonic and improving_transitions == total_transitions:
        verdict = "strictly_improving"
    elif monotonic and improving_transitions > 0:
        verdict = "non_worsening"
    elif directed_net_change > TREND_EPSILON and directed_slope > TREND_EPSILON:
        verdict = "overall_improving_with_noise"
    else:
        verdict = "no_clear_improvement"

    return TrendAnalysis(
        metric=metric,
        expected_direction=expected_direction,
        start_value=values[0],
        end_value=values[-1],
        net_change=net_change,
        slope_per_iteration=slope,
        improving_transitions=improving_transitions,
        total_transitions=total_transitions,
        monotonic=monotonic,
        verdict=verdict,
    )


def _load_iteration_metrics(iteration: int, history_files: list[Path]) -> IterationMetrics:
    pandas = _require_pandas()

    total_reward = 0.0
    reward_steps = 0
    positive_reward_steps = 0
    error_steps = 0

    for history_file in history_files:
        df = pandas.read_parquet(history_file)
        if "reward" not in df.columns:
            raise SystemExit(f"Missing `reward` column in: {history_file}")

        reward_series = pandas.to_numeric(df["reward"], errors="coerce").fillna(0.0)
        rewards = reward_series.tolist()
        total_reward += float(sum(rewards))
        reward_steps += len(rewards)
        positive_reward_steps += sum(reward > 0 for reward in rewards)

        if "after_last_action_error" in df.columns:
            error_steps += sum(_is_last_action_error(value) for value in df["after_last_action_error"].tolist())

    return IterationMetrics(
        iteration=iteration,
        trajectory_count=len(history_files),
        step_count=reward_steps,
        total_reward=total_reward,
        mean_reward=_safe_rate(total_reward, reward_steps),
        positive_reward_rate=_safe_rate(positive_reward_steps, reward_steps),
        last_action_error_rate=_safe_rate(error_steps, reward_steps),
        reward_delta=None,
        error_rate_delta=None,
    )


def analyze_stage(stage_dir: str | Path) -> dict[str, Any]:
    stage_path = Path(stage_dir).expanduser().resolve()
    if not stage_path.exists():
        raise SystemExit(f"Stage directory does not exist: {stage_path}")
    if not stage_path.is_dir():
        raise SystemExit(f"Stage path is not a directory: {stage_path}")

    iteration_dirs = _iter_iteration_dirs(stage_path)
    if not iteration_dirs:
        raise SystemExit(f"No iteration directories found in: {stage_path}")

    metrics: list[IterationMetrics] = []
    skipped_iterations: list[SkippedIteration] = []
    for iteration, path in iteration_dirs:
        history_files = sorted((path / "histories").glob("*.parquet"))
        if not history_files:
            skipped_iterations.append(
                SkippedIteration(
                    iteration=iteration,
                    path=str(path),
                    reason="missing_histories",
                )
            )
            continue
        metrics.append(_load_iteration_metrics(iteration, history_files))

    if not metrics:
        raise SystemExit(f"No complete iteration histories found in: {stage_path}")

    for idx in range(1, len(metrics)):
        metrics[idx].reward_delta = metrics[idx].mean_reward - metrics[idx - 1].mean_reward
        metrics[idx].error_rate_delta = (
            metrics[idx].last_action_error_rate - metrics[idx - 1].last_action_error_rate
        )

    reward_values = [item.mean_reward for item in metrics]
    error_values = [item.last_action_error_rate for item in metrics]

    return {
        "stage_dir": str(stage_path),
        "iteration_count": len(metrics),
        "iterations": [asdict(item) for item in metrics],
        "skipped_iterations": [asdict(item) for item in skipped_iterations],
        "trends": {
            "reward_mean": asdict(
                _analyze_metric_trend(
                    metric="reward_mean",
                    values=reward_values,
                    expected_direction="increase",
                )
            ),
            "last_action_error_rate": asdict(
                _analyze_metric_trend(
                    metric="last_action_error_rate",
                    values=error_values,
                    expected_direction="decrease",
                )
            ),
        },
    }


def _format_ratio(value: float) -> str:
    return f"{value:.4f}"


def _format_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _format_skipped_iterations(items: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[int]] = {}
    for item in items:
        grouped.setdefault(item["reason"], []).append(int(item["iteration"]))

    parts: list[str] = []
    for reason, iterations in grouped.items():
        sorted_iterations = sorted(iterations)
        range_start = sorted_iterations[0]
        range_end = sorted_iterations[0]

        for iteration in sorted_iterations[1:]:
            if iteration == range_end + 1:
                range_end = iteration
                continue
            parts.append(_format_skipped_range(range_start, range_end, reason))
            range_start = iteration
            range_end = iteration

        parts.append(_format_skipped_range(range_start, range_end, reason))

    return ", ".join(parts)


def _format_skipped_range(start: int, end: int, reason: str) -> str:
    if start == end:
        return f"{start}({reason})"
    return f"{start}-{end}({reason})"


def _render_text_report(result: dict[str, Any]) -> str:
    lines = [
        f"Stage: {result['stage_dir']}",
        f"Iterations: {result['iteration_count']}",
    ]
    if result["skipped_iterations"]:
        skipped_text = _format_skipped_iterations(result["skipped_iterations"])
        lines.append(f"Skipped: {skipped_text}")
    lines.extend(
        [
            "",
            "iteration | trajectories | steps | mean_reward | positive_reward_rate | last_action_error_rate | reward_delta | error_rate_delta",
            "-" * 120,
        ]
    )

    for item in result["iterations"]:
        lines.append(
            " | ".join(
                [
                    str(item["iteration"]),
                    str(item["trajectory_count"]),
                    str(item["step_count"]),
                    _format_ratio(item["mean_reward"]),
                    _format_ratio(item["positive_reward_rate"]),
                    _format_ratio(item["last_action_error_rate"]),
                    _format_delta(item["reward_delta"]),
                    _format_delta(item["error_rate_delta"]),
                ]
            )
        )

    lines.append("")
    for trend_name, trend in result["trends"].items():
        lines.append(
            (
                f"{trend_name}: verdict={trend['verdict']}, "
                f"start={trend['start_value']:.4f}, end={trend['end_value']:.4f}, "
                f"net_change={trend['net_change']:+.4f}, "
                f"slope_per_iteration={trend['slope_per_iteration']:+.4f}, "
                f"improving_transitions={trend['improving_transitions']}/{trend['total_transitions']}, "
                f"monotonic={trend['monotonic']}"
            )
        )

    return "\n".join(lines)


def main(args) -> None:
    result = analyze_stage(args.stage_dir)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(_render_text_report(result))


__all__ = [
    "analyze_stage",
    "main",
]
