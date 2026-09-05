"""Summarize every app/session trajectory in selected evaluation roots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _model_name(root: Path) -> str:
    return "Baseline-4B" if root.name.startswith("baseline_4b") else "COMPACT-4B"


def _status(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "empty"
    if bool(frame["terminated"].fillna(False).any()):
        return "terminated"
    if bool(frame["truncated"].fillna(False).any()):
        return "truncated"
    return "completed"


def _summarize(root: Path, trajectory: Path) -> dict[str, object]:
    frame = pd.read_parquet(trajectory).sort_values("step").reset_index(drop=True)
    action = frame["action"].fillna("").astype(str) if "action" in frame else pd.Series(dtype=str)
    rewards = frame["reward"].fillna(0.0).astype(float) if "reward" in frame else pd.Series(dtype=float)
    cumulative = (
        frame["cumulative_reward"].fillna(0.0).astype(float)
        if "cumulative_reward" in frame
        else rewards.cumsum()
    )
    coverage = frame["current_score"].fillna(0.0).astype(float) if "current_score" in frame else pd.Series(dtype=float)
    return {
        "model": _model_name(root),
        "result_root": root.name,
        "app": trajectory.parent.parent.name,
        "session": trajectory.parent.name,
        "steps": int(len(frame)),
        "total_reward": float(rewards.sum()) if len(rewards) else 0.0,
        "final_cumulative_reward": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "rewarded_steps": int((rewards > 0).sum()) if len(rewards) else 0,
        "first_reward_step": int(frame.loc[rewards > 0, "step"].iloc[0]) if len(frame) and bool((rewards > 0).any()) else "-",
        "last_reward_step": int(frame.loc[rewards > 0, "step"].iloc[-1]) if len(frame) and bool((rewards > 0).any()) else "-",
        "final_coverage_score": float(coverage.iloc[-1]) if len(coverage) else "-",
        "unique_actions": int(action.nunique()) if len(action) else 0,
        "repeated_actions": int(action.duplicated().sum()) if len(action) else 0,
        "page_changes": int(frame["page_changed"].fillna(False).astype(bool).sum()) if "page_changed" in frame else "-",
        "status": _status(frame),
        "trajectory": str(trajectory),
    }


def _markdown(frame: pd.DataFrame) -> str:
    columns = [
        "model", "result_root", "app", "session", "steps", "total_reward",
        "rewarded_steps", "first_reward_step", "last_reward_step",
        "final_coverage_score", "unique_actions", "repeated_actions",
        "page_changes", "status",
    ]
    display = frame[columns].copy()
    display["total_reward"] = display["total_reward"].map(lambda value: f"{value:.0f}")
    display.columns = [
        "Model", "Result root", "App", "Session", "Steps", "Reward",
        "Rewarded steps", "First reward", "Last reward", "Final coverage",
        "Unique actions", "Repeated actions", "Page changes", "Status",
    ]
    lines = [
        "# 4B Evaluation Session Scoreboard",
        "",
        "This table covers the current 4B result roots supplied to the summarizer. Each row is one app/session trajectory.",
        "The score is cumulative reward; `Final coverage` is included only as auxiliary context because coverage values can be much larger than reward.",
        "",
        display.to_markdown(index=False),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for root in args.root:
        paths = sorted(root.glob("*/session*/trajectory_*.parquet"))
        if not paths:
            print(f"[summary] no trajectories found in {root}")
        for trajectory in paths:
            records.append(_summarize(root, trajectory))
    if not records:
        raise SystemExit("No trajectories found")
    frame = pd.DataFrame(records).sort_values(["model", "result_root", "app", "session"]).reset_index(drop=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv_output, index=False)
    args.markdown_output.write_text(_markdown(frame), encoding="utf-8")
    print(f"Saved {len(frame)} session rows to {args.csv_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()
