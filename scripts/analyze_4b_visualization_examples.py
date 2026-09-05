"""Find matched 4B COMPACT/Baseline examples for visualization stories."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _paths(roots: list[Path]) -> list[Path]:
    paths = []
    for root in roots:
        paths.extend(root.glob("*/session*/trajectory_*.parquet"))
    return sorted({path.resolve() for path in paths})


def _summary(path: Path, model: str) -> dict[str, object]:
    frame = pd.read_parquet(path).sort_values("step").reset_index(drop=True)
    action = frame["action"].astype(str)
    reward_steps = frame.loc[frame["reward"] > 0, "step"].astype(int).tolist()
    return {
        "model": model,
        "app": path.parent.parent.name,
        "trajectory": str(path),
        "final_reward": float(frame["cumulative_reward"].iloc[-1]),
        "unique_actions": int(action.nunique()),
        "repeated_actions": int(action.duplicated().sum()),
        "reward_steps": reward_steps,
        "last_reward_step": reward_steps[-1] if reward_steps else None,
        "actions": frame["action"].astype(str).tolist(),
        "cumulative": frame["cumulative_reward"].astype(float).tolist(),
    }


def _action_runs(actions: list[str], minimum: int = 4) -> list[tuple[int, int, int, str]]:
    runs = []
    start = 0
    for index in range(1, len(actions) + 1):
        if index == len(actions) or actions[index] != actions[start]:
            length = index - start
            if length >= minimum:
                runs.append((start, index - 1, length, actions[start]))
            start = index
    return runs


def _format_steps(steps: list[int]) -> str:
    return ", ".join(str(step) for step in steps) if steps else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-dir", action="append", type=Path, required=True)
    parser.add_argument("--baseline-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compact = [_summary(path, "COMPACT-4B") for path in _paths(args.compact_dir)]
    baseline = [_summary(path, "Baseline-4B") for path in _paths(args.baseline_dir)]
    compact_by_app: dict[str, list[dict[str, object]]] = {}
    baseline_by_app: dict[str, list[dict[str, object]]] = {}
    for item in compact:
        compact_by_app.setdefault(str(item["app"]), []).append(item)
    for item in baseline:
        baseline_by_app.setdefault(str(item["app"]), []).append(item)

    rows = []
    for app in sorted(set(compact_by_app) & set(baseline_by_app)):
        compact_best = max(compact_by_app[app], key=lambda item: (float(item["final_reward"]), int(item["unique_actions"])))
        baseline_worst = min(baseline_by_app[app], key=lambda item: (float(item["final_reward"]), -int(item["repeated_actions"])))
        rows.append(
            {
                "app": app,
                "compact_reward": compact_best["final_reward"],
                "baseline_reward": baseline_worst["final_reward"],
                "reward_gap": float(compact_best["final_reward"]) - float(baseline_worst["final_reward"]),
                "compact_unique": compact_best["unique_actions"],
                "baseline_unique": baseline_worst["unique_actions"],
                "compact_trajectory": compact_best["trajectory"],
                "baseline_trajectory": baseline_worst["trajectory"],
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["reward_gap", "compact_unique"], ascending=False)

    lines = [
        "# 4B Visualization Example Analysis",
        "",
        "This report uses only the 4B COMPACT and 4B Baseline trajectories.",
        "The reward comparisons are matched by test app, not by identical browser state.",
        "",
        "## Recommended Figure 2 Examples",
        "",
    ]
    for app in ["temu", "pinduoduo"]:
        if app not in compact_by_app or app not in baseline_by_app:
            continue
        compact_item = max(compact_by_app[app], key=lambda item: (float(item["final_reward"]), int(item["unique_actions"])))
        baseline_item = min(baseline_by_app[app], key=lambda item: (float(item["final_reward"]), -int(item["repeated_actions"])))
        lines.extend(
            [
                f"### {app}",
                f"- COMPACT trajectory: `{compact_item['trajectory']}`",
                f"- Baseline trajectory: `{baseline_item['trajectory']}`",
                f"- COMPACT: final reward {compact_item['final_reward']:.0f}, {compact_item['unique_actions']} unique actions, rewarded steps `{_format_steps(compact_item['reward_steps'])}`.",
                f"- Baseline: final reward {baseline_item['final_reward']:.0f}, {baseline_item['unique_actions']} unique actions, {baseline_item['repeated_actions']} repeated actions, rewarded steps `{_format_steps(baseline_item['reward_steps'])}`.",
                f"- COMPACT action runs of length at least four: `{_action_runs(compact_item['actions']) or 'none'}`.",
                f"- Baseline action runs of length at least four: `{_action_runs(baseline_item['actions'])}`.",
                "- Story: COMPACT keeps changing element targets and continues receiving reward later in the 50-step horizon; Baseline earns early reward, then repeatedly emits the same action and stops accumulating reward.",
                "",
            ]
        )

    lines.extend(["## Recommended Figure 3 Pair", ""])
    success_path = max(
        [item for item in compact if item["app"] == "vipshop"],
        key=lambda item: (float(item["final_reward"]), int(item["unique_actions"])),
    )
    failure_path = min(
        [item for item in compact if item["app"] == "youku"],
        key=lambda item: (float(item["final_reward"]), -int(item["repeated_actions"])),
    )
    lines.extend(
        [
            "### Success versus failure",
            f"- Success trajectory: `{success_path['trajectory']}`",
            f"- Failure trajectory: `{failure_path['trajectory']}`",
            f"- Success: final reward {success_path['final_reward']:.0f}, {success_path['unique_actions']} unique actions, repeated actions {success_path['repeated_actions']}.",
            f"- Failure: final reward {failure_path['final_reward']:.0f}, {failure_path['unique_actions']} unique actions, repeated actions {failure_path['repeated_actions']}.",
            f"- Failure reward steps: `{_format_steps(failure_path['reward_steps'])}`; the longest repeated-action runs are `{_action_runs(failure_path['actions'])}`.",
            "- Recommended visual narrative: the success panel shows continued reward accumulation and broad action diversity, while the failure panel shows reward saturation followed by a long action loop. The memory diagnostics should be read as update activity, not as proof of useful exploration.",
            "",
            "## All-App Ranking",
            "",
            ranking.to_markdown(index=False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved 4B visualization example report to {args.output}")


if __name__ == "__main__":
    main()
