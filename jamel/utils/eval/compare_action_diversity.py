#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VALID_ACTIONS = [
    "noop",
    "send_msg_to_user",
    "report_infeasible",
    "scroll",
    "fill",
    "select_option",
    "click",
    "dblclick",
    "hover",
    "press",
    "focus",
    "clear",
    "drag_and_drop",
    "upload_file",
    "tab_close",
    "tab_focus",
    "new_tab",
    "go_back",
    "go_forward",
    "goto",
]
VALID_SET = set(VALID_ACTIONS)


def _find_last_action_call(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text
    matches: List[Tuple[int, str]] = []
    for name in VALID_ACTIONS:
        idx = 0
        needle = name + "("
        while True:
            pos = lower.find(needle, idx)
            if pos == -1:
                break
            matches.append((pos, name))
            idx = pos + 1
    if not matches:
        return None
    matches.sort()
    start, name = matches[-1]
    i = start + len(name)
    # Expect '(' at i
    if i >= len(text) or text[i] != '(':
        return text[start:].splitlines()[0].strip()
    depth = 0
    end = i
    while end < len(text):
        ch = text[end]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                end += 1
                break
        end += 1
    return text[start:end].strip()


def _normalize_action(action: str) -> str:
    if action is None:
        return ""
    s = action.strip()
    # strip code fences
    if s.startswith("```"):
        s = s.strip("`").strip()
    # Remove surrounding quotes if any
    return s


def _action_type_from_action(action: str) -> Optional[str]:
    if not action:
        return None
    s = action.strip()
    for name in VALID_ACTIONS:
        if s.startswith(name + "(") or s == name + "()":
            return name
    return None


def extract_action(item: Dict) -> Tuple[Optional[str], Optional[str], bool]:
    # Prefer parsing from response, fallback to action field
    response = item.get("response", "")
    action = _find_last_action_call(response)
    if not action:
        action = item.get("action", "")
    action = _normalize_action(action)
    action_type = _action_type_from_action(action)
    valid = action_type in VALID_SET
    return action, action_type, valid


def shannon_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p, 2)
    return h


def simpson_diversity(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    s = 0.0
    for c in counts.values():
        p = c / total
        s += p * p
    return 1.0 - s


def compute_stats(results: List[Dict]) -> Dict:
    actions = []
    action_types = []
    valid_flags = []
    for item in results:
        action, action_type, valid = extract_action(item)
        actions.append(action)
        action_types.append(action_type)
        valid_flags.append(valid)

    valid_actions = [a for a, v in zip(actions, valid_flags) if v]
    valid_action_types = [t for t, v in zip(action_types, valid_flags) if v and t]

    action_type_counts: Dict[str, int] = {}
    for t in valid_action_types:
        action_type_counts[t] = action_type_counts.get(t, 0) + 1

    action_counts: Dict[str, int] = {}
    for a in valid_actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    stats = {
        "total": len(results),
        "valid_total": len(valid_actions),
        "invalid_total": len(results) - len(valid_actions),
        "unique_actions": len(action_counts),
        "unique_action_types": len(action_type_counts),
        "action_type_counts": action_type_counts,
        "action_counts": action_counts,
        "entropy_action_types": shannon_entropy(action_type_counts),
        "simpson_action_types": simpson_diversity(action_type_counts),
    }
    return stats


def render_matplotlib_bar_compare(
    out_path: Path,
    labels: List[str],
    vals_a: List[float],
    vals_b: List[float],
    label_a: str,
    label_b: str,
    title: str,
    rotate: int = 45,
) -> None:
    import matplotlib.pyplot as plt

    x = list(range(len(labels)))
    width = 0.35

    fig_w = max(10, 0.35 * len(labels))
    fig_h = 5 if rotate <= 45 else 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.bar([i - width / 2 for i in x], vals_a, width, label=label_a, color="#2b6cb0")
    ax.bar([i + width / 2 for i in x], vals_b, width, label=label_b, color="#c05621")

    ax.set_title(title)
    ax.set_ylabel("Fraction of valid actions")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right")
    ax.legend()
    ax.set_ylim(0, max([1.0] + vals_a + vals_b) * 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path("outputs")
    file_a = base / "single_test_results.json"
    file_b = base / "single_test_results_qwen3.json"

    data_a = json.loads(file_a.read_text())
    data_b = json.loads(file_b.read_text())

    res_a = data_a["results"]
    res_b = data_b["results"]

    stats_a = compute_stats(res_a)
    stats_b = compute_stats(res_b)

    # Build aligned distribution over union of action types
    labels = sorted(set(stats_a["action_type_counts"]) | set(stats_b["action_type_counts"]))
    vals_a = []
    vals_b = []
    for label in labels:
        ca = stats_a["action_type_counts"].get(label, 0)
        cb = stats_b["action_type_counts"].get(label, 0)
        va = ca / stats_a["valid_total"] if stats_a["valid_total"] else 0
        vb = cb / stats_b["valid_total"] if stats_b["valid_total"] else 0
        vals_a.append(va)
        vals_b.append(vb)

    # Write summary JSON
    summary = {
        "model_a": {
            "file": str(file_a),
            **stats_a,
        },
        "model_b": {
            "file": str(file_b),
            **stats_b,
        },
        "distribution_action_type_fraction": {
            "labels": labels,
            "model_a": vals_a,
            "model_b": vals_b,
        },
    }
    (base / "action_diversity_compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write CSV for action type distribution
    lines = ["action_type,model_a_fraction,model_b_fraction,model_a_count,model_b_count"]
    for label, va, vb in zip(labels, vals_a, vals_b):
        ca = stats_a["action_type_counts"].get(label, 0)
        cb = stats_b["action_type_counts"].get(label, 0)
        lines.append(f"{label},{va:.6f},{vb:.6f},{ca},{cb}")
    (base / "action_diversity_compare.csv").write_text("\n".join(lines), encoding="utf-8")

    # Matplotlib plots
    render_matplotlib_bar_compare(
        base / "action_type_distribution_compare.png",
        labels,
        vals_a,
        vals_b,
        label_a="single_test_results",
        label_b="single_test_results_qwen3",
        title="Action Type Distribution (Valid Actions Only)",
        rotate=30,
    )

    # Full action distribution (treat different bids as different actions)
    action_labels = sorted(set(stats_a["action_counts"]) | set(stats_b["action_counts"]))
    action_vals_a = []
    action_vals_b = []
    for label in action_labels:
        ca = stats_a["action_counts"].get(label, 0)
        cb = stats_b["action_counts"].get(label, 0)
        va = ca / stats_a["valid_total"] if stats_a["valid_total"] else 0
        vb = cb / stats_b["valid_total"] if stats_b["valid_total"] else 0
        action_vals_a.append(va)
        action_vals_b.append(vb)

    render_matplotlib_bar_compare(
        base / "action_distribution_compare.png",
        action_labels,
        action_vals_a,
        action_vals_b,
        label_a="single_test_results",
        label_b="single_test_results_qwen3",
        title="Action Distribution (Exact Action Strings, Valid Only)",
        rotate=60,
    )


if __name__ == "__main__":
    main()
