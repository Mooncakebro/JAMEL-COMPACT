import json
from pathlib import Path


def _load_total(summary_path: Path) -> dict:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data.get("total", {})


def build_trend(report_root: Path) -> list[dict]:
    entries = []
    prev = None
    for idx, summary_path in enumerate(sorted(report_root.glob("*/coverage-summary.json")), start=1):
        timestamp_folder = summary_path.parent.name
        total = _load_total(summary_path)
        lines = total.get("lines", {})
        statements = total.get("statements", {})
        functions = total.get("functions", {})
        branches = total.get("branches", {})

        row = {
            "step": idx,
            "timestamp_folder": timestamp_folder,
            "lines_total": lines.get("total", 0),
            "lines_covered": lines.get("covered", 0),
            "lines_pct": lines.get("pct", 0),
            "statements_total": statements.get("total", 0),
            "statements_covered": statements.get("covered", 0),
            "statements_pct": statements.get("pct", 0),
            "functions_total": functions.get("total", 0),
            "functions_covered": functions.get("covered", 0),
            "functions_pct": functions.get("pct", 0),
            "branches_total": branches.get("total", 0),
            "branches_covered": branches.get("covered", 0),
            "branches_pct": branches.get("pct", 0),
        }

        if prev is None:
            row.update(
                {
                    "lines_covered_delta": 0,
                    "lines_pct_delta": 0,
                    "statements_covered_delta": 0,
                    "statements_pct_delta": 0,
                    "functions_covered_delta": 0,
                    "functions_pct_delta": 0,
                    "branches_covered_delta": 0,
                    "branches_pct_delta": 0,
                }
            )
        else:
            row.update(
                {
                    "lines_covered_delta": row["lines_covered"] - prev["lines_covered"],
                    "lines_pct_delta": row["lines_pct"] - prev["lines_pct"],
                    "statements_covered_delta": row["statements_covered"] - prev["statements_covered"],
                    "statements_pct_delta": row["statements_pct"] - prev["statements_pct"],
                    "functions_covered_delta": row["functions_covered"] - prev["functions_covered"],
                    "functions_pct_delta": row["functions_pct"] - prev["functions_pct"],
                    "branches_covered_delta": row["branches_covered"] - prev["branches_covered"],
                    "branches_pct_delta": row["branches_pct"] - prev["branches_pct"],
                }
            )

        entries.append(row)
        prev = row
    return entries


def write_outputs(trend: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "coverage_trend_from_reports.json"
    json_path.write_text(json.dumps(trend, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "coverage_trend_from_reports.csv"
    headers = [
        "step",
        "timestamp_folder",
        "lines_total",
        "lines_covered",
        "lines_pct",
        "lines_covered_delta",
        "lines_pct_delta",
        "statements_total",
        "statements_covered",
        "statements_pct",
        "statements_covered_delta",
        "statements_pct_delta",
        "functions_total",
        "functions_covered",
        "functions_pct",
        "functions_covered_delta",
        "functions_pct_delta",
        "branches_total",
        "branches_covered",
        "branches_pct",
        "branches_covered_delta",
        "branches_pct_delta",
    ]
    csv_lines = [",".join(headers)]
    for row in trend:
        csv_lines.append(",".join(str(row[h]) for h in headers))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_path = output_dir / "coverage_trend_from_reports.md"
    md_lines = [
        "| Step | Timestamp | Lines Covered | Lines Delta | Statements Covered | Statements Delta | Functions Covered | Functions Delta | Branches Covered | Branches Delta |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in trend:
        md_lines.append(
            f"| {row['step']} | {row['timestamp_folder']} | {row['lines_covered']} | "
            f"{row['lines_covered_delta']} | {row['statements_covered']} | "
            f"{row['statements_covered_delta']} | {row['functions_covered']} | "
            f"{row['functions_covered_delta']} | {row['branches_covered']} | "
            f"{row['branches_covered_delta']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def plot_trend_matplotlib(trend: list[dict], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    if not trend:
        out_path = output_dir / "coverage_trend_from_reports.png"
        return out_path

    steps = [row["step"] for row in trend]
    cumulative_series = [
        # ("Lines", [row["lines_covered"] for row in trend]),
        # ("Statements", [row["statements_covered"] for row in trend]),
        # ("Functions", [row["functions_covered"] for row in trend]),
        ("Branches", [row["branches_covered"] for row in trend]),
    ]
    delta_series = [
        # ("Lines", [row["lines_covered_delta"] for row in trend]),
        # ("Statements", [row["statements_covered_delta"] for row in trend]),
        # ("Functions", [row["functions_covered_delta"] for row in trend]),
        ("Branches", [row["branches_covered_delta"] for row in trend]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    for name, values in cumulative_series:
        axes[0].plot(steps, values, marker="o", linewidth=2, label=name)
    axes[0].set_title("Cumulative Covered Counts")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Covered Count")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    for name, values in delta_series:
        axes[1].plot(steps, values, marker="o", linewidth=2, label=name)
    axes[1].set_title("Delta Covered Counts")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Delta Covered Count")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Coverage Trend", y=1.02)
    fig.tight_layout()

    out_path = output_dir / "coverage_trend_from_reports.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main() -> None:
    report_root = Path("data/0316/coverage-report")
    output_dir = Path("data/0316/coverage-report")
    trend = build_trend(report_root)
    write_outputs(trend, output_dir)
    plot_trend_matplotlib(trend, output_dir)


if __name__ == "__main__":
    main()
