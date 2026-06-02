from __future__ import annotations

import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from jamel.core.reward.web.utils import compute_monocart_coverage_reward_details


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _extract_action(row: pd.Series) -> str:
    parsed = row.get("parsed_content")
    if isinstance(parsed, dict):
        action = parsed.get("action")
        if action:
            return str(action)
    raw = row.get("raw_content")
    if raw:
        return str(raw)
    return ""


def _extract_extra_fields(row: pd.Series) -> dict[str, Any]:
    value = row.get("extra_fields")
    if isinstance(value, dict):
        return value
    return {}


def _image_data_uri(image_bytes: bytes | None, max_side: int) -> tuple[str | None, str]:
    if not image_bytes:
        return None, ""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGBA")
            if max_side > 0:
                image.thumbnail((max_side, max_side))
            width, height = image.size
            buf = io.BytesIO()
            image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}", f"{width}x{height}"
    except Exception:
        return None, ""


def _extract_observation_text(text: Any, limit: int) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    if not value:
        return ""
    return _shorten(value, limit)


def _extract_executed_intervals(snapshot: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    scripts: dict[str, dict[str, Any]] = {}
    for script in snapshot:
        url = str(script.get("url") or "")
        source = str(script.get("source") or "")
        script_id = str(script.get("scriptId") or "")
        script_key = f"{url}::{script_id}"
        intervals: list[tuple[int, int]] = []
        executed_functions = 0
        for fn in script.get("functions") or []:
            ranges = fn.get("ranges") or []
            has_execution = False
            for item in ranges:
                count = int(item.get("count") or 0)
                if count <= 0:
                    continue
                start = int(item.get("startOffset") or 0)
                end = int(item.get("endOffset") or 0)
                if end <= start:
                    continue
                intervals.append((start, end))
                has_execution = True
            if has_execution:
                executed_functions += 1
        scripts[script_key] = {
            "url": url,
            "script_id": script_id,
            "source": source,
            "intervals": _merge_intervals(intervals),
            "executed_functions": executed_functions,
        }
    return scripts


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
            continue
        merged.append((start, end))
    return merged


def _subtract_intervals(
    current: list[tuple[int, int]],
    baseline: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not current:
        return []
    if not baseline:
        return list(current)
    result: list[tuple[int, int]] = []
    baseline = _merge_intervals(baseline)
    for start, end in current:
        cursor = start
        for base_start, base_end in baseline:
            if base_end <= cursor:
                continue
            if base_start >= end:
                break
            if base_start > cursor:
                result.append((cursor, min(base_start, end)))
            cursor = max(cursor, base_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def _interval_total_length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _coverage_delta(baseline_paths: Path | list[Path] | None, current_path: Path | None) -> dict[str, Any]:
    if baseline_paths is None:
        baseline_path_list: list[Path] = []
    elif isinstance(baseline_paths, Path):
        baseline_path_list = [baseline_paths]
    else:
        baseline_path_list = list(baseline_paths)
    details = compute_monocart_coverage_reward_details(
        current_path=current_path,
        baseline_paths=baseline_path_list,
    )
    file_counts = {
        "baseline_files": len(baseline_path_list),
        "current_files": len(baseline_path_list) + (1 if current_path is not None else 0),
    }
    if details.get("skip_reason"):
        return {"error": details["skip_reason"], **details, **file_counts}
    return {**details, **file_counts}

def _build_snippets(source: str, intervals: list[tuple[int, int]], radius: int = 80) -> list[dict[str, Any]]:
    snippets = []
    for start, end in intervals[:8]:
        left = max(0, start - radius)
        right = min(len(source), end + radius)
        snippet = source[left:right].replace("\x00", "")
        snippets.append(
            {
                "start": start,
                "end": end,
                "text": snippet,
            }
        )
    return snippets


@dataclass
class StepAudit:
    step: int
    reward: float
    action: str
    target_url: str
    coverage_path: str
    before_obs: str
    after_obs: str
    before_image_uri: str | None
    before_image_size: str
    after_image_uri: str | None
    after_image_size: str
    delta: dict[str, Any]

    @property
    def raw_delta_bytes(self) -> int:
        if not isinstance(self.delta, dict):
            return 0
        return int(self.delta.get("delta_score", 0) or 0)

    @property
    def suspicious_reward(self) -> bool:
        return self.reward > 0 and self.raw_delta_bytes == 0


def _load_step_audits(parquet_path: Path, image_max_side: int, obs_limit: int) -> tuple[list[StepAudit], dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    step_audits: list[StepAudit] = []
    baseline_coverage_paths: list[Path] = []
    metadata = {
        "parquet_path": str(parquet_path),
        "num_rows": int(len(df)),
    }
    for _, row in df.iterrows():
        extra_fields = _extract_extra_fields(row)
        coverage_path_str = str(extra_fields.get("coverage_path") or "")
        coverage_path = Path(coverage_path_str) if coverage_path_str else None
        before_image_uri, before_image_size = _image_data_uri(row.get("before_screenshot"), image_max_side)
        after_image_uri, after_image_size = _image_data_uri(row.get("after_screenshot"), image_max_side)
        step_audits.append(
            StepAudit(
                step=int(row.get("step", 0)),
                reward=float(row.get("reward", 0.0) or 0.0),
                action=_extract_action(row),
                target_url=str(extra_fields.get("target_url") or ""),
                coverage_path=coverage_path_str,
                before_obs=_extract_observation_text(row.get("before_observation_str"), obs_limit),
                after_obs=_extract_observation_text(row.get("after_observation_str"), obs_limit),
                before_image_uri=before_image_uri,
                before_image_size=before_image_size,
                after_image_uri=after_image_uri,
                after_image_size=after_image_size,
                delta=(
                    {
                        "previous_score": int(extra_fields.get("coverage_previous_score", 0) or 0),
                        "current_score": int(extra_fields.get("coverage_current_score", 0) or 0),
                        "delta_score": int(extra_fields.get("coverage_delta_score", 0) or 0),
                        "baseline_files": len(baseline_coverage_paths),
                        "current_files": len(baseline_coverage_paths) + 1,
                    }
                    if "coverage_delta_score" in extra_fields
                    else _coverage_delta(baseline_coverage_paths, coverage_path)
                ),
            )
        )
        if coverage_path is not None:
            baseline_coverage_paths.append(coverage_path)
    if step_audits:
        metadata["target_url"] = step_audits[0].target_url
    return step_audits, metadata


def _render_script_details(script: dict[str, Any]) -> str:
    snippets = []
    for snippet in script["snippets"]:
        snippets.append(
            "<div class='snippet'>"
            f"<div class='snippet-meta'>offset [{snippet['start']}, {snippet['end']})</div>"
            f"<pre>{_escape(snippet['text'])}</pre>"
            "</div>"
        )
    snippet_html = "".join(snippets) if snippets else "<div class='muted'>No source snippet</div>"
    return (
        "<details class='script-card'>"
        f"<summary><span class='script-url'>{_escape(script['url'])}</span>"
        f"<span class='script-metrics'>new_bytes={script['new_bytes']} | new_intervals={script['new_interval_count']}</span></summary>"
        f"{snippet_html}"
        "</details>"
    )


def _render_step(step: StepAudit) -> str:
    delta = step.delta
    if "error" in delta:
        delta_html = f"<div class='delta-error'>{_escape(delta['error'])}</div>"
    else:
        delta_html = (
            "<div class=\"delta-summary\">",
            "<span>previous_score={}</span>".format(delta.get("previous_score", 0)),
            "<span>current_score={}</span>".format(delta.get("current_score", 0)),
            "<span>delta_score={}</span>".format(delta.get("delta_score", 0)),
            "<span>baseline_files={}</span>".format(delta.get("baseline_files", 0)),
            "</div>"
        )

    before_image = (
        f"<img src='{step.before_image_uri}' alt='before screenshot'/>"
        f"<div class='img-meta'>{_escape(step.before_image_size)}</div>"
        if step.before_image_uri
        else "<div class='img-missing'>No image</div>"
    )
    after_image = (
        f"<img src='{step.after_image_uri}' alt='after screenshot'/>"
        f"<div class='img-meta'>{_escape(step.after_image_size)}</div>"
        if step.after_image_uri
        else "<div class='img-missing'>No image</div>"
    )

    reward_class = "reward-positive" if step.reward > 0 else "reward-zero"
    suspicious_html = (
        "<div class='warning-badge'>reward>0 but Monocart delta is 0</div>"
        if step.suspicious_reward
        else ""
    )
    return (
        "<section class='step-card'>"
        f"<div class='step-header'><div class='step-title'>step {step.step}</div>"
        f"<div class='reward {reward_class}'>reward={step.reward:.1f}</div></div>"
        f"{suspicious_html}"
        f"<div class='meta-row'><span>action={_escape(step.action)}</span>"
        f"<span>coverage={_escape(step.coverage_path)}</span></div>"
        "<div class='image-grid'>"
        f"<div><div class='panel-title'>Before</div>{before_image}</div>"
        f"<div><div class='panel-title'>After</div>{after_image}</div>"
        "</div>"
        "<div class='obs-grid'>"
        f"<div><div class='panel-title'>Before observation</div><pre>{_escape(step.before_obs)}</pre></div>"
        f"<div><div class='panel-title'>After observation</div><pre>{_escape(step.after_obs)}</pre></div>"
        "</div>"
        "<div class='panel-title'>Monocart coverage score delta vs cumulative history</div>"
        f"{delta_html}"
        "</section>"
    )


def _render_html(step_audits: list[StepAudit], metadata: dict[str, Any]) -> str:
    positive_steps = sum(1 for step in step_audits if step.reward > 0)
    total_new_bytes = sum(step.delta.get("delta_score", 0) for step in step_audits if isinstance(step.delta, dict))
    suspicious_steps = sum(1 for step in step_audits if step.suspicious_reward)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Coverage Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #111827; color: #e5e7eb; margin: 0; padding: 24px; }}
    .container {{ max-width: 1500px; margin: 0 auto; }}
    .summary {{ background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 20px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .summary-item {{ background: #111827; border-radius: 8px; padding: 12px; }}
    .summary-label {{ color: #9ca3af; font-size: 12px; margin-bottom: 6px; }}
    .summary-value {{ font-size: 18px; font-weight: 700; word-break: break-word; }}
    .step-card {{ background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .step-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .step-title {{ font-size: 20px; font-weight: 700; }}
    .reward {{ padding: 6px 10px; border-radius: 999px; font-weight: 700; }}
    .reward-positive {{ background: #14532d; color: #86efac; }}
    .reward-zero {{ background: #3f3f46; color: #d4d4d8; }}
    .warning-badge {{ display: inline-block; margin-bottom: 10px; padding: 6px 10px; border-radius: 999px; background: #7c2d12; color: #fdba74; font-weight: 700; }}
    .meta-row, .delta-summary {{ display: flex; gap: 16px; flex-wrap: wrap; color: #cbd5e1; margin: 8px 0 12px; }}
    .image-grid, .obs-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
    .panel-title {{ color: #93c5fd; font-size: 13px; font-weight: 700; margin-bottom: 8px; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #374151; background: white; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; border-radius: 8px; padding: 12px; margin: 0; }}
    .img-meta, .muted, .delta-error, .snippet-meta {{ color: #9ca3af; font-size: 12px; margin-top: 6px; }}
    .img-missing {{ color: #9ca3af; background: #0f172a; border-radius: 8px; padding: 24px; }}
    .script-card {{ border: 1px solid #374151; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; background: #111827; }}
    .script-card summary {{ cursor: pointer; display: flex; justify-content: space-between; gap: 16px; }}
    .script-url {{ color: #f9fafb; word-break: break-all; }}
    .script-metrics {{ color: #93c5fd; white-space: nowrap; }}
    .snippet {{ margin-top: 10px; }}
    @media (max-width: 960px) {{
      .summary-grid, .image-grid, .obs-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="summary">
      <h1>Coverage Audit</h1>
      <div class="summary-grid">
        <div class="summary-item"><div class="summary-label">trajectory</div><div class="summary-value">{_escape(metadata.get("parquet_path", ""))}</div></div>
        <div class="summary-item"><div class="summary-label">target_url</div><div class="summary-value">{_escape(metadata.get("target_url", ""))}</div></div>
        <div class="summary-item"><div class="summary-label">steps / positive_rewards</div><div class="summary-value">{len(step_audits)} / {positive_steps}</div></div>
        <div class="summary-item"><div class="summary-label">sum Monocart delta score</div><div class="summary-value">{total_new_bytes}</div></div>
        <div class="summary-item"><div class="summary-label">suspicious reward steps</div><div class="summary-value">{suspicious_steps}</div></div>
      </div>
    </section>
    {"".join(_render_step(step) for step in step_audits)}
  </div>
</body>
</html>"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render trajectory + raw coverage audit HTML")
    parser.add_argument("parquet_path", help="Trajectory parquet path")
    parser.add_argument("--output-html", required=True, help="Output HTML path")
    parser.add_argument("--image-max-side", type=int, default=640, help="Rendered image max side")
    parser.add_argument("--obs-limit", type=int, default=5000, help="Observation text truncate length")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args([] if args is None else []) if False else None
    if args is None:
        args = parser.parse_args()
    parquet_path = Path(args.parquet_path).resolve()
    output_html = Path(args.output_html).resolve()
    step_audits, metadata = _load_step_audits(
        parquet_path=parquet_path,
        image_max_side=args.image_max_side,
        obs_limit=args.obs_limit,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(_render_html(step_audits, metadata), encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
