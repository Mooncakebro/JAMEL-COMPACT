from __future__ import annotations

import hashlib
import json
import fcntl
import logging
import os
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from jamel.core.env.web.utils import StepHistory

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATE_REPORT_SCRIPT = REPO_ROOT / "jamel/core/env/web/javascript/generate_report.js"


def get_step_coverage_path(step: Optional[StepHistory]) -> Optional[Path]:
    if step is None:
        return None

    extra_fields = step.extra_fields or {}
    coverage_path = extra_fields.get("coverage_path")
    if not coverage_path:
        return None

    return Path(coverage_path)


def normalize_coverage_path(path: str | Path | None) -> Optional[str]:
    if path is None:
        return None
    path_obj = Path(path).resolve()
    return str(path_obj)


def collect_history_coverage_paths(history: Optional[Iterable[StepHistory]]) -> list[str]:
    coverage_paths: list[str] = []
    for step in history or []:
        coverage_path = get_step_coverage_path(step)
        normalized_path = normalize_coverage_path(coverage_path)
        if normalized_path is None:
            continue
        if not Path(normalized_path).exists():
            logger.warning(
                "History step coverage file missing: step=%s coverage_path=%s",
                step.step,
                normalized_path,
            )
            continue
        coverage_paths.append(normalized_path)
    return coverage_paths


def dedupe_coverage_paths(coverage_paths: Iterable[str | Path]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for coverage_path in coverage_paths:
        normalized_path = normalize_coverage_path(coverage_path)
        if normalized_path is None:
            continue
        if normalized_path in seen:
            continue
        seen.add(normalized_path)
        deduped.append(normalized_path)
    return deduped


def extract_total_summary(summary_data: dict) -> dict:
    return summary_data.get("total", summary_data)


def coverage_score(summary_data: dict) -> int:
    total_summary = extract_total_summary(summary_data)
    return (
        total_summary.get("lines", {}).get("covered", 0)
        + total_summary.get("branches", {}).get("covered", 0)
        + total_summary.get("statements", {}).get("covered", 0)
        + total_summary.get("functions", {}).get("covered", 0)
    )


def coverage_file_has_entries(path: str | Path) -> bool:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(data, list):
        return len(data) > 0
    if isinstance(data, dict):
        return bool(data)
    return False


def monocart_coverage_score_for_paths(coverage_paths: Iterable[str | Path] | None) -> int:
    normalized_paths = dedupe_coverage_paths(coverage_paths or [])
    if not normalized_paths:
        return 0
    summary = generate_cumulative_coverage_summary(tuple(normalized_paths))
    if not summary:
        return 0
    return int(coverage_score(summary))


def compute_monocart_coverage_reward_details(
    *,
    current_path: str | Path | None,
    baseline_paths: Iterable[str | Path] | None = None,
    previous_score: int = 0,
) -> dict[str, Any]:
    """Compute binary novelty reward from cumulative Monocart coverage.

    The scalar score is the sum of Monocart/Istanbul covered counts for lines,
    branches, statements, and functions. Reward is 1 iff the cumulative score
    increases after adding the current coverage artifact.
    """
    if current_path is None:
        return {
            "reward": 0.0,
            "previous_score": int(previous_score),
            "current_score": int(previous_score),
            "delta_score": 0,
            "skip_reason": "missing_current_coverage_path",
        }

    current_path_obj = Path(current_path)
    if not current_path_obj.exists():
        return {
            "reward": 0.0,
            "previous_score": int(previous_score),
            "current_score": int(previous_score),
            "delta_score": 0,
            "skip_reason": "missing_current_coverage_file",
        }
    normalized_baseline = dedupe_coverage_paths(baseline_paths or [])
    current_paths = tuple(dedupe_coverage_paths([*normalized_baseline, current_path_obj]))
    current_summary = generate_cumulative_coverage_summary(current_paths)
    if not current_summary:
        return {
            "reward": 0.0,
            "previous_score": int(previous_score),
            "current_score": int(previous_score),
            "delta_score": 0,
            "skip_reason": "missing_current_summary",
        }

    current_score = int(coverage_score(current_summary))
    computed_previous_score = int(previous_score)
    if normalized_baseline:
        previous_summary = generate_cumulative_coverage_summary(tuple(normalized_baseline))
        if not previous_summary:
            return {
                "reward": 0.0,
                "previous_score": computed_previous_score,
                "current_score": current_score,
                "delta_score": 0,
                "skip_reason": "missing_previous_summary",
            }
        computed_previous_score = int(coverage_score(previous_summary))

    delta_score = max(0, current_score - computed_previous_score)
    return {
        "reward": 1.0 if delta_score > 0 else 0.0,
        "previous_score": computed_previous_score,
        "current_score": current_score,
        "delta_score": delta_score,
        "skip_reason": None,
    }


def _coverage_report_output_dir(resolved_paths: list[str]) -> Path:
    digest = hashlib.sha256("\0".join(resolved_paths).encode("utf-8")).hexdigest()[:16]
    last_file_stem = Path(resolved_paths[-1]).stem or "coverage"
    return REPO_ROOT / "data" / "coverage-report" / f"{last_file_stem}_{digest}"


@contextmanager
def _coverage_report_lock(output_dir: Path):
    lock_dir = REPO_ROOT / "data" / "coverage-report" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{output_dir.name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@lru_cache(maxsize=512)
def generate_cumulative_coverage_summary(coverage_paths: tuple[str, ...]) -> dict:
    if not coverage_paths:
        return {}

    resolved_paths = [str(Path(path).resolve()) for path in coverage_paths]

    if not any(coverage_file_has_entries(path) for path in resolved_paths):
        return {}

    if not GENERATE_REPORT_SCRIPT.exists():
        logger.error("Coverage utility failed: report script missing: %s", GENERATE_REPORT_SCRIPT)
        return {}

    output_dir = _coverage_report_output_dir(resolved_paths)
    summary_path = output_dir / "coverage-summary.json"

    with _coverage_report_lock(output_dir):
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Coverage utility found invalid summary JSON; regenerating: %s", summary_path)

        try:
            env = os.environ.copy()
            if not env.get("NODE_PATH"):
                npm_root = subprocess.run(
                    ["npm", "root", "-g"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if npm_root.returncode == 0 and npm_root.stdout.strip():
                    env["NODE_PATH"] = npm_root.stdout.strip()
            subprocess.run(
                ["node", str(GENERATE_REPORT_SCRIPT), "--output-dir", str(output_dir), *resolved_paths],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except FileNotFoundError:
            logger.error("Coverage utility failed: node executable not found")
            return {}
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Coverage utility failed: report script execution error returncode=%s stdout=%s stderr=%s",
                exc.returncode,
                exc.stdout,
                exc.stderr,
            )
            return {}

        if not summary_path.exists():
            logger.error("Coverage utility failed: summary file missing: %s", summary_path)
            return {}

        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
