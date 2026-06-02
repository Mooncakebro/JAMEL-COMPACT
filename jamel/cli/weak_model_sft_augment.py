from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import math
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from jamel.cli.weak_model_sft_label import (
    ACTION_SPACE,
    PROMPT_TEMPLATE,
    _format_interactive_elements,
)
from jamel.core.reward.web.utils import (
    compute_monocart_coverage_reward_details,
    monocart_coverage_score_for_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_INDEX = (
    REPO_ROOT
    / "outputs"
    / "weak_model_sft_labeling"
    / "weak_model_sft_full_multimodal_20260518_102300"
    / "all_episode_manifest.jsonl"
)
SFT_SORT_FIELDS = ("target_app", "session_id", "episode_idx", "step_idx")
MANIFEST_SORT_FIELDS = ("target_app", "session_id", "episode_idx")
RESET_ACTION = "reset()"
RESET_THINK = "Reset to the initial state and start a new exploration episode."
RESET_RESPONSE = f"<think>{RESET_THINK}</think><action>{RESET_ACTION}</action>"
ACTION_SPACE_WITH_RESET = f"{ACTION_SPACE}\n{RESET_ACTION}"


@dataclass(frozen=True)
class CoverageEvidence:
    raw_bytes: bytes | None
    sha256: str
    size_bytes: int
    gzip_bytes: bytes | None
    cache_path: str | None
    valid: bool
    skip_reason: str | None = None


@dataclass
class StepRecord:
    row: dict[str, Any]
    step: int
    source_episode_id: int
    source_trace_path: str
    backup_trace_path: str
    coverage: CoverageEvidence
    action_format_valid: bool
    action_execution_valid: bool


@dataclass
class EpisodeRecord:
    target_app: str
    source_episode_id: int
    checkpoint_id: str
    source_trace_path: str
    backup_trace_path: str
    source_episode_dir: str
    backup_episode_dir: str
    worker_id: Any = None
    manifest_line: int | None = None
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def uid(self) -> str:
        return f"{self.target_app}:{self.source_episode_id}:{self.source_trace_path}"


@dataclass
class SyntheticEpisodeResult:
    manifest_record: dict[str, Any]
    sft_rows: list[dict[str, Any]]
    accepted: bool
    deduped: bool


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, bytes):
        return {"__bytes_len__": len(value)}
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _sort_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    key: list[Any] = []
    for field in fields:
        value = row.get(field)
        if field.endswith("_idx"):
            key.append(_as_int(value, 0))
        else:
            key.append(str(value or ""))
    return tuple(key)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value)) if not isinstance(value, (bytes, dict, list, tuple, set)) else False
    except (TypeError, ValueError):
        return False


def _as_int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _extra_fields(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("extra_fields")
    return dict(value) if isinstance(value, dict) else {}


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key)
    if _is_missing(value):
        extra = _extra_fields(row)
        value = extra.get(key, default)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("manifest_line", line_number)
            records.append(record)
    return records


def _parse_apps(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _filter_manifest_records(
    records: list[dict[str, Any]],
    *,
    apps: set[str] | None,
    max_apps: int | None,
    max_episodes_per_app: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        target_app = str(record.get("target_app") or "")
        if not target_app:
            continue
        if apps is not None and target_app not in apps:
            continue
        grouped.setdefault(target_app, []).append(record)

    selected: list[dict[str, Any]] = []
    for target_app in sorted(grouped)[: max_apps or None]:
        app_records = sorted(
            grouped[target_app],
            key=lambda item: (
                _as_int(item.get("episode_id"), 0),
                str(item.get("trace_path") or ""),
            ),
        )
        if max_episodes_per_app is not None:
            app_records = app_records[:max_episodes_per_app]
        selected.extend(app_records)
    return selected


def _safe_app_name(target_app: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in target_app)


def _copy_file(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source_path": str(src.resolve()),
        "backup_path": str(dst.resolve()),
        "size_bytes": dst.stat().st_size,
        "sha256": _sha256_file(dst),
    }


def backup_source_records(
    *,
    source_index: Path,
    records: list[dict[str, Any]],
    backup_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Backup directory already exists and is not empty: {backup_dir}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    index_backup = backup_dir / "source_index.jsonl"
    shutil.copy2(source_index, index_backup)

    trace_maps: list[dict[str, Any]] = []
    total_files = 1
    total_bytes = index_backup.stat().st_size
    seen_trace_paths: set[str] = set()

    for record in records:
        source_trace = Path(str(record.get("trace_path") or "")).expanduser().resolve()
        if not source_trace.exists():
            raise FileNotFoundError(f"Trace parquet not found: {source_trace}")
        if str(source_trace) in seen_trace_paths:
            continue
        seen_trace_paths.add(str(source_trace))

        target_app = str(record["target_app"])
        episode_id = _as_int(record.get("episode_id"), 0)
        trace_digest = hashlib.sha256(str(source_trace).encode("utf-8")).hexdigest()[:10]
        episode_dir = (
            backup_dir
            / "traces"
            / _safe_app_name(target_app)
            / f"episode_{episode_id:06d}_{trace_digest}"
        )
        backup_trace = episode_dir / "trace.parquet"
        trace_file = _copy_file(source_trace, backup_trace)
        total_files += 1
        total_bytes += trace_file["size_bytes"]

        coverage_files: list[dict[str, Any]] = []
        source_coverage_dir = source_trace.parent / "coverage"
        backup_coverage_dir = episode_dir / "coverage"
        if source_coverage_dir.exists():
            backup_coverage_dir.mkdir(parents=True, exist_ok=True)
            for source_coverage in sorted(source_coverage_dir.glob("*.json")):
                backup_coverage = backup_coverage_dir / source_coverage.name
                file_record = _copy_file(source_coverage, backup_coverage)
                coverage_files.append(file_record)
                total_files += 1
                total_bytes += file_record["size_bytes"]

        trace_map = {
            "target_app": target_app,
            "episode_id": episode_id,
            "checkpoint_id": record.get("checkpoint_id"),
            "worker_id": record.get("worker_id"),
            "manifest_line": record.get("manifest_line"),
            "source_trace_path": str(source_trace),
            "backup_trace_path": str(backup_trace.resolve()),
            "source_episode_dir": str(source_trace.parent.resolve()),
            "backup_episode_dir": str(episode_dir.resolve()),
            "source_coverage_dir": str(source_coverage_dir.resolve()),
            "backup_coverage_dir": str(backup_coverage_dir.resolve()),
            "trace_file": trace_file,
            "coverage_files": coverage_files,
        }
        trace_maps.append(trace_map)
        _append_jsonl(backup_dir / "trace_map.jsonl", trace_map)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_index": str(source_index.resolve()),
        "source_index_backup": str(index_backup.resolve()),
        "backup_dir": str(backup_dir.resolve()),
        "selected_manifest_records": len(records),
        "unique_traces": len(trace_maps),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "trace_map_path": str((backup_dir / "trace_map.jsonl").resolve()),
    }
    _write_json(backup_dir / "backup_manifest.json", summary)
    return trace_maps, summary


def _mapped_backup_coverage_path(row_path: str, episode: EpisodeRecord) -> Path | None:
    if not row_path:
        return None
    source_episode_dir = Path(episode.source_episode_dir)
    backup_episode_dir = Path(episode.backup_episode_dir)
    path = Path(row_path)
    try:
        relative = path.resolve().relative_to(source_episode_dir.resolve())
        return backup_episode_dir / relative
    except (ValueError, FileNotFoundError, RuntimeError):
        candidate = backup_episode_dir / "coverage" / path.name
        return candidate if path.name else None


def _coverage_evidence_from_row(
    *,
    row: dict[str, Any],
    episode: EpisodeRecord,
    coverage_cache_dir: Path,
) -> CoverageEvidence:
    expected_sha = str(_row_value(row, "coverage_sha256", "") or "")
    embedded = row.get("coverage_json_gzip_bytes")
    raw_bytes: bytes | None = None
    skip_reason: str | None = None

    if isinstance(embedded, bytes) and embedded:
        try:
            raw_bytes = gzip.decompress(embedded)
        except (OSError, EOFError) as exc:
            skip_reason = f"invalid_embedded_coverage_gzip:{exc}"
    elif isinstance(embedded, bytearray) and embedded:
        try:
            raw_bytes = gzip.decompress(bytes(embedded))
        except (OSError, EOFError) as exc:
            skip_reason = f"invalid_embedded_coverage_gzip:{exc}"

    if raw_bytes is None:
        row_path = str(_row_value(row, "coverage_path", "") or "")
        backup_path = _mapped_backup_coverage_path(row_path, episode)
        if backup_path is None or not backup_path.exists():
            return CoverageEvidence(
                raw_bytes=None,
                sha256=expected_sha,
                size_bytes=0,
                gzip_bytes=None,
                cache_path=None,
                valid=False,
                skip_reason=skip_reason or "missing_coverage_artifact",
            )
        raw_bytes = backup_path.read_bytes()

    actual_sha = _sha256_bytes(raw_bytes)
    if expected_sha and actual_sha != expected_sha:
        return CoverageEvidence(
            raw_bytes=raw_bytes,
            sha256=actual_sha,
            size_bytes=len(raw_bytes),
            gzip_bytes=gzip.compress(raw_bytes, mtime=0),
            cache_path=None,
            valid=False,
            skip_reason="coverage_sha256_mismatch",
        )

    cache_path = coverage_cache_dir / "raw" / f"{actual_sha}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        cache_path.write_bytes(raw_bytes)

    return CoverageEvidence(
        raw_bytes=raw_bytes,
        sha256=actual_sha,
        size_bytes=len(raw_bytes),
        gzip_bytes=gzip.compress(raw_bytes, mtime=0),
        cache_path=str(cache_path.resolve()),
        valid=True,
        skip_reason=None,
    )


def load_backed_up_episodes(
    *,
    trace_maps: list[dict[str, Any]],
    coverage_cache_dir: Path,
) -> list[EpisodeRecord]:
    episodes: list[EpisodeRecord] = []
    for trace_map in trace_maps:
        episode = EpisodeRecord(
            target_app=str(trace_map["target_app"]),
            source_episode_id=_as_int(trace_map.get("episode_id"), 0),
            checkpoint_id=str(trace_map.get("checkpoint_id") or ""),
            source_trace_path=str(trace_map["source_trace_path"]),
            backup_trace_path=str(trace_map["backup_trace_path"]),
            source_episode_dir=str(trace_map["source_episode_dir"]),
            backup_episode_dir=str(trace_map["backup_episode_dir"]),
            worker_id=trace_map.get("worker_id"),
            manifest_line=_as_int(trace_map.get("manifest_line"), 0) or None,
        )
        df = pd.read_parquet(episode.backup_trace_path)
        if "step" in df.columns:
            df = df.sort_values("step")
        for _, series in df.iterrows():
            row = series.to_dict()
            evidence = _coverage_evidence_from_row(
                row=row,
                episode=episode,
                coverage_cache_dir=coverage_cache_dir,
            )
            step = StepRecord(
                row=row,
                step=_as_int(_row_value(row, "step"), 0),
                source_episode_id=episode.source_episode_id,
                source_trace_path=episode.source_trace_path,
                backup_trace_path=episode.backup_trace_path,
                coverage=evidence,
                action_format_valid=_as_bool(_row_value(row, "action_format_valid"), False),
                action_execution_valid=_as_bool(_row_value(row, "action_execution_valid"), False),
            )
            episode.steps.append(step)
        episodes.append(episode)
    return episodes


def _prefix_action_hash(episode: EpisodeRecord, last_step: int) -> str:
    actions = []
    for step in episode.steps:
        if step.step > last_step:
            continue
        action = _row_value(step.row, "action", "")
        if not action:
            parsed = step.row.get("parsed_content")
            if isinstance(parsed, dict):
                action = parsed.get("action", "")
        actions.append(str(action))
    payload = "\n".join(actions).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prefix_valid(episode: EpisodeRecord, last_step: int) -> tuple[bool, str]:
    for step in episode.steps:
        if step.step > last_step:
            continue
        if not step.action_format_valid:
            return False, f"invalid_action_format_at_step_{step.step}"
        if not step.action_execution_valid:
            return False, f"invalid_action_execution_at_step_{step.step}"
        if not step.coverage.valid:
            return False, f"invalid_coverage_at_step_{step.step}"
        if not step.coverage.sha256:
            return False, f"missing_coverage_sha256_at_step_{step.step}"
    return True, "accepted"


def _first_invalid_prefix_reason(episode: EpisodeRecord) -> str | None:
    for step in episode.steps:
        if not step.action_format_valid:
            return f"invalid_action_format_at_step_{step.step}"
        if not step.action_execution_valid:
            return f"invalid_action_execution_at_step_{step.step}"
        if not step.coverage.valid:
            return f"invalid_coverage_at_step_{step.step}"
        if not step.coverage.sha256:
            return f"missing_coverage_sha256_at_step_{step.step}"
    return None


SFT_REMOVED_FIELDS = {
    "episode_id",
    "step",
    "global_step",
    "checkpoint_id",
    "permutation_id",
    "permutation_index",
    "permutation_order_index",
    "augmentation_id",
    "accepted_prefix_last_step",
    "augmented_checkpoint_id",
    "synthetic_episode_id",
    "synthetic_global_step",
    "source_episode_id",
    "source_step",
    "source_trace_path",
    "backup_trace_path",
    "source_reward",
    "source_coverage_delta_score",
    "source_checkpoint_id",
    "source_manifest_line",
}


def _clean_training_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in SFT_REMOVED_FIELDS}


def _ensure_reset_in_prompt(prompt: Any) -> Any:
    if not isinstance(prompt, str) or RESET_ACTION in prompt:
        return prompt
    marker = "\n\nCurrent observation:"
    if marker in prompt:
        return prompt.replace(marker, f"\n{RESET_ACTION}{marker}", 1)
    return f"{prompt}\n{RESET_ACTION}"


def _build_prompt(target_app: str, start_url: str, observation: str) -> str:
    return PROMPT_TEMPLATE.format(
        target_app=target_app,
        start_url=start_url,
        action_space=ACTION_SPACE_WITH_RESET,
        observation=observation,
        interactive_elements=_format_interactive_elements(observation),
    )


def _updated_extra_fields(
    row: dict[str, Any],
    *,
    session_id: str,
    episode_idx: int,
    step_idx: int,
    reward: float,
    previous_score: int,
    current_score: int,
    delta_score: int,
    skip_reason: str | None,
    coverage: CoverageEvidence,
) -> dict[str, Any]:
    extra = _clean_training_fields(_extra_fields(row))
    if "prompt" in extra:
        extra["prompt"] = _ensure_reset_in_prompt(extra["prompt"])
    extra.update(
        {
            "session_id": session_id,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "reward_source": "coverage" if reward > 0 else "none",
            "reward": reward,
            "coverage_previous_score": previous_score,
            "coverage_current_score": current_score,
            "coverage_delta_score": delta_score,
            "coverage_skip_reason": skip_reason,
            "coverage_path": coverage.cache_path,
            "coverage_sha256": coverage.sha256,
            "coverage_size_bytes": coverage.size_bytes,
            "coverage_exists_at_write": coverage.valid,
        }
    )
    return extra


def _build_sft_row(
    step: StepRecord,
    *,
    target_app: str,
    session_id: str,
    episode_idx: int,
    step_idx: int,
    reward: float,
    previous_score: int,
    current_score: int,
    delta_score: int,
    skip_reason: str | None,
) -> dict[str, Any]:
    row = _clean_training_fields(dict(step.row))
    if "prompt" in row:
        row["prompt"] = _ensure_reset_in_prompt(row["prompt"])
    row.update(
        {
            "session_id": session_id,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "target_app": target_app,
            "reward": reward,
            "coverage_delta_score": delta_score,
            "coverage_previous_score": previous_score,
            "coverage_current_score": current_score,
            "coverage_skip_reason": skip_reason,
            "coverage_path": step.coverage.cache_path,
            "coverage_json_gzip_bytes": step.coverage.gzip_bytes,
            "coverage_sha256": step.coverage.sha256,
            "coverage_size_bytes": step.coverage.size_bytes,
            "coverage_exists_at_write": step.coverage.valid,
        }
    )
    row["extra_fields"] = _updated_extra_fields(
        step.row,
        session_id=session_id,
        episode_idx=episode_idx,
        step_idx=step_idx,
        reward=reward,
        previous_score=previous_score,
        current_score=current_score,
        delta_score=delta_score,
        skip_reason=skip_reason,
        coverage=step.coverage,
    )
    return row


def _last_step_score(rewards_by_step: dict[int, dict[str, Any]], default: int) -> int:
    if not rewards_by_step:
        return default
    last_step = max(rewards_by_step)
    return _as_int(rewards_by_step[last_step].get("current_score"), default)


def _build_reset_row(
    *,
    episode: EpisodeRecord,
    target_app: str,
    session_id: str,
    episode_idx: int,
    step_idx: int,
    coverage_score_at_reset: int,
) -> dict[str, Any]:
    source_step = episode.steps[-1] if episode.steps else None
    source_row = dict(source_step.row) if source_step is not None else {}
    row = _clean_training_fields(source_row)
    final_observation = str(
        source_row.get("after_observation_str")
        or source_row.get("before_observation_str")
        or ""
    )
    start_url = str(source_row.get("start_url") or source_row.get("target_url") or "")
    prompt = _build_prompt(target_app, start_url, final_observation)

    after_to_before_fields = {
        "before_info": "after_info",
        "before_chat_messages": "after_chat_messages",
        "before_screenshot": "after_screenshot",
        "before_goal_object": "after_goal_object",
        "before_last_action": "after_last_action",
        "before_last_action_error": "after_last_action_error",
        "before_open_pages_urls": "after_open_pages_urls",
        "before_open_pages_titles": "after_open_pages_titles",
        "before_active_page_index": "after_active_page_index",
        "before_axtree_object": "after_axtree_object",
        "before_dom_object": "after_dom_object",
    }
    for before_key, after_key in after_to_before_fields.items():
        if after_key in source_row:
            row[before_key] = source_row.get(after_key)
    row["before_observation_str"] = final_observation
    for key in (
        "after_info",
        "after_chat_messages",
        "after_screenshot",
        "after_goal_object",
        "after_last_action",
        "after_last_action_error",
        "after_open_pages_urls",
        "after_open_pages_titles",
        "after_active_page_index",
        "after_axtree_object",
        "after_dom_object",
        "after_observation_str",
    ):
        if key in row:
            row[key] = None

    row.update(
        {
            "session_id": session_id,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "target_app": target_app,
            "prompt": prompt,
            "response": RESET_RESPONSE,
            "think": RESET_THINK,
            "action": RESET_ACTION,
            "raw_content": RESET_RESPONSE,
            "parsed_content": {"think": RESET_THINK, "action": RESET_ACTION},
            "result": None,
            "action_format_valid": True,
            "action_validation_error": None,
            "action_execution_valid": True,
            "model_retry_attempts": 0,
            "reward": 0.0,
            "coverage_delta_score": 0,
            "coverage_previous_score": coverage_score_at_reset,
            "coverage_current_score": coverage_score_at_reset,
            "coverage_skip_reason": "reset_action",
            "coverage_path": None,
            "coverage_json_gzip_bytes": None,
            "coverage_sha256": "",
            "coverage_size_bytes": 0,
            "coverage_exists_at_write": False,
        }
    )
    extra = _clean_training_fields(_extra_fields(source_row))
    extra.update(
        {
            "session_id": session_id,
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "prompt": prompt,
            "response": RESET_RESPONSE,
            "think": RESET_THINK,
            "action": RESET_ACTION,
            "action_format_valid": True,
            "action_validation_error": None,
            "action_execution_valid": True,
            "model_retry_attempts": 0,
            "reward_source": "reset",
            "reward": 0.0,
            "coverage_previous_score": coverage_score_at_reset,
            "coverage_current_score": coverage_score_at_reset,
            "coverage_delta_score": 0,
            "coverage_skip_reason": "reset_action",
            "coverage_path": None,
            "coverage_sha256": "",
            "coverage_size_bytes": 0,
            "coverage_exists_at_write": False,
        }
    )
    row["extra_fields"] = extra
    return row


def _simulate_synthetic_episode(
    *,
    episode: EpisodeRecord,
    app_seen_coverage_paths: list[str],
    target_app: str,
    session_id: str,
    episode_idx: int,
    prefix_seen: set[tuple[str, str, int, str]],
    dedupe_mode: str,
) -> tuple[SyntheticEpisodeResult, list[str]]:
    episode_seen_coverage_paths: list[str] = []
    rewards_by_step: dict[int, dict[str, Any]] = {}
    positive_steps: list[int] = []
    positive_coverage_paths_for_app: list[str] = []
    last_coverage_score = monocart_coverage_score_for_paths(app_seen_coverage_paths)

    for step in episode.steps:
        skip_reason = step.coverage.skip_reason
        if not step.action_format_valid or not step.action_execution_valid:
            skip_reason = skip_reason or "invalid_action"
            reward_data = {
                "reward": 0.0,
                "previous_score": last_coverage_score,
                "current_score": last_coverage_score,
                "delta_score": 0,
                "skip_reason": skip_reason,
            }
        elif not step.coverage.valid:
            skip_reason = skip_reason or "invalid_coverage"
            reward_data = {
                "reward": 0.0,
                "previous_score": last_coverage_score,
                "current_score": last_coverage_score,
                "delta_score": 0,
                "skip_reason": skip_reason,
            }
        else:
            reward_data = compute_monocart_coverage_reward_details(
                current_path=step.coverage.cache_path,
                baseline_paths=[*app_seen_coverage_paths, *episode_seen_coverage_paths],
                previous_score=last_coverage_score,
            )
            last_coverage_score = _as_int(
                reward_data.get("current_score"),
                last_coverage_score,
            )

        reward = float(reward_data.get("reward", 0.0) or 0.0)
        rewards_by_step[step.step] = {
            "reward": reward,
            "previous_score": _as_int(reward_data.get("previous_score"), last_coverage_score),
            "current_score": _as_int(reward_data.get("current_score"), last_coverage_score),
            "delta_score": _as_int(reward_data.get("delta_score"), 0),
            "skip_reason": reward_data.get("skip_reason") or skip_reason,
            "coverage_path": step.coverage.cache_path,
            "coverage_sha256": step.coverage.sha256,
            "coverage_valid": step.coverage.valid,
            "action_format_valid": step.action_format_valid,
            "action_execution_valid": step.action_execution_valid,
        }
        if step.coverage.valid and step.coverage.cache_path:
            episode_seen_coverage_paths.append(step.coverage.cache_path)
        if reward > 0:
            positive_steps.append(step.step)
            if step.coverage.cache_path:
                positive_coverage_paths_for_app.append(step.coverage.cache_path)

    last_positive_step = max(positive_steps) if positive_steps else None
    prefix_reason = "no_positive_reward"
    accepted = False
    deduped = False
    sft_rows: list[dict[str, Any]] = []

    if last_positive_step is not None:
        prefix_ok, prefix_reason = _prefix_valid(episode, last_positive_step)
        if prefix_ok:
            action_hash = _prefix_action_hash(episode, last_positive_step)
            prefix_key = (target_app, episode.source_trace_path, last_positive_step, action_hash)
            deduped = dedupe_mode == "prefix" and prefix_key in prefix_seen
            if not deduped:
                prefix_seen.add(prefix_key)
                accepted = True
                for step in episode.steps:
                    if step.step > last_positive_step:
                        continue
                    reward_data = rewards_by_step[step.step]
                    sft_rows.append(
                        _build_sft_row(
                            step,
                            target_app=target_app,
                            session_id=session_id,
                            episode_idx=episode_idx,
                            step_idx=step.step,
                            reward=reward_data["reward"],
                            previous_score=reward_data["previous_score"],
                            current_score=reward_data["current_score"],
                            delta_score=reward_data["delta_score"],
                            skip_reason=reward_data["skip_reason"],
                        )
                    )
        else:
            prefix_reason = prefix_reason
    else:
        prefix_reason = _first_invalid_prefix_reason(episode) or prefix_reason

    reset_step_idx = None
    if accepted and last_positive_step is not None:
        reset_step_idx = last_positive_step + 1
        reset_score = _as_int(
            rewards_by_step[last_positive_step].get("current_score"),
            last_coverage_score,
        )
        sft_rows.append(
            _build_reset_row(
                episode=episode,
                target_app=target_app,
                session_id=session_id,
                episode_idx=episode_idx,
                step_idx=reset_step_idx,
                coverage_score_at_reset=reset_score,
            )
        )

    manifest_record = {
        "session_id": session_id,
        "target_app": target_app,
        "episode_idx": episode_idx,
        "source_episode_id": episode.source_episode_id,
        "source_trace_path": episode.source_trace_path,
        "backup_trace_path": episode.backup_trace_path,
        "source_checkpoint_id": episode.checkpoint_id,
        "source_manifest_line": episode.manifest_line,
        "steps": len(episode.steps),
        "reset_step_idx": reset_step_idx,
        "reset_action": RESET_ACTION if reset_step_idx is not None else None,
        "positive_step_indices": positive_steps,
        "accepted_prefix_last_step_idx": last_positive_step if accepted else None,
        "candidate_prefix_last_step_idx": last_positive_step,
        "accepted_prefix_reason": "deduped_prefix" if deduped else prefix_reason,
        "accepted": accepted,
        "deduped": deduped,
        "coverage_growth": sum(
            rewards_by_step[step]["delta_score"] for step in positive_steps
        ),
        "valid_action_rate": (
            sum(1 for step in episode.steps if step.action_format_valid and step.action_execution_valid)
            / max(1, len(episode.steps))
        ),
        "coverage_valid_steps": sum(1 for step in episode.steps if step.coverage.valid),
        "step_rewards": [
            {
                "step_idx": step.step,
                "source_step": step.step,
                "reward": rewards_by_step[step.step]["reward"],
                "coverage_delta_score": rewards_by_step[step.step]["delta_score"],
                "coverage_previous_score": rewards_by_step[step.step]["previous_score"],
                "coverage_current_score": rewards_by_step[step.step]["current_score"],
                "coverage_skip_reason": rewards_by_step[step.step]["skip_reason"],
                "coverage_path": rewards_by_step[step.step]["coverage_path"],
                "coverage_sha256": rewards_by_step[step.step]["coverage_sha256"],
                "coverage_valid": rewards_by_step[step.step]["coverage_valid"],
                "action_format_valid": rewards_by_step[step.step]["action_format_valid"],
                "action_execution_valid": rewards_by_step[step.step]["action_execution_valid"],
            }
            for step in episode.steps
        ],
    }
    return (
        SyntheticEpisodeResult(
            manifest_record=manifest_record,
            sft_rows=sft_rows,
            accepted=accepted,
            deduped=deduped,
        ),
        positive_coverage_paths_for_app,
    )


def _permutation_orders(
    episodes: list[EpisodeRecord],
    *,
    permutations_per_app: int,
    rng: random.Random,
) -> list[list[EpisodeRecord]]:
    if not episodes or permutations_per_app <= 0:
        return []
    if len(episodes) == 1:
        return [list(episodes)]

    orders: list[list[EpisodeRecord]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = max(100, permutations_per_app * 20)
    while len(orders) < permutations_per_app and attempts < max_attempts:
        attempts += 1
        order = list(episodes)
        rng.shuffle(order)
        key = tuple(item.uid for item in order)
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders


def augment_app_episodes(
    *,
    target_app: str,
    episodes: list[EpisodeRecord],
    permutations_per_app: int,
    seed: int,
    session_namespace: str,
    dedupe_mode: str = "prefix",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(f"{seed}:{target_app}")
    orders = _permutation_orders(
        episodes,
        permutations_per_app=permutations_per_app,
        rng=rng,
    )
    manifest_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    prefix_seen: set[tuple[str, str, int, str]] = set()

    for session_index, order in enumerate(orders):
        app_seen_coverage_paths: list[str] = []
        order_hash = hashlib.sha256("|".join(item.uid for item in order).encode("utf-8")).hexdigest()[:12]
        session_id = (
            f"{session_namespace}:{_safe_app_name(target_app)}:"
            f"session-{session_index:04d}-{order_hash}"
        )
        for episode_idx, episode in enumerate(order):
            result, positive_coverage_paths = _simulate_synthetic_episode(
                episode=episode,
                app_seen_coverage_paths=app_seen_coverage_paths,
                target_app=target_app,
                session_id=session_id,
                episode_idx=episode_idx,
                prefix_seen=prefix_seen,
                dedupe_mode=dedupe_mode,
            )
            app_seen_coverage_paths.extend(positive_coverage_paths)
            manifest_rows.append(result.manifest_record)
            sft_rows.extend(result.sft_rows)

    manifest_rows.sort(key=lambda row: _sort_key(row, MANIFEST_SORT_FIELDS))
    sft_rows.sort(key=lambda row: _sort_key(row, SFT_SORT_FIELDS))
    return manifest_rows, sft_rows


def _render_audit_html(summary: dict[str, Any], app_summaries: list[dict[str, Any]]) -> str:
    rows = []
    for item in app_summaries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('target_app', '')))}</td>"
            f"<td>{item.get('source_episodes', 0)}</td>"
            f"<td>{item.get('synthetic_episodes', 0)}</td>"
            f"<td>{item.get('accepted_episodes', 0)}</td>"
            f"<td>{item.get('deduped_episodes', 0)}</td>"
            f"<td>{item.get('sft_rows', 0)}</td>"
            f"<td>{item.get('reset_rows', 0)}</td>"
            f"<td>{item.get('coverage_growth', 0)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Weak Model SFT Augmentation Audit</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    pre {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Weak Model SFT Augmentation Audit</h1>
  <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))}</pre>
  <table>
    <thead>
      <tr>
        <th>app</th><th>source episodes</th><th>synthetic episodes</th>
        <th>accepted</th><th>deduped</th><th>sft rows</th><th>reset rows</th><th>coverage growth</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def _canonical_delta(
    baseline_paths: list[str],
    current_path: str | None,
) -> dict[str, Any]:
    details = compute_monocart_coverage_reward_details(
        current_path=current_path,
        baseline_paths=baseline_paths,
        previous_score=monocart_coverage_score_for_paths(baseline_paths),
    )
    if details.get("skip_reason") in {
        "missing_current_coverage_path",
        "missing_current_coverage_file",
        "missing_current_summary",
        "missing_previous_summary",
    }:
        return {"error": details["skip_reason"], **details}
    return {
        **details,
        "baseline_files": len(baseline_paths),
        "current_files": len([*baseline_paths, current_path] if current_path else baseline_paths),
    }


def run_canonical_audit(
    *,
    manifest_rows: list[dict[str, Any]],
    sample_count: int,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    candidates = [row for row in manifest_rows if row.get("step_rewards")]
    if sample_count <= 0 or not candidates:
        summary = {
            "requested_samples": sample_count,
            "checked_samples": 0,
            "positive_reward_disagreements": 0,
            "output_path": str(output_path.resolve()),
            "skipped_reason": "no_samples_requested_or_available",
        }
        _write_json(output_path.with_suffix(".summary.json"), summary)
        return summary

    rng = random.Random(seed)
    sample_rows = list(candidates)
    if len(candidates) > sample_count:
        sample_rows = rng.sample(candidates, sample_count)
    selected_keys = {
        (
            str(row.get("target_app")),
            str(row.get("session_id")),
            int(row.get("episode_idx") or 0),
        )
        for row in sample_rows
    }

    replay_rows = sorted(
        manifest_rows,
        key=lambda row: (
            str(row.get("target_app") or ""),
            str(row.get("session_id") or ""),
            int(row.get("episode_idx") or 0),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checked = 0
    disagreements = 0
    app_baseline_paths: dict[tuple[str, str], list[str]] = {}
    with output_path.open("w", encoding="utf-8") as fout:
        for episode in replay_rows:
            target_app = str(episode.get("target_app"))
            session_id = str(episode.get("session_id"))
            app_key = (target_app, session_id)
            selected_key = (
                target_app,
                session_id,
                int(episode.get("episode_idx") or 0),
            )
            baseline_paths = list(app_baseline_paths.get(app_key, []))
            should_audit = selected_key in selected_keys
            for step in episode.get("step_rewards") or []:
                current_path = step.get("coverage_path")
                if should_audit:
                    canonical = _canonical_delta(baseline_paths, current_path)
                    recorded_reward = float(step.get("reward") or 0.0)
                    canonical_reward = (
                        float(canonical.get("reward") or 0.0)
                        if "error" not in canonical
                        else 0.0
                    )
                    disagreement = recorded_reward != canonical_reward
                    if disagreement:
                        disagreements += 1
                    checked += 1
                    fout.write(
                        json.dumps(
                            {
                                "target_app": episode.get("target_app"),
                                "session_id": episode.get("session_id"),
                                "episode_idx": episode.get("episode_idx"),
                                "source_episode_id": episode.get("source_episode_id"),
                                "step_idx": step.get("step_idx"),
                                "source_step": step.get("source_step"),
                                "recorded_reward": recorded_reward,
                                "recorded_delta_score": step.get("coverage_delta_score"),
                                "canonical": canonical,
                                "positive_reward_disagreement": disagreement,
                            },
                            ensure_ascii=False,
                            default=_json_default,
                        )
                        + "\n"
                    )
                if (
                    step.get("coverage_valid")
                    and current_path
                ):
                    baseline_paths.append(str(current_path))
            positive_paths = [
                str(step.get("coverage_path"))
                for step in episode.get("step_rewards") or []
                if float(step.get("reward") or 0.0) > 0
                and step.get("coverage_valid")
                and step.get("coverage_path")
            ]
            app_baseline_paths[app_key] = [
                *app_baseline_paths.get(app_key, []),
                *positive_paths,
            ]

    summary = {
        "requested_samples": sample_count,
        "sampled_episodes": len(sample_rows),
        "checked_samples": checked,
        "positive_reward_disagreements": disagreements,
        "output_path": str(output_path.resolve()),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def run_augmentation(args: argparse.Namespace) -> dict[str, Any]:
    source_index = Path(args.source_index or DEFAULT_SOURCE_INDEX).expanduser().resolve()
    if not source_index.exists():
        raise FileNotFoundError(
            f"Source index not found: {source_index}. Pass --source-index explicitly."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"augmented_{timestamp}"
    output_dir = Path(
        args.output_dir
        or REPO_ROOT / "outputs" / "weak_model_sft_labeling" / run_id
    ).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    backup_dir = Path(
        args.backup_dir
        or REPO_ROOT
        / "outputs"
        / "weak_model_sft_labeling"
        / "backups"
        / f"{source_index.parent.name}_{timestamp}"
    ).expanduser().resolve()

    all_records = _load_jsonl(source_index)
    selected_records = _filter_manifest_records(
        all_records,
        apps=_parse_apps(args.apps),
        max_apps=args.max_apps,
        max_episodes_per_app=args.max_episodes_per_app,
    )
    if not selected_records:
        raise ValueError("No source manifest records selected for augmentation")

    trace_maps, backup_summary = backup_source_records(
        source_index=source_index,
        records=selected_records,
        backup_dir=backup_dir,
    )
    if args.require_backup and not trace_maps:
        raise RuntimeError("Backup completed with zero trace files")

    coverage_cache_dir = output_dir / "coverage_cache"
    episodes = load_backed_up_episodes(
        trace_maps=trace_maps,
        coverage_cache_dir=coverage_cache_dir,
    )
    grouped: dict[str, list[EpisodeRecord]] = {}
    for episode in episodes:
        grouped.setdefault(episode.target_app, []).append(episode)

    all_manifest_rows: list[dict[str, Any]] = []
    all_sft_rows: list[dict[str, Any]] = []
    app_summaries: list[dict[str, Any]] = []
    session_namespace = f"{run_id}-{hashlib.sha256(str(output_dir).encode('utf-8')).hexdigest()[:10]}"

    for target_app in sorted(grouped):
        app_episodes = sorted(grouped[target_app], key=lambda item: (item.source_episode_id, item.source_trace_path))
        manifest_rows, sft_rows = augment_app_episodes(
            target_app=target_app,
            episodes=app_episodes,
            permutations_per_app=args.permutations_per_app,
            seed=args.seed,
            session_namespace=session_namespace,
            dedupe_mode=args.dedupe_mode,
        )
        all_manifest_rows.extend(manifest_rows)
        all_sft_rows.extend(sft_rows)
        app_summaries.append(
            {
                "target_app": target_app,
                "source_episodes": len(app_episodes),
                "synthetic_episodes": len(manifest_rows),
                "accepted_episodes": sum(1 for row in manifest_rows if row.get("accepted")),
                "deduped_episodes": sum(1 for row in manifest_rows if row.get("deduped")),
                "sft_rows": len(sft_rows),
                "reset_rows": sum(1 for row in sft_rows if row.get("action") == RESET_ACTION),
                "coverage_growth": sum(_as_int(row.get("coverage_growth"), 0) for row in manifest_rows),
            }
        )

    all_manifest_rows.sort(key=lambda row: _sort_key(row, MANIFEST_SORT_FIELDS))
    all_sft_rows.sort(key=lambda row: _sort_key(row, SFT_SORT_FIELDS))

    manifest_path = output_dir / "augmentation_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fout:
        for row in all_manifest_rows:
            fout.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    sft_dir = output_dir / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = sft_dir / "augmented_accepted_samples.parquet"
    jsonl_path = sft_dir / "augmented_accepted_samples.jsonl"
    sft_df = pd.DataFrame(all_sft_rows)
    sft_df.to_parquet(parquet_path)
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for row in all_sft_rows:
            fout.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_index": str(source_index),
        "output_dir": str(output_dir),
        "backup_dir": str(backup_dir),
        "backup_manifest": str((backup_dir / "backup_manifest.json").resolve()),
        "selected_manifest_records": len(selected_records),
        "source_episodes": len(episodes),
        "target_apps": len(grouped),
        "permutations_per_app": args.permutations_per_app,
        "synthetic_episodes": len(all_manifest_rows),
        "accepted_episodes": sum(1 for row in all_manifest_rows if row.get("accepted")),
        "deduped_episodes": sum(1 for row in all_manifest_rows if row.get("deduped")),
        "sft_rows": len(all_sft_rows),
        "reset_rows": sum(1 for row in all_sft_rows if row.get("action") == RESET_ACTION),
        "sft_parquet": str(parquet_path.resolve()),
        "sft_jsonl": str(jsonl_path.resolve()),
        "augmentation_manifest": str(manifest_path.resolve()),
        "coverage_cache_dir": str(coverage_cache_dir.resolve()),
        "dedupe_mode": args.dedupe_mode,
        "scorer": "shared-monocart-lines-branches-statements-functions",
        "backup": backup_summary,
        "apps": app_summaries,
    }
    canonical_audit_summary = run_canonical_audit(
        manifest_rows=all_manifest_rows,
        sample_count=int(args.audit_canonical_samples or 0),
        seed=args.seed,
        output_path=output_dir / "audit" / "canonical_reward_audit.jsonl",
    )
    summary["canonical_audit"] = canonical_audit_summary
    _write_json(output_dir / "summary.json", summary)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "augmentation_audit.html").write_text(
        _render_audit_html(summary, app_summaries),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline episode-order augmentation for weak-model SFT traces."
    )
    parser.add_argument("--source-index", default=None, help="Path to all_episode_manifest.jsonl")
    parser.add_argument("--output-dir", default=None, help="Output directory for augmented data")
    parser.add_argument("--backup-dir", default=None, help="Backup directory; defaults under outputs/.../backups")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--augmentation-id", dest="run_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--apps", default=None, help="Comma-separated target app filter")
    parser.add_argument("--max-apps", type=int, default=None)
    parser.add_argument("--max-episodes-per-app", type=int, default=None)
    parser.add_argument("--permutations-per-app", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--dedupe-mode", choices=["prefix", "none"], default="prefix")
    parser.add_argument(
        "--audit-canonical-samples",
        type=int,
        default=0,
        help=(
            "Number of synthetic episodes to audit with canonical Monocart reward "
            "recomputation. Defaults to 0 because full-run baselines can be slow."
        ),
    )
    parser.add_argument("--require-backup", dest="require_backup", action="store_true", default=True)
    parser.add_argument("--no-require-backup", dest="require_backup", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
    summary = run_augmentation(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
