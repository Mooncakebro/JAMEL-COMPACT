from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable

from jamel.core.reward.web.utils import (
    dedupe_coverage_paths,
    monocart_coverage_score_for_paths,
    normalize_coverage_path,
)
from jamel.log import log_utils

logger = log_utils.get_logger(__name__)


@dataclass
class URLCurriculumState:
    target_url: str
    frozen_coverage_paths: list[str] = field(default_factory=list)
    stage_candidate_coverage_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_url": self.target_url,
            "frozen_coverage_paths": list(self.frozen_coverage_paths),
            "stage_candidate_coverage_paths": list(self.stage_candidate_coverage_paths),
        }


def _get_coverage_score(coverage_paths: Iterable[str]) -> int:
    return monocart_coverage_score_for_paths(coverage_paths)


def select_incremental_coverage_paths(
    frozen_coverage_paths: Iterable[str],
    candidate_coverage_paths: Iterable[str],
) -> list[str]:
    accepted_paths = dedupe_coverage_paths(frozen_coverage_paths)
    promoted_paths: list[str] = []
    current_score = _get_coverage_score(accepted_paths)

    for coverage_path in dedupe_coverage_paths(candidate_coverage_paths):
        normalized_path = normalize_coverage_path(coverage_path)
        if normalized_path is None:
            continue
        if not Path(normalized_path).exists():
            logger.warning("Skip promotion candidate because coverage file is missing", coverage_path=normalized_path)
            continue

        next_paths = [*accepted_paths, normalized_path]
        next_score = _get_coverage_score(next_paths)
        if next_score <= current_score:
            continue

        accepted_paths.append(normalized_path)
        promoted_paths.append(normalized_path)
        current_score = next_score

    return promoted_paths


@dataclass
class CurriculumState:
    curriculum_stage_iterations: int
    stage_index: int = 0
    completed_iterations_in_stage: int = 0
    urls: Dict[str, URLCurriculumState] = field(default_factory=dict)

    @classmethod
    def create(cls, target_urls: list[str], curriculum_stage_iterations: int) -> "CurriculumState":
        return cls(
            curriculum_stage_iterations=curriculum_stage_iterations,
            urls={target_url: URLCurriculumState(target_url=target_url) for target_url in target_urls},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        target_urls: list[str],
        curriculum_stage_iterations: int,
        start_iteration_step: int = 0,
    ) -> "CurriculumState":
        state_path = Path(path)
        if not state_path.exists():
            if start_iteration_step > 0:
                raise FileNotFoundError(
                    f"Missing curriculum state file for resume: {state_path}. "
                    "Please resume with the saved curriculum state."
                )
            return cls.create(
                target_urls=target_urls,
                curriculum_stage_iterations=curriculum_stage_iterations,
            )

        with state_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        urls_payload = payload.get("urls", {})
        urls = {}
        for target_url in target_urls:
            url_payload = urls_payload.get(target_url, {})
            urls[target_url] = URLCurriculumState(
                target_url=target_url,
                frozen_coverage_paths=dedupe_coverage_paths(url_payload.get("frozen_coverage_paths", [])),
                stage_candidate_coverage_paths=dedupe_coverage_paths(url_payload.get("stage_candidate_coverage_paths", [])),
            )

        return cls(
            curriculum_stage_iterations=curriculum_stage_iterations,
            stage_index=int(payload.get("stage_index", 0)),
            completed_iterations_in_stage=int(payload.get("completed_iterations_in_stage", 0)),
            urls=urls,
        )

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "curriculum_stage_iterations": self.curriculum_stage_iterations,
            "stage_index": self.stage_index,
            "completed_iterations_in_stage": self.completed_iterations_in_stage,
            "urls": {target_url: state.to_dict() for target_url, state in self.urls.items()},
        }
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def get_frozen_coverage_paths(self, target_url: str) -> list[str]:
        return list(self.urls[target_url].frozen_coverage_paths)

    def add_stage_candidate_coverage_paths(self, target_url: str, coverage_paths: Iterable[str]) -> None:
        url_state = self.urls[target_url]
        url_state.stage_candidate_coverage_paths = dedupe_coverage_paths(
            [*url_state.stage_candidate_coverage_paths, *coverage_paths]
        )

    def complete_iteration(self) -> dict[str, list[str]]:
        self.completed_iterations_in_stage += 1
        if self.completed_iterations_in_stage < self.curriculum_stage_iterations:
            return {}

        promoted_by_url: dict[str, list[str]] = {}
        for target_url, url_state in self.urls.items():
            promoted_paths = select_incremental_coverage_paths(
                frozen_coverage_paths=url_state.frozen_coverage_paths,
                candidate_coverage_paths=url_state.stage_candidate_coverage_paths,
            )
            url_state.frozen_coverage_paths = dedupe_coverage_paths(
                [*url_state.frozen_coverage_paths, *promoted_paths]
            )
            url_state.stage_candidate_coverage_paths = []
            promoted_by_url[target_url] = promoted_paths

        self.stage_index += 1
        self.completed_iterations_in_stage = 0
        return promoted_by_url
