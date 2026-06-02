from typing import Any, List, Optional, Sequence

from jamel.core.env.web.utils import StepHistory
from jamel.core.reward.jaccard import jamel_reward_fn_jaccard
from jamel.log import log_utils
from .utils import (
    collect_history_coverage_paths,
    compute_monocart_coverage_reward_details,
    dedupe_coverage_paths,
    get_step_coverage_path,
    normalize_coverage_path,
)

logger = log_utils.get_logger(__name__)


def _is_action_format_valid(obs: Optional[dict]) -> bool:
    """
    Use env execution feedback to decide if the action format is valid.
    If the env reports an error or the action is empty, treat it as invalid.
    """
    if not isinstance(obs, dict):
        logger.warning(f"found invalid obs type", type=type(obs))
        return False

    last_action_error = obs.get("last_action_error")
    if last_action_error and 'Success.' not in last_action_error:
        logger.warning(f"found last action error", last_action_error=last_action_error)
        return False

    last_action = obs.get("last_action")
    if last_action is None:
        logger.warning(f"last action not found", last_action=last_action)
        return False
    if isinstance(last_action, str) and not last_action.strip():
        logger.warning(f"last action not found", last_action=last_action)
        return False

    return True


def jamel_reward_fn_web_coverage(
    current_step: Optional[StepHistory],
    frozen_global_coverage_paths: Optional[Sequence[str]] = None,
    trajectory_history: Optional[Sequence[StepHistory]] = None,
) -> float:
    return float(
        jamel_reward_fn_web_coverage_details(
            current_step,
            frozen_global_coverage_paths=frozen_global_coverage_paths,
            trajectory_history=trajectory_history,
        )["reward"]
    )


def jamel_reward_fn_web_coverage_details(
    current_step: Optional[StepHistory],
    frozen_global_coverage_paths: Optional[Sequence[str]] = None,
    trajectory_history: Optional[Sequence[StepHistory]] = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "reward": 0.0,
        "previous_score": 0,
        "current_score": 0,
        "delta_score": 0,
        "skip_reason": None,
    }

    if current_step is None:
        details["skip_reason"] = "missing_current_step"
        return details

    if not _is_action_format_valid(current_step.after_obs):
        details["skip_reason"] = "invalid_action"
        return details

    current_coverage_path = get_step_coverage_path(current_step)
    normalized_current_path = normalize_coverage_path(current_coverage_path)
    if normalized_current_path is None:
        logger.warning("Reward skipped: current step missing coverage path", step=current_step.step)
        details["skip_reason"] = "missing_current_coverage_path"
        return details

    if not current_coverage_path.exists():
        logger.warning(
            "Reward skipped: current step coverage file missing",
            step=current_step.step,
            coverage_path=current_coverage_path,
        )
        details["skip_reason"] = "missing_current_coverage_file"
        return details

    frozen_paths = dedupe_coverage_paths(frozen_global_coverage_paths or [])
    trajectory_paths = collect_history_coverage_paths(trajectory_history)
    baseline_paths = dedupe_coverage_paths([*frozen_paths, *trajectory_paths])

    reward_details = compute_monocart_coverage_reward_details(
        current_path=normalized_current_path,
        baseline_paths=baseline_paths,
    )
    details.update(reward_details)
    logger.info(
        "Coverage reward computed",
        step=current_step.step,
        frozen_history_count=len(frozen_paths),
        trajectory_history_count=len(trajectory_paths),
        previous_score=details["previous_score"],
        current_score=details["current_score"],
        delta_score=details["delta_score"],
        reward=details["reward"],
    )
    return details


def jamel_reward_fn_web_jaccard(
    current_step: StepHistory,
    history: List[StepHistory],
) -> float:
    if current_step is None:
        return 0.0

    if not _is_action_format_valid(current_step.after_obs):
        return 0.0

    past_observations = [item.after_observation for item in history] if history else []
    reward = jamel_reward_fn_jaccard(current_step.after_observation, past_observations)
    logger.info(
        "Jaccard reward computed",
        step=current_step.step,
        reward=reward,
    )
    return reward
