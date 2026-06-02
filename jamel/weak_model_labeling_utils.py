from __future__ import annotations

import re
from typing import Any


def extract_action_response(response: str) -> tuple[str, str, bool]:
    match = re.search(r"<think>(.*?)</think>\s*<action>(.*?)</action>", response, flags=re.DOTALL)
    if not match:
        return "", "", False
    think = match.group(1).strip()
    action = match.group(2).strip()
    return think, action, bool(action)


def is_action_execution_valid(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    last_action = obs.get("last_action")
    if last_action is None or (isinstance(last_action, str) and not last_action.strip()):
        return False
    last_action_error = obs.get("last_action_error")
    return not last_action_error or "Success." in str(last_action_error)


def select_successful_prefix(history: list[Any]) -> tuple[int | None, str]:
    positive_steps = [int(step.step) for step in history if float(step.reward or 0.0) > 0]
    if not positive_steps:
        return None, "no_positive_reward"

    last_positive_step = max(positive_steps)
    prefix = [step for step in history if int(step.step) <= last_positive_step]
    for step in prefix:
        extra = step.extra_fields or {}
        if not extra.get("action_format_valid"):
            return None, f"invalid_action_format_at_step_{step.step}"
        if not extra.get("action_execution_valid"):
            return None, f"invalid_action_execution_at_step_{step.step}"
        if not extra.get("coverage_exists_at_write"):
            return None, f"missing_coverage_at_step_{step.step}"
        if not extra.get("coverage_sha256"):
            return None, f"missing_coverage_sha256_at_step_{step.step}"
    return last_positive_step, "accepted"
