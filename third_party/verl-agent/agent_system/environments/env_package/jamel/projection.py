import re
from typing import List

DEFAULT_ACTION = "noop()"
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def jamel_projection(actions: List[str]):
    projected_actions: list[str] = []
    valids: list[int] = []

    for raw_action in actions:
        if not isinstance(raw_action, str):
            projected_actions.append(DEFAULT_ACTION)
            valids.append(0)
            continue

        action_match = ACTION_PATTERN.search(raw_action)
        extracted_action = action_match.group(1).strip() if action_match else DEFAULT_ACTION

        has_think = "<think>" in raw_action and "</think>" in raw_action
        has_chinese = re.search(r"[\u4e00-\u9fff]", raw_action) is not None
        is_valid = bool(action_match and extracted_action and has_think and not has_chinese)

        projected_actions.append(extracted_action or DEFAULT_ACTION)
        valids.append(int(is_valid))

    return projected_actions, valids
