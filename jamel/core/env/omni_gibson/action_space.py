"""
OmniGibson 动作空间模块

具身环境动作：移动基座、转向、手臂/夹爪、与物体交互。
设计为 **文本化函数调用**（如 move_forward()、grasp('apple_0')），供 LLM 输出。
"""

from typing import Dict, List, Any, Union
import re


def get_action_space() -> str:
    """
    获取 OmniGibson 动作空间描述文本（供 LLM 阅读）

    设计原则：
    - 基座移动与转向
    - 手臂与夹爪控制
    - 与场景物体交互（带参数的对象 ID）
    """
    return '''
Note: This action set is for the OmniGibson embodied simulation environment —
a realistic 3D scene with a mobile manipulator robot. You can move, turn, and
interact with objects in the scene.

== Base Movement ==
move_forward()      - Move robot base forward
move_backward()     - Move robot base backward
move_left()         - Strafe left
move_right()        - Strafe right
turn_left()         - Turn base left (yaw)
turn_right()        - Turn base right (yaw)

== Arm & Gripper ==
arm_up()            - Move arm upward
arm_down()          - Move arm downward
arm_forward()       - Move arm forward
arm_back()          - Move arm backward
grasp()             - Close gripper / grasp object in reach
release()            - Open gripper / release object

== Object Interaction (use object_id from observation) ==
interact(object_id) - Interact with object (e.g. open door, push button)
navigate_to(room_id) - Navigate toward a room (room_id from observation)

== Utility ==
noop()              - Do nothing for one step

For example, valid actions are:
move_forward()
turn_left()
grasp()
interact("cabinet_0")
noop()
'''


# 离散动作索引（用于 Mock 或离散化包装）
# 带参数的动作在运行时解析，映射到固定槽位 + 参数
ACTION_INDEX_MAP: Dict[str, int] = {
    'move_forward': 0,
    'move_backward': 1,
    'move_left': 2,
    'move_right': 3,
    'turn_left': 4,
    'turn_right': 5,
    'arm_up': 6,
    'arm_down': 7,
    'arm_forward': 8,
    'arm_back': 9,
    'grasp': 10,
    'release': 11,
    'interact': 12,   # 需要参数 object_id
    'navigate_to': 13,  # 需要参数 room_id
    'noop': 14,
}

ACTION_MAP = ACTION_INDEX_MAP

# 参数化动作的正则
PARAM_ACTION_PATTERN = re.compile(r'^(interact|navigate_to)\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)\s*$', re.I)


def _parse_text_action(text_action: str) -> tuple:
    """
    解析文本动作，返回 (action_name, args_dict)。
    """
    text_action = text_action.strip()
    # 无参动作: "move_forward()"
    if text_action.endswith('()'):
        name = text_action[:-2].strip()
        return name, {}

    # 带参动作: interact("cabinet_0") 或 navigate_to(room_1)
    m = PARAM_ACTION_PATTERN.match(text_action)
    if m:
        name, arg = m.group(1).strip(), m.group(2).strip()
        if name.lower() == 'interact':
            return 'interact', {'object_id': arg}
        if name.lower() == 'navigate_to':
            return 'navigate_to', {'room_id': arg}

    # 兼容：只取括号前的名字
    if '(' in text_action:
        name = text_action.split('(')[0].strip()
        if name in ACTION_INDEX_MAP:
            return name, {}
    return text_action, {}


def text_to_action(text_action: str) -> Union[int, Dict[str, Any]]:
    """
    将文本动作转换为环境可执行表示。

    无参动作返回离散索引 (int)。
    带参动作返回 {"action_index": int, "args": dict}，由 Task 层解析后调用 env。
    """
    name, args = _parse_text_action(text_action)
    if name not in ACTION_INDEX_MAP:
        raise ValueError(
            f"Unknown action '{text_action}'. "
            f"Available: {list(ACTION_INDEX_MAP.keys())}"
        )
    idx = ACTION_INDEX_MAP[name]
    if args:
        return {"action_index": idx, "action_name": name, "args": args}
    return idx


def get_available_actions() -> List[str]:
    """返回所有无参动作名；带参动作以 'interact(object_id)', 'navigate_to(room_id)' 形式说明。"""
    base = [a for a in ACTION_INDEX_MAP.keys() if a not in ('interact', 'navigate_to')]
    return base + ['interact(object_id)', 'navigate_to(room_id)']


def get_num_actions() -> int:
    return len(ACTION_INDEX_MAP)


INDEX_TO_ACTION: Dict[int, str] = {v: k for k, v in ACTION_INDEX_MAP.items()}


def get_action_name(action_index: int) -> str:
    """根据离散动作索引返回动作名。"""
    return INDEX_TO_ACTION.get(int(action_index), "unknown")
