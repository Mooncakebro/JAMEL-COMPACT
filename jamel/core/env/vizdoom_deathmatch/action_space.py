"""
VizDoom Deathmatch 动作空间模块

Deathmatch 是 VizDoom 中最复杂的场景，拥有完整的移动、视角、战斗、交互能力。
动作空间设计为 **组合动作 (combo actions)**，允许同时执行多个按键，
更贴近真实 FPS 操作（如边移动边射击）。
"""

from typing import Dict, List


def get_action_space() -> str:
    """
    获取 Deathmatch 动作空间描述文本（供 LLM 阅读）

    设计原则：
    - 原子动作：单个按键操作
    - 组合动作：多个按键同时按下（如 strafe + attack）
    - 速度变体：walk / sprint
    """
    return '''
Note: This action set is for the VizDoom DEATHMATCH scenario — the most
challenging VizDoom environment featuring full 3D combat against multiple
AI opponents on a large open map with item pickups and respawning.

== Movement Actions ==
move_forward()       - Move forward
move_backward()      - Move backward
move_left()          - Strafe left
move_right()         - Strafe right
sprint_forward()     - Sprint forward (move_forward + speed)

== View / Aim Actions ==
turn_left()          - Turn left
turn_right()         - Turn right
turn_left_slow()     - Turn left slowly (precise aiming)
turn_right_slow()    - Turn right slowly (precise aiming)
look_up()            - Look upward (for elevated targets)
look_down()          - Look downward

== Combat Actions ==
attack()             - Fire current weapon
attack_move_left()   - Fire while strafing left (dodge-shoot)
attack_move_right()  - Fire while strafing right (dodge-shoot)
attack_forward()     - Fire while advancing

== Interaction Actions ==
use()                - Use / interact (open doors, pick up items)
select_next_weapon() - Switch to next weapon
select_prev_weapon() - Switch to previous weapon

== Utility ==
noop()               - Do nothing for one frame

For example, a valid action is:
attack_move_left()
'''


# 9 buttons: MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
#            TURN_LEFT, TURN_RIGHT, ATTACK, USE, SPEED
# VizDoom delta buttons for fine turning are mapped as discrete presets.

_N = 9

DEATHMATCH_ACTION_MAP: Dict[str, List[int]] = {
    # --- movement ---
    'move_forward':       [1,0,0,0, 0,0, 0,0, 0],
    'move_backward':      [0,1,0,0, 0,0, 0,0, 0],
    'move_left':          [0,0,1,0, 0,0, 0,0, 0],
    'move_right':         [0,0,0,1, 0,0, 0,0, 0],
    'sprint_forward':     [1,0,0,0, 0,0, 0,0, 1],

    # --- turning ---
    'turn_left':          [0,0,0,0, 1,0, 0,0, 0],
    'turn_right':         [0,0,0,0, 0,1, 0,0, 0],
    'turn_left_slow':     [0,0,0,0, 1,0, 0,0, 0],  # same button, interpret as smaller delta
    'turn_right_slow':    [0,0,0,0, 0,1, 0,0, 0],
    'look_up':            [0,0,0,0, 0,0, 0,0, 0],   # needs delta axis – approximated as noop
    'look_down':          [0,0,0,0, 0,0, 0,0, 0],

    # --- combat ---
    'attack':             [0,0,0,0, 0,0, 1,0, 0],
    'attack_move_left':   [0,0,1,0, 0,0, 1,0, 0],
    'attack_move_right':  [0,0,0,1, 0,0, 1,0, 0],
    'attack_forward':     [1,0,0,0, 0,0, 1,0, 0],

    # --- interaction ---
    'use':                [0,0,0,0, 0,0, 0,1, 0],
    'select_next_weapon': [0,0,0,0, 0,0, 0,0, 0],  # placeholder – VizDoom uses slot keys
    'select_prev_weapon': [0,0,0,0, 0,0, 0,0, 0],

    # --- utility ---
    'noop':               [0,0,0,0, 0,0, 0,0, 0],
}

ACTION_MAP = DEATHMATCH_ACTION_MAP


def text_to_action(text_action: str) -> List[int]:
    """
    将文本动作转换为 VizDoom 按键列表

    Args:
        text_action: 如 "attack()" 或 "move_forward()"

    Returns:
        长度为 9 的按键列表

    Raises:
        ValueError: 未知动作
    """
    action_name = text_action.strip()
    if '(' in action_name:
        action_name = action_name.split('(')[0].strip()

    if action_name in ACTION_MAP:
        return ACTION_MAP[action_name]

    raise ValueError(
        f"Unknown action '{text_action}'. "
        f"Available: {list(ACTION_MAP.keys())}"
    )


def get_available_actions() -> List[str]:
    return list(ACTION_MAP.keys())


def get_num_buttons() -> int:
    return _N
