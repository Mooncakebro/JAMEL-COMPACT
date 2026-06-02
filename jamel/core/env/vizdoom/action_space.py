"""
Vizdoom 动作空间模块 - 定义文本化的函数调用供 LLM 输出
"""

from typing import Dict, List, Optional


def get_action_space(scenario: str = "basic") -> str:
    """
    获取 Vizdoom 动作空间描述
    
    Args:
        scenario: 场景名称，不同场景有不同的动作空间
                 - "basic": 基础场景（移动+射击）
                 - "deadly_corridor": 死亡走廊
                 - "defend_the_center": 中心防御
                 - "health_gathering": 收集生命值
                 - "my_way_home": 回家之路
                 - "deathmatch": 死亡竞赛
    
    Returns:
        动作空间描述文本
    """
    
    if scenario == "basic":
        return '''
Note: This action set allows you to interact with the Vizdoom Basic scenario.
The agent can move left/right and shoot at the monster.
Available actions:

move_left()
    - Move the player to the left

move_right()
    - Move the player to the right

attack()
    - Fire the weapon at the current target

noop()
    - Do nothing for one frame

For example, a valid action is:
attack()
'''

    elif scenario == "deadly_corridor":
        return '''
Note: This action set allows you to interact with the Vizdoom Deadly Corridor scenario.
Navigate a dangerous corridor while avoiding enemies.
Available actions:

move_forward()
    - Move the player forward

move_backward()
    - Move the player backward

move_left()
    - Strafe left

move_right()
    - Strafe right

turn_left()
    - Turn the player left

turn_right()
    - Turn the player right

attack()
    - Fire the weapon

noop()
    - Do nothing for one frame

For example, a valid action is:
move_forward()
'''

    elif scenario == "defend_the_center":
        return '''
Note: This action set allows you to interact with the Vizdoom Defend The Center scenario.
Stand in the center and shoot incoming enemies from all directions.
Available actions:

turn_left()
    - Turn the player left

turn_right()
    - Turn the player right

attack()
    - Fire the weapon

noop()
    - Do nothing for one frame

For example, a valid action is:
turn_left()
'''

    elif scenario == "health_gathering":
        return '''
Note: This action set allows you to interact with the Vizdoom Health Gathering scenario.
Collect health packs to survive as long as possible.
Available actions:

move_forward()
    - Move the player forward

turn_left()
    - Turn the player left

turn_right()
    - Turn the player right

noop()
    - Do nothing for one frame

For example, a valid action is:
move_forward()
'''

    elif scenario == "my_way_home":
        return '''
Note: This action set allows you to interact with the Vizdoom My Way Home scenario.
Navigate through a maze to find the goal (vest).
Available actions:

move_forward()
    - Move the player forward

turn_left()
    - Turn the player left

turn_right()
    - Turn the player right

noop()
    - Do nothing for one frame

For example, a valid action is:
turn_right()
'''

    elif scenario == "deathmatch":
        return '''
Note: This action set allows you to interact with the Vizdoom Deathmatch scenario.
Full combat scenario with all movement and combat actions.
Available actions:

move_forward()
    - Move the player forward

move_backward()
    - Move the player backward

move_left()
    - Strafe left

move_right()
    - Strafe right

turn_left()
    - Turn the player left

turn_right()
    - Turn the player right

attack()
    - Fire the weapon

use()
    - Use/interact with objects (doors, switches)

noop()
    - Do nothing for one frame

For example, a valid action is:
attack()
'''

    else:
        # 默认返回基础动作空间
        return get_action_space("basic")


# 每个场景的动作映射
# Vizdoom 使用 button 索引，这里定义 text -> button list 的映射

# Basic 场景的动作映射（3 buttons: MOVE_LEFT, MOVE_RIGHT, ATTACK）
BASIC_ACTION_MAP: Dict[str, List[int]] = {
    'move_left': [1, 0, 0],      # MOVE_LEFT pressed
    'move_right': [0, 1, 0],     # MOVE_RIGHT pressed
    'attack': [0, 0, 1],         # ATTACK pressed
    'noop': [0, 0, 0],           # No buttons pressed
}

# Deadly Corridor 场景（7 buttons）
DEADLY_CORRIDOR_ACTION_MAP: Dict[str, List[int]] = {
    'move_forward': [1, 0, 0, 0, 0, 0, 0],
    'move_backward': [0, 1, 0, 0, 0, 0, 0],
    'move_left': [0, 0, 1, 0, 0, 0, 0],
    'move_right': [0, 0, 0, 1, 0, 0, 0],
    'turn_left': [0, 0, 0, 0, 1, 0, 0],
    'turn_right': [0, 0, 0, 0, 0, 1, 0],
    'attack': [0, 0, 0, 0, 0, 0, 1],
    'noop': [0, 0, 0, 0, 0, 0, 0],
}

# Defend The Center 场景（3 buttons: TURN_LEFT, TURN_RIGHT, ATTACK）
DEFEND_CENTER_ACTION_MAP: Dict[str, List[int]] = {
    'turn_left': [1, 0, 0],
    'turn_right': [0, 1, 0],
    'attack': [0, 0, 1],
    'noop': [0, 0, 0],
}

# Health Gathering 场景（3 buttons: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT）
HEALTH_GATHERING_ACTION_MAP: Dict[str, List[int]] = {
    'move_forward': [1, 0, 0],
    'turn_left': [0, 1, 0],
    'turn_right': [0, 0, 1],
    'noop': [0, 0, 0],
}

# My Way Home 场景（3 buttons: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT）
MY_WAY_HOME_ACTION_MAP: Dict[str, List[int]] = {
    'move_forward': [1, 0, 0],
    'turn_left': [0, 1, 0],
    'turn_right': [0, 0, 1],
    'noop': [0, 0, 0],
}

# Deathmatch 场景（8 buttons）
DEATHMATCH_ACTION_MAP: Dict[str, List[int]] = {
    'move_forward': [1, 0, 0, 0, 0, 0, 0, 0],
    'move_backward': [0, 1, 0, 0, 0, 0, 0, 0],
    'move_left': [0, 0, 1, 0, 0, 0, 0, 0],
    'move_right': [0, 0, 0, 1, 0, 0, 0, 0],
    'turn_left': [0, 0, 0, 0, 1, 0, 0, 0],
    'turn_right': [0, 0, 0, 0, 0, 1, 0, 0],
    'attack': [0, 0, 0, 0, 0, 0, 1, 0],
    'use': [0, 0, 0, 0, 0, 0, 0, 1],
    'noop': [0, 0, 0, 0, 0, 0, 0, 0],
}

# 场景到动作映射的映射
SCENARIO_ACTION_MAPS: Dict[str, Dict[str, List[int]]] = {
    'basic': BASIC_ACTION_MAP,
    'deadly_corridor': DEADLY_CORRIDOR_ACTION_MAP,
    'defend_the_center': DEFEND_CENTER_ACTION_MAP,
    'health_gathering': HEALTH_GATHERING_ACTION_MAP,
    'my_way_home': MY_WAY_HOME_ACTION_MAP,
    'deathmatch': DEATHMATCH_ACTION_MAP,
}

# 通用动作映射（基础场景）
ACTION_MAP = BASIC_ACTION_MAP


def text_to_action(text_action: str, scenario: str = "basic") -> List[int]:
    """
    将文本动作转换为 Vizdoom 按键列表
    
    Args:
        text_action: 文本形式的动作，如 "move_left()" 或 "attack()"
        scenario: 场景名称
    
    Returns:
        按键列表，如 [1, 0, 0] 表示第一个按键被按下
    
    Raises:
        ValueError: 如果动作无效
    """
    # 提取函数名（移除括号和参数）
    action_name = text_action.strip()
    if '(' in action_name:
        action_name = action_name.split('(')[0].strip()
    
    # 获取对应场景的动作映射
    action_map = SCENARIO_ACTION_MAPS.get(scenario, BASIC_ACTION_MAP)
    
    if action_name in action_map:
        return action_map[action_name]
    
    raise ValueError(f"Unknown action '{text_action}' for scenario '{scenario}'. "
                     f"Available actions: {list(action_map.keys())}")


def get_available_actions(scenario: str = "basic") -> List[str]:
    """
    获取指定场景的可用动作列表
    
    Args:
        scenario: 场景名称
    
    Returns:
        可用动作名称列表
    """
    action_map = SCENARIO_ACTION_MAPS.get(scenario, BASIC_ACTION_MAP)
    return list(action_map.keys())


def get_num_buttons(scenario: str = "basic") -> int:
    """
    获取指定场景的按键数量
    
    Args:
        scenario: 场景名称
    
    Returns:
        按键数量
    """
    action_map = SCENARIO_ACTION_MAPS.get(scenario, BASIC_ACTION_MAP)
    # 获取任意一个动作的长度
    first_action = next(iter(action_map.values()))
    return len(first_action)
