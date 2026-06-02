from typing import Dict

def get_action_space():
    return '''
Note: This action set allows you to interact with the Crafter environment.
The following actions are available:

noop()
move_left()
move_right()
move_up()
move_down()
do()
sleep()
place_stone()
place_table()
place_furnace()
place_plant()
make_wood_pickaxe()
make_stone_pickaxe()
make_iron_pickaxe()
make_wood_sword()
make_stone_sword()
make_iron_sword()

For example, a valid action is:
move_left()
'''

ACTION_MAP: Dict[str, int] = {
    'noop': 0,
    'move_left': 1,
    'move_right': 2,
    'move_up': 3,
    'move_down': 4,
    'do': 5,
    'sleep': 6,
    'place_stone': 7,
    'place_table': 8,
    'place_furnace': 9,
    'place_plant': 10,
    'make_wood_pickaxe': 11,
    'make_stone_pickaxe': 12,
    'make_iron_pickaxe': 13,
    'make_wood_sword': 14,
    'make_stone_sword': 15,
    'make_iron_sword': 16,
}

def text_to_action(text_action: str) -> int:
    """
    Convert text action (e.g., "move_left()") to integer action index.
    """
    action_name = text_action.split('(')[0].strip()
    if action_name in ACTION_MAP:
        return ACTION_MAP[action_name]
    raise ValueError(f"Unknown action: {text_action}")
