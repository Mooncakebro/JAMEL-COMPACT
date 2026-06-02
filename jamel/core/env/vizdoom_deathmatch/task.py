"""
VizDoom Deathmatch 任务模块

封装 VizDoom Deathmatch 环境 —— VizDoom 中最具挑战性的场景：
- 大型开放地图，多个房间和走廊
- 多个 AI 敌人同时存在并会主动追击
- 武器/弹药/血包散落在地图各处
- 玩家死亡后会在随机位置重生

统一接口：
  reset()  -> (observation_str, info)
  step(action_str) -> (observation_str, reward, done, truncated, info)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import os
import logging

try:
    import vizdoom as vzd
    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False

from jamel.core.env.vizdoom_deathmatch.observer import Observer
from jamel.core.env.vizdoom_deathmatch.action_space import (
    text_to_action,
    get_action_space,
    get_available_actions,
    get_num_buttons,
    ACTION_MAP,
)

try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MockDeathmatchEnv:
    """
    Deathmatch 模拟环境 —— 用于无 VizDoom 安装时的测试。

    模拟特性：
    - 大地图中的随机移动与位置追踪
    - 多个敌人在视野中随机出现/消失
    - 武器拾取和弹药消耗
    - 伤害和被伤害机制
    - 重生逻辑
    """

    _ENEMY_POOL = [
        'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'DoomImp',
        'Demon', 'Cacodemon', 'HellKnight', 'Revenant',
        'Mancubus', 'Archvile',
    ]

    _ITEM_POOL = [
        ('Medikit', 'health'), ('Stimpack', 'health'),
        ('GreenArmor', 'armor'), ('BlueArmor', 'armor'),
        ('Shotgun', 'weapon'), ('SuperShotgun', 'weapon'),
        ('Chaingun', 'weapon'), ('RocketLauncher', 'weapon'),
        ('PlasmaRifle', 'weapon'),
        ('Shell', 'ammo'), ('Clip', 'ammo'), ('RocketAmmo', 'ammo'),
        ('Cell', 'ammo'), ('Backpack', 'ammo'),
        ('Soulsphere', 'health'), ('MegaSphere', 'health'),
        ('InvulnerabilitySphere', 'powerup'), ('BlurSphere', 'powerup'),
        ('Berserk', 'powerup'),
    ]

    def __init__(self, seed: int = None):
        self._rng = np.random.RandomState(seed)
        self._sw, self._sh = 320, 240

        # player state
        self._health = 100.0
        self._armor = 0.0
        self._ammo = {'pistol': 50.0, 'shotgun': 0.0, 'rocket': 0.0, 'cell': 0.0}
        self._selected_weapon = 1
        self._pos = np.array([0.0, 0.0, 0.0])
        self._angle = 0.0
        self._frag_count = 0
        self._death_count = 0
        self._item_count = 0
        self._step = 0
        self._done = False
        self._respawn_cooldown = 0

    def reset(self) -> Dict[str, np.ndarray]:
        self._health = 100.0
        self._armor = 0.0
        self._ammo = {'pistol': 50.0, 'shotgun': 0.0, 'rocket': 0.0, 'cell': 0.0}
        self._selected_weapon = 1
        self._pos = self._rng.uniform(-500, 500, size=3)
        self._pos[2] = 0.0
        self._angle = self._rng.uniform(0, 360)
        self._frag_count = 0
        self._death_count = 0
        self._item_count = 0
        self._step = 0
        self._done = False
        self._respawn_cooldown = 0
        return self._get_observation()

    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict]:
        self._step += 1
        reward = 0.0

        # --- movement ---
        speed = 12.0
        if len(action) >= 9 and action[8]:  # SPEED
            speed = 20.0
        rad = np.radians(self._angle)

        if len(action) >= 1 and action[0]:  # fwd
            self._pos[0] += speed * np.cos(rad)
            self._pos[1] += speed * np.sin(rad)
        if len(action) >= 2 and action[1]:  # back
            self._pos[0] -= speed * 0.6 * np.cos(rad)
            self._pos[1] -= speed * 0.6 * np.sin(rad)
        if len(action) >= 3 and action[2]:  # strafe left
            self._pos[0] += speed * 0.8 * np.cos(rad + np.pi / 2)
            self._pos[1] += speed * 0.8 * np.sin(rad + np.pi / 2)
        if len(action) >= 4 and action[3]:  # strafe right
            self._pos[0] += speed * 0.8 * np.cos(rad - np.pi / 2)
            self._pos[1] += speed * 0.8 * np.sin(rad - np.pi / 2)

        # --- turning ---
        if len(action) >= 5 and action[4]:
            self._angle = (self._angle + 15) % 360
        if len(action) >= 6 and action[5]:
            self._angle = (self._angle - 15) % 360

        # clamp position
        self._pos[0] = np.clip(self._pos[0], -1500, 1500)
        self._pos[1] = np.clip(self._pos[1], -1500, 1500)

        # --- combat ---
        if len(action) >= 7 and action[6]:  # attack
            if self._ammo['pistol'] > 0:
                self._ammo['pistol'] -= 1
                if self._rng.random() < 0.2:
                    self._frag_count += 1
                    reward += 1.0

        # --- use ---
        if len(action) >= 8 and action[7]:
            if self._rng.random() < 0.1:
                self._item_count += 1
                reward += 0.3

        # --- enemy damage to player ---
        if self._rng.random() < 0.08:
            dmg = self._rng.uniform(5, 25)
            absorbed = min(self._armor, dmg * 0.5)
            self._armor -= absorbed
            self._health -= (dmg - absorbed)

        # --- random item pickup ---
        if self._rng.random() < 0.06:
            self._item_count += 1
            pick = self._rng.choice(len(self._ITEM_POOL))
            item_name, item_cat = self._ITEM_POOL[pick]
            if item_cat == 'health':
                self._health = min(200, self._health + self._rng.uniform(10, 50))
            elif item_cat == 'armor':
                self._armor = min(200, self._armor + self._rng.uniform(50, 100))
            elif item_cat == 'ammo':
                key = self._rng.choice(list(self._ammo.keys()))
                self._ammo[key] = min(300, self._ammo[key] + self._rng.uniform(10, 40))
            elif item_cat == 'weapon':
                self._selected_weapon = self._rng.randint(1, 8)
            reward += 0.1

        # --- respawn if dead ---
        if self._health <= 0:
            self._death_count += 1
            reward -= 1.0
            self._health = 100.0
            self._armor = 0.0
            self._ammo = {'pistol': 50.0, 'shotgun': 0.0, 'rocket': 0.0, 'cell': 0.0}
            self._pos = self._rng.uniform(-500, 500, size=3)
            self._pos[2] = 0.0
            self._angle = self._rng.uniform(0, 360)

        # --- living penalty ---
        reward -= 0.001

        if self._step >= 4200:  # ~2 min at 35 fps
            self._done = True

        return self._get_observation(), reward, self._done, self._get_info()

    def _generate_labels(self) -> List[Dict[str, Any]]:
        labels = []
        # enemies
        n_enemies = self._rng.randint(0, 6)
        for i in range(n_enemies):
            name = self._rng.choice(self._ENEMY_POOL)
            x = self._rng.randint(0, self._sw - 40)
            y = self._rng.randint(20, self._sh - 40)
            w = self._rng.randint(15, 80)
            h = self._rng.randint(25, 100)
            labels.append({'object_id': i, 'object_name': name, 'value': i + 1,
                           'x': x, 'y': y, 'width': w, 'height': h})
        # items
        n_items = self._rng.randint(0, 4)
        for j in range(n_items):
            idx = self._rng.randint(0, len(self._ITEM_POOL))
            name = self._ITEM_POOL[idx][0]
            x = self._rng.randint(0, self._sw - 30)
            y = self._rng.randint(self._sh // 2, self._sh - 20)
            w = self._rng.randint(10, 40)
            h = self._rng.randint(10, 40)
            labels.append({'object_id': len(labels), 'object_name': name, 'value': len(labels) + 1,
                           'x': x, 'y': y, 'width': w, 'height': h})
        return labels

    def _get_observation(self) -> Dict[str, np.ndarray]:
        screen = self._rng.randint(0, 256, (self._sh, self._sw, 3), dtype=np.uint8)
        # draw position marker
        ox = int(abs(self._pos[0])) % (self._sw - 20)
        oy = int(abs(self._pos[1])) % (self._sh - 20)
        screen[oy:oy + 20, ox:ox + 20, 1] = 255

        depth = np.full((self._sh, self._sw), 400.0, dtype=np.float32)
        ch, cw = self._sh // 2, self._sw // 2
        depth[ch - 40:ch + 40, cw - 60:cw + 60] = self._rng.uniform(30, 600)

        return {'screen': screen, 'depth': depth, 'labels': None, 'automap': None}

    def _get_info(self) -> Dict[str, Any]:
        return {
            'game_variables': {
                'HEALTH': self._health,
                'ARMOR': self._armor,
                'AMMO2': self._ammo['pistol'],
                'AMMO3': self._ammo['shotgun'],
                'AMMO4': self._ammo['rocket'],
                'AMMO5': self._ammo['cell'],
                'SELECTED_WEAPON': self._selected_weapon,
                'FRAGCOUNT': self._frag_count,
                'DEATHCOUNT': self._death_count,
                'ITEMCOUNT': self._item_count,
                'POSITION_X': self._pos[0],
                'POSITION_Y': self._pos[1],
                'POSITION_Z': self._pos[2],
                'ANGLE': self._angle,
            },
            'labels': self._generate_labels(),
            'step_count': self._step,
        }

    def close(self):
        pass


class DeathmatchTask:
    """
    VizDoom Deathmatch 环境任务封装

    统一接口：
      reset()  -> (observation_str, info)
      step(action_str) -> (observation_str, reward, done, truncated, info)
    """

    def __init__(
        self,
        seed: int = None,
        frame_skip: int = 4,
        screen_resolution: str = "320X240",
        render: bool = False,
        episode_timeout: int = 4200,
        num_bots: int = 8,
    ):
        self.seed = seed
        self.frame_skip = frame_skip
        self.render = render
        self.episode_timeout = episode_timeout
        self.num_bots = num_bots

        self.action_space_desc = get_action_space()
        self.available_actions = get_available_actions()

        if VIZDOOM_AVAILABLE:
            self.game = self._create_game()
        else:
            logger.warning("VizDoom not available — using MockDeathmatchEnv")
            self.game = MockDeathmatchEnv(seed=seed)

        self._last_obs: Optional[Dict] = None
        self._last_info: Optional[Dict] = None
        self._step_count = 0

    def _create_game(self):
        game = vzd.DoomGame()

        # Try deathmatch config first, fallback to manual setup
        dm_cfg = os.path.join(vzd.scenarios_path, "deathmatch.cfg")
        if os.path.exists(dm_cfg):
            game.load_config(dm_cfg)
        else:
            # Manual deathmatch setup
            wad = os.path.join(vzd.scenarios_path, "deathmatch.wad")
            if not os.path.exists(wad):
                wad = os.path.join(vzd.scenarios_path, "freedoom2.wad")
            game.set_doom_scenario_path(wad)
            game.set_doom_map("map01")

        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

        # Buttons
        game.add_available_button(vzd.Button.MOVE_FORWARD)
        game.add_available_button(vzd.Button.MOVE_BACKWARD)
        game.add_available_button(vzd.Button.MOVE_LEFT)
        game.add_available_button(vzd.Button.MOVE_RIGHT)
        game.add_available_button(vzd.Button.TURN_LEFT)
        game.add_available_button(vzd.Button.TURN_RIGHT)
        game.add_available_button(vzd.Button.ATTACK)
        game.add_available_button(vzd.Button.USE)
        game.add_available_button(vzd.Button.SPEED)

        # Game variables
        game.add_available_game_variable(vzd.GameVariable.HEALTH)
        game.add_available_game_variable(vzd.GameVariable.ARMOR)
        game.add_available_game_variable(vzd.GameVariable.AMMO2)
        game.add_available_game_variable(vzd.GameVariable.AMMO3)
        game.add_available_game_variable(vzd.GameVariable.AMMO4)
        game.add_available_game_variable(vzd.GameVariable.AMMO5)
        game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON)
        game.add_available_game_variable(vzd.GameVariable.FRAGCOUNT)
        game.add_available_game_variable(vzd.GameVariable.DEATHCOUNT)
        game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)
        game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
        game.add_available_game_variable(vzd.GameVariable.ANGLE)

        game.set_labels_buffer_enabled(True)
        game.set_depth_buffer_enabled(True)

        game.set_window_visible(self.render)
        game.set_episode_timeout(self.episode_timeout)

        # Deathmatch mode
        game.set_doom_skill(4)  # Hard
        game.set_mode(vzd.Mode.PLAYER)

        if self.seed is not None:
            game.set_seed(self.seed)

        game.init()

        # Add bots
        for _ in range(self.num_bots):
            game.send_game_command("addbot")

        return game

    def reset(self) -> Tuple[str, Dict]:
        self._step_count = 0

        if VIZDOOM_AVAILABLE:
            self.game.new_episode()
            for _ in range(self.num_bots):
                self.game.send_game_command("addbot")
            state = self.game.get_state()
            obs = self._state_to_obs(state)
            info = self._extract_info(state)
        else:
            obs = self.game.reset()
            info = self.game._get_info()

        self._last_obs = obs
        self._last_info = info
        return Observer.get_observation(obs, info), info

    def step(self, action_str: str) -> Tuple[str, float, bool, bool, Dict]:
        self._step_count += 1

        try:
            action = text_to_action(action_str)
        except ValueError as e:
            err_info = {
                'error': str(e),
                'valid_actions': self.available_actions,
                'game_variables': self._last_info.get('game_variables', {}) if self._last_info else {},
            }
            return f"Invalid action: {action_str}. {e}", 0.0, False, False, err_info

        if VIZDOOM_AVAILABLE:
            reward = self.game.make_action(action, self.frame_skip)
            done = self.game.is_episode_finished()
            if not done:
                state = self.game.get_state()
                obs = self._state_to_obs(state)
                info = self._extract_info(state)
            else:
                obs = self._last_obs
                info = self._last_info
                info['episode_finished'] = True
        else:
            obs, reward, done, info = self.game.step(action)

        self._last_obs = obs
        self._last_info = info

        info['last_action'] = action_str
        info['last_reward'] = reward
        info['step_count'] = self._step_count

        obs_str = Observer.get_observation(obs, info)
        truncated = self._step_count >= self.episode_timeout

        return obs_str, reward, done, truncated, info

    def _state_to_obs(self, state) -> Dict[str, np.ndarray]:
        if state is None:
            return {'screen': np.zeros((240, 320, 3), dtype=np.uint8),
                    'depth': None, 'labels': None, 'automap': None}
        return {
            'screen': state.screen_buffer if state.screen_buffer is not None
                      else np.zeros((240, 320, 3), dtype=np.uint8),
            'depth': state.depth_buffer,
            'labels': state.labels_buffer,
            'automap': state.automap_buffer,
        }

    def _extract_info(self, state) -> Dict[str, Any]:
        if state is None:
            return {'game_variables': {}, 'labels': []}

        # Use get_game_variable() by enum to avoid index-order issues
        # when deathmatch.cfg defines its own variable set.
        gv: Dict[str, Any] = {}
        var_map = {
            'HEALTH': vzd.GameVariable.HEALTH,
            'ARMOR': vzd.GameVariable.ARMOR,
            'AMMO2': vzd.GameVariable.AMMO2,
            'AMMO3': vzd.GameVariable.AMMO3,
            'AMMO4': vzd.GameVariable.AMMO4,
            'AMMO5': vzd.GameVariable.AMMO5,
            'SELECTED_WEAPON': vzd.GameVariable.SELECTED_WEAPON,
            'FRAGCOUNT': vzd.GameVariable.FRAGCOUNT,
            'DEATHCOUNT': vzd.GameVariable.DEATHCOUNT,
            'ITEMCOUNT': vzd.GameVariable.ITEMCOUNT,
            'POSITION_X': vzd.GameVariable.POSITION_X,
            'POSITION_Y': vzd.GameVariable.POSITION_Y,
            'POSITION_Z': vzd.GameVariable.POSITION_Z,
            'ANGLE': vzd.GameVariable.ANGLE,
        }
        for name, var_enum in var_map.items():
            try:
                gv[name] = self.game.get_game_variable(var_enum)
            except Exception:
                pass

        labels = []
        if hasattr(state, 'labels') and state.labels is not None:
            for lab in state.labels:
                labels.append({
                    'object_id': lab.object_id,
                    'object_name': lab.object_name,
                    'value': lab.value,
                    'x': lab.x, 'y': lab.y,
                    'width': lab.width, 'height': lab.height,
                })

        return {
            'game_variables': gv,
            'labels': labels,
            'number': state.number if hasattr(state, 'number') else 0,
        }

    def get_action_space_description(self) -> str:
        return self.action_space_desc

    def get_available_actions(self) -> List[str]:
        return self.available_actions

    def get_scenario_description(self) -> str:
        return ("VizDoom Deathmatch — full combat on a large open map with multiple "
                "AI opponents, weapon/item pickups, and player respawning. "
                "The hardest VizDoom scenario.")

    def close(self):
        if hasattr(self.game, 'close'):
            self.game.close()


def make_deathmatch_env(seed: int = None, **kwargs) -> DeathmatchTask:
    """便捷工厂函数"""
    return DeathmatchTask(seed=seed, **kwargs)
