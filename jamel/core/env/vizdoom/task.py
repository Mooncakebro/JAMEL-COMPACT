"""
Vizdoom 任务模块 - 封装 Vizdoom 环境任务，提供统一的接口
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import os
import logging

# 检查 vizdoom 是否可用
try:
    import vizdoom as vzd
    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False
    print("ViZDoom not found. Using MockVizdoomEnv for testing.")

from jamel.core.env.vizdoom.observer import Observer
from jamel.core.env.vizdoom.action_space import (
    text_to_action, 
    get_action_space,
    get_available_actions,
    get_num_buttons,
    SCENARIO_ACTION_MAPS
)

# Fallback logger
try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MockVizdoomEnv:
    """
    Vizdoom 模拟环境，用于在没有安装 vizdoom 时进行测试。
    改进：生成模拟的物体标签 (labels)，使场景描述模块能产生有意义的输出。
    """

    # 不同场景可能出现的物体
    _SCENARIO_OBJECTS = {
        "basic": [
            {"object_name": "Zombieman", "base_x": 160, "base_y": 120},
        ],
        "health_gathering": [
            {"object_name": "Medikit", "base_x": 80, "base_y": 180},
            {"object_name": "Stimpack", "base_x": 240, "base_y": 160},
        ],
        "my_way_home": [
            {"object_name": "GreenArmor", "base_x": 200, "base_y": 140},
        ],
        "deadly_corridor": [
            {"object_name": "ShotgunGuy", "base_x": 100, "base_y": 110},
            {"object_name": "ShotgunGuy", "base_x": 220, "base_y": 130},
            {"object_name": "Medikit", "base_x": 160, "base_y": 200},
        ],
        "defend_the_center": [
            {"object_name": "DoomImp", "base_x": 50, "base_y": 120},
            {"object_name": "DoomImp", "base_x": 270, "base_y": 120},
            {"object_name": "Zombieman", "base_x": 160, "base_y": 100},
        ],
        "deathmatch": [
            {"object_name": "Zombieman", "base_x": 100, "base_y": 120},
            {"object_name": "ShotgunGuy", "base_x": 200, "base_y": 110},
            {"object_name": "Medikit", "base_x": 260, "base_y": 200},
            {"object_name": "Shotgun", "base_x": 50, "base_y": 180},
        ],
    }

    def __init__(self, scenario: str = "basic", seed: int = None):
        self.scenario = scenario
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # 游戏状态
        self._health = 100.0
        self._ammo = 50.0
        self._position = np.array([0.0, 0.0, 0.0])
        self._angle = 0.0
        self._kill_count = 0
        self._item_count = 0
        self._step_count = 0
        self._done = False

        # 屏幕尺寸
        self._screen_width = 320
        self._screen_height = 240

    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self._health = 100.0
        self._ammo = 50.0
        self._position = np.array([0.0, 0.0, 0.0])
        self._angle = 0.0
        self._kill_count = 0
        self._item_count = 0
        self._step_count = 0
        self._done = False

        return self._get_observation()

    def step(self, action: List[int]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 按键列表，如 [1, 0, 0]

        Returns:
            (observation, reward, done, info)
        """
        self._step_count += 1
        reward = 0.0

        # 根据场景和动作更新状态
        if self.scenario == "basic":
            if len(action) >= 3:
                if action[0]:  # move_left
                    self._position[0] -= 5.0
                    reward -= 0.01
                if action[1]:  # move_right
                    self._position[0] += 5.0
                    reward -= 0.01
                if action[2]:  # attack
                    if self._rng.random() < 0.3:
                        self._kill_count += 1
                        reward += 1.0
                    self._ammo = max(0, self._ammo - 1)

        elif self.scenario in ["health_gathering", "my_way_home"]:
            if len(action) >= 3:
                if action[0]:  # move_forward
                    self._position[0] += 10.0 * np.cos(np.radians(self._angle))
                    self._position[1] += 10.0 * np.sin(np.radians(self._angle))
                    reward += 0.01
                if action[1]:  # turn_left
                    self._angle = (self._angle + 10) % 360
                if action[2]:  # turn_right
                    self._angle = (self._angle - 10) % 360

            if self._rng.random() < 0.05:
                self._item_count += 1
                self._health = min(100, self._health + 10)
                reward += 1.0

        else:
            move_delta = 5.0
            for i, pressed in enumerate(action):
                if pressed:
                    self._position[i % 2] += self._rng.uniform(-move_delta, move_delta)

        # 生命值衰减
        if self.scenario == "health_gathering":
            self._health -= 0.5
            if self._health <= 0:
                self._done = True
                reward -= 1.0

        if self._step_count >= 1000:
            self._done = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, self._done, info

    def _generate_mock_labels(self) -> List[Dict[str, Any]]:
        """
        生成模拟的物体标签，让场景描述模块有数据可解析。
        物体位置会随机抖动以模拟动态场景。
        """
        base_objects = self._SCENARIO_OBJECTS.get(self.scenario, [])
        labels = []
        for i, obj in enumerate(base_objects):
            # 随机决定物体是否在视野中（80% 概率可见）
            if self._rng.random() > 0.8:
                continue
            # 位置随机抖动
            jitter_x = self._rng.randint(-30, 31)
            jitter_y = self._rng.randint(-20, 21)
            x = max(0, min(self._screen_width - 40, obj["base_x"] + jitter_x))
            y = max(0, min(self._screen_height - 40, obj["base_y"] + jitter_y))
            # 大小随步数变化（模拟距离变化）
            base_size = 30 + self._rng.randint(-10, 11)
            width = max(10, base_size + self._rng.randint(-5, 6))
            height = max(15, base_size + self._rng.randint(0, 20))
            labels.append({
                'object_id': i,
                'object_name': obj["object_name"],
                'value': i + 1,
                'x': int(x),
                'y': int(y),
                'width': int(width),
                'height': int(height),
            })
        return labels

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """生成观察"""
        screen = self._rng.randint(0, 256,
                                    (self._screen_height, self._screen_width, 3),
                                    dtype=np.uint8)

        # 添加视觉特征
        x_offset = int(abs(self._position[0])) % (self._screen_width - 20)
        y_offset = int(abs(self._position[1])) % (self._screen_height - 20)
        screen[y_offset:y_offset+20, x_offset:x_offset+20, 0] = 255

        # 生成模拟深度图
        depth = np.full((self._screen_height, self._screen_width), 300.0, dtype=np.float32)
        # 中心区域深度随机变化（模拟前方空间）
        center_h, center_w = self._screen_height // 2, self._screen_width // 2
        depth[center_h-40:center_h+40, center_w-60:center_w+60] = self._rng.uniform(50, 500)

        return {
            'screen': screen,
            'depth': depth,
            'labels': None,
            'automap': None,
        }

    def _get_info(self) -> Dict[str, Any]:
        """生成信息字典，包含物体标签"""
        return {
            'game_variables': {
                'HEALTH': self._health,
                'AMMO': self._ammo,
                'ARMOR': 0.0,
                'KILLCOUNT': self._kill_count,
                'ITEMCOUNT': self._item_count,
                'SECRETCOUNT': 0,
                'POSITION_X': self._position[0],
                'POSITION_Y': self._position[1],
                'POSITION_Z': self._position[2],
                'ANGLE': self._angle,
            },
            'labels': self._generate_mock_labels(),
            'step_count': self._step_count,
        }

    def close(self):
        """关闭环境"""
        pass


class VizdoomTask:
    """
    Vizdoom 环境任务封装
    
    提供统一的接口：
    - reset() -> (observation_str, info)
    - step(action_str) -> (observation_str, reward, done, truncated, info)
    """
    
    # 场景配置
    SCENARIOS = {
        "basic": {
            "config": "basic.cfg",
            "description": "Shoot a monster in front of you",
        },
        "deadly_corridor": {
            "config": "deadly_corridor.cfg",
            "description": "Navigate a corridor with enemies",
        },
        "defend_the_center": {
            "config": "defend_the_center.cfg",
            "description": "Defend yourself from enemies coming from all directions",
        },
        "health_gathering": {
            "config": "health_gathering.cfg",
            "description": "Collect health packs to survive",
        },
        "my_way_home": {
            "config": "my_way_home.cfg",
            "description": "Navigate a maze to find the goal",
        },
        "deathmatch": {
            "config": "deathmatch.cfg",
            "description": "Full combat scenario",
        },
    }
    
    def __init__(
        self,
        scenario: str = "basic",
        seed: int = None,
        frame_skip: int = 4,
        screen_resolution: str = "320X240",
        render: bool = False,
    ):
        """
        初始化 Vizdoom 任务
        
        Args:
            scenario: 场景名称
            seed: 随机种子
            frame_skip: 帧跳过数（每个动作持续的帧数）
            screen_resolution: 屏幕分辨率
            render: 是否渲染窗口
        """
        self.scenario = scenario
        self.seed = seed
        self.frame_skip = frame_skip
        self.render = render
        
        # 验证场景
        if scenario not in self.SCENARIOS and scenario not in SCENARIO_ACTION_MAPS:
            logger.warning(f"Unknown scenario '{scenario}', using 'basic'")
            self.scenario = "basic"
        
        # 获取动作空间描述
        self.action_space_desc = get_action_space(self.scenario)
        self.available_actions = get_available_actions(self.scenario)
        
        # 初始化环境
        if VIZDOOM_AVAILABLE:
            self.game = self._create_vizdoom_game()
        else:
            logger.warning("ViZDoom not available, using mock environment")
            self.game = MockVizdoomEnv(scenario=self.scenario, seed=seed)
        
        # 状态追踪
        self._last_obs = None
        self._last_info = None
        self._step_count = 0
    
    def _create_vizdoom_game(self):
        """创建并配置 ViZDoom 游戏实例"""
        game = vzd.DoomGame()
        
        # 加载场景配置
        scenario_info = self.SCENARIOS.get(self.scenario, self.SCENARIOS["basic"])
        config_path = os.path.join(vzd.scenarios_path, scenario_info["config"])
        
        if os.path.exists(config_path):
            game.load_config(config_path)
        else:
            logger.warning(f"Config file not found: {config_path}, using default settings")
            # 使用默认设置
            game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))
            game.set_doom_map("map01")
        
        # 设置屏幕格式
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # 设置分辨率
        resolution_map = {
            "160X120": vzd.ScreenResolution.RES_160X120,
            "320X240": vzd.ScreenResolution.RES_320X240,
            "640X480": vzd.ScreenResolution.RES_640X480,
        }
        res = resolution_map.get("320X240", vzd.ScreenResolution.RES_320X240)
        game.set_screen_resolution(res)
        
        # 启用游戏变量
        game.add_available_game_variable(vzd.GameVariable.HEALTH)
        game.add_available_game_variable(vzd.GameVariable.AMMO2)
        game.add_available_game_variable(vzd.GameVariable.ARMOR)
        game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)
        game.add_available_game_variable(vzd.GameVariable.SECRETCOUNT)
        game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
        game.add_available_game_variable(vzd.GameVariable.ANGLE)
        
        # 启用 labels buffer，用于场景描述（识别可见物体）
        game.set_labels_buffer_enabled(True)
        # 启用 depth buffer，用于估算距离
        game.set_depth_buffer_enabled(True)

        # 设置窗口模式
        game.set_window_visible(self.render)

        # 设置随机种子
        if self.seed is not None:
            game.set_seed(self.seed)

        # 初始化游戏
        game.init()
        
        return game
    
    def reset(self) -> Tuple[str, Dict]:
        """
        重置环境
        
        Returns:
            (observation_str, info)
        """
        self._step_count = 0
        
        if VIZDOOM_AVAILABLE:
            self.game.new_episode()
            state = self.game.get_state()
            obs = self._state_to_obs(state)
            info = self._get_info(state)
        else:
            obs = self.game.reset()
            info = self.game._get_info()
        
        self._last_obs = obs
        self._last_info = info
        
        obs_str = Observer.get_observation(obs, info)
        
        return obs_str, info
    
    def step(self, action_str: str) -> Tuple[str, float, bool, bool, Dict]:
        """
        执行动作
        
        Args:
            action_str: 文本形式的动作，如 "move_left()" 或 "attack()"
        
        Returns:
            (observation_str, reward, done, truncated, info)
        """
        self._step_count += 1
        
        # 将文本动作转换为按键列表
        try:
            action = text_to_action(action_str, self.scenario)
        except ValueError as e:
            # 无效动作，返回错误信息
            error_info = {
                "error": str(e),
                "valid_actions": self.available_actions,
                "game_variables": self._last_info.get('game_variables', {}) if self._last_info else {},
            }
            return f"Invalid action: {action_str}. {str(e)}", 0.0, False, False, error_info
        
        # 执行动作
        if VIZDOOM_AVAILABLE:
            reward = self.game.make_action(action, self.frame_skip)
            done = self.game.is_episode_finished()
            
            if not done:
                state = self.game.get_state()
                obs = self._state_to_obs(state)
                info = self._get_info(state)
            else:
                obs = self._last_obs
                info = self._last_info
                info['episode_finished'] = True
        else:
            obs, reward, done, info = self.game.step(action)
        
        # 更新状态
        self._last_obs = obs
        self._last_info = info
        
        # 添加动作信息到 info
        info['last_action'] = action_str
        info['last_reward'] = reward
        info['step_count'] = self._step_count
        
        # 生成观察文本
        obs_str = Observer.get_observation(obs, info)
        
        # truncated: 是否因为步数限制而终止
        truncated = self._step_count >= 10000
        
        return obs_str, reward, done, truncated, info
    
    def _state_to_obs(self, state) -> Dict[str, np.ndarray]:
        """将 vizdoom state 转换为观察字典"""
        if state is None:
            return {
                'screen': np.zeros((240, 320, 3), dtype=np.uint8),
                'depth': None,
                'labels': None,
                'automap': None,
            }

        return {
            'screen': state.screen_buffer if state.screen_buffer is not None else np.zeros((240, 320, 3), dtype=np.uint8),
            'depth': state.depth_buffer,
            'labels': state.labels_buffer,
            'automap': state.automap_buffer,
        }

    def _get_info(self, state) -> Dict[str, Any]:
        """从 state 提取 info 字典，包含物体标签用于场景描述"""
        if state is None:
            return {'game_variables': {}, 'labels': []}

        # 游戏变量名称
        var_names = [
            'HEALTH', 'AMMO2', 'ARMOR', 'KILLCOUNT', 'ITEMCOUNT',
            'SECRETCOUNT', 'POSITION_X', 'POSITION_Y', 'POSITION_Z', 'ANGLE'
        ]

        game_vars = {}
        if state.game_variables is not None:
            for i, name in enumerate(var_names):
                if i < len(state.game_variables):
                    game_vars[name] = state.game_variables[i]

        # 提取物体标签（核心改进：为场景描述提供结构化数据）
        labels = []
        if hasattr(state, 'labels') and state.labels is not None:
            for label in state.labels:
                labels.append({
                    'object_id': label.object_id,
                    'object_name': label.object_name,
                    'value': label.value,
                    'x': label.x,
                    'y': label.y,
                    'width': label.width,
                    'height': label.height,
                })

        return {
            'game_variables': game_vars,
            'labels': labels,
            'number': state.number if hasattr(state, 'number') else 0,
        }
    
    def get_action_space_description(self) -> str:
        """获取动作空间描述"""
        return self.action_space_desc
    
    def get_available_actions(self) -> List[str]:
        """获取可用动作列表"""
        return self.available_actions
    
    def get_scenario_description(self) -> str:
        """获取场景描述"""
        scenario_info = self.SCENARIOS.get(self.scenario, {})
        return scenario_info.get("description", f"Scenario: {self.scenario}")
    
    def close(self):
        """关闭环境"""
        if hasattr(self.game, 'close'):
            self.game.close()


# 便捷的工厂函数
def make_vizdoom_env(
    scenario: str = "basic",
    seed: int = None,
    **kwargs
) -> VizdoomTask:
    """
    创建 Vizdoom 环境的便捷函数
    
    Args:
        scenario: 场景名称
        seed: 随机种子
        **kwargs: 传递给 VizdoomTask 的其他参数
    
    Returns:
        VizdoomTask 实例
    """
    return VizdoomTask(scenario=scenario, seed=seed, **kwargs)
