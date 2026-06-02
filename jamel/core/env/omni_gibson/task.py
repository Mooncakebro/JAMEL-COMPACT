"""
OmniGibson 任务模块

封装 OmniGibson 具身环境，统一接口：
  reset()  -> (observation_str, info)
  step(action_str) -> (observation_str, reward, done, truncated, info)

当 omnigibson 未安装时使用 MockOmniGibsonEnv 模拟位置、房间、可见物体与 RGB。
"""

from __future__ import annotations

import os
import logging
from typing import Tuple, Dict, Any, Optional, List, Union

import numpy as np

from jamel.core.env.omni_gibson.observer import Observer
from jamel.core.env.omni_gibson.action_space import (
    text_to_action,
    get_action_space,
    get_available_actions,
    get_action_name,
    get_num_actions,
    ACTION_INDEX_MAP,
)

try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# 尝试导入真实 OmniGibson（Isaac Sim / omnigibson 包）
try:
    import omnigibson as og
    from omnigibson.envs import Environment
    OMNIGIBSON_AVAILABLE = True
except ImportError:
    OMNIGIBSON_AVAILABLE = False


# Mock 场景中的房间与物体池
MOCK_ROOMS = ["living_room", "kitchen", "bedroom", "bathroom", "office"]
MOCK_OBJECTS = [
    {"object_id": "apple_0", "name": "apple", "category": "food", "distance": 1.2, "region": "left"},
    {"object_id": "cabinet_0", "name": "cabinet", "category": "furniture", "distance": 2.0, "region": "center"},
    {"object_id": "door_0", "name": "door", "category": "furniture", "distance": 3.0, "region": "right"},
    {"object_id": "table_0", "name": "table", "category": "furniture", "distance": 1.5, "region": "center"},
    {"object_id": "chair_0", "name": "chair", "category": "furniture", "distance": 2.2, "region": "left"},
    {"object_id": "laptop_0", "name": "laptop", "category": "electronic", "distance": 0.8, "region": "center"},
    {"object_id": "cup_0", "name": "cup", "category": "container", "distance": 1.0, "region": "right"},
]


class MockOmniGibsonEnv:
    """
    OmniGibson 模拟环境：无 Isaac Sim 时的可测试替代。

    - 3D 位置与朝向，多房间切换
    - 随机可见物体列表
    - 简单奖励：进入新房间、接近物体、执行 interact
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._pos = np.array([0.0, 0.0, 0.0])
        self._yaw = 0.0  # 度
        self._gripper = 0.0  # 0=open, 1=closed
        self._room_index = 0
        self._step = 0
        self._done = False
        self._max_steps = 500
        self._img_h, self._img_w = 64, 64
        self._rooms_visited = {0}
        self._cells_visited: set = set()
        self._grid_res = 0.5

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self._pos = np.array([0.0, 0.0, 0.0])
        self._yaw = 0.0
        self._gripper = 0.0
        self._room_index = 0
        self._step = 0
        self._done = False
        self._rooms_visited = {0}
        self._cells_visited = {self._cell_key()}
        return self._get_obs(), self._get_info()

    def step(self, action: Union[int, Dict[str, Any]]) -> Tuple[Dict, float, bool, Dict]:
        self._step += 1
        reward = 0.0
        idx = action if isinstance(action, int) else action.get("action_index", 0)
        args = action.get("args", {}) if isinstance(action, dict) else {}

        speed = 0.3
        rad = np.radians(self._yaw)
        dx = speed * np.cos(rad)
        dy = speed * np.sin(rad)
        side_x = speed * np.cos(rad + np.pi / 2)
        side_y = speed * np.sin(rad + np.pi / 2)

        if idx == 0:   # move_forward
            self._pos[0] += dx
            self._pos[1] += dy
        elif idx == 1:  # move_backward
            self._pos[0] -= dx
            self._pos[1] -= dy
        elif idx == 2:  # move_left
            self._pos[0] += side_x
            self._pos[1] += side_y
        elif idx == 3:  # move_right
            self._pos[0] -= side_x
            self._pos[1] -= side_y
        elif idx == 4:  # turn_left
            self._yaw = (self._yaw + 15) % 360
        elif idx == 5:  # turn_right
            self._yaw = (self._yaw - 15) % 360
        elif idx == 10:  # grasp
            self._gripper = 1.0
            reward += 0.1
        elif idx == 11:  # release
            self._gripper = 0.0
        elif idx == 12:  # interact
            reward += 0.2
        elif idx == 13:  # navigate_to
            # 随机跳到一个房间
            self._room_index = self._rng.randint(0, len(MOCK_ROOMS))
            self._rooms_visited.add(self._room_index)
            self._pos[0] = self._rng.uniform(-2, 2)
            self._pos[1] = self._rng.uniform(-2, 2)
            reward += 0.3

        self._pos[0] = np.clip(self._pos[0], -5, 5)
        self._pos[1] = np.clip(self._pos[1], -5, 5)
        self._cells_visited.add(self._cell_key())
        reward += 0.001  # 生存奖励
        self._done = self._step >= self._max_steps
        return self._get_obs(), reward, self._done, self._get_info()

    def _cell_key(self) -> Tuple[int, int]:
        return (int(self._pos[0] / self._grid_res), int(self._pos[1] / self._grid_res))

    def _get_obs(self) -> Dict[str, np.ndarray]:
        rgb = self._rng.randint(0, 256, (self._img_h, self._img_w, 3), dtype=np.uint8)
        return {"rgb": rgb, "depth": np.full((self._img_h, self._img_w), 2.0, dtype=np.float32)}

    def _get_info(self) -> Dict[str, Any]:
        n_visible = self._rng.randint(2, 6)
        visible = [dict(MOCK_OBJECTS[self._rng.randint(0, len(MOCK_OBJECTS))]) for _ in range(n_visible)]
        for i, v in enumerate(visible):
            v["distance"] = self._rng.uniform(0.5, 3.0)
            v["region"] = self._rng.choice(["left", "center", "right"])
        room_id = MOCK_ROOMS[self._room_index]
        return {
            "robot_position": self._pos.tolist(),
            "robot_orientation": [0, 0, np.sin(np.radians(self._yaw) / 2), np.cos(np.radians(self._yaw) / 2)],
            "gripper_state": self._gripper,
            "room_id": room_id,
            "room_name": room_id.replace("_", " ").title(),
            "visible_objects": visible,
            "step_count": self._step,
            "joint_positions_summary": "arm nominal",
        }

    def close(self):
        pass


class OmniGibsonTask:
    """
    OmniGibson 环境任务封装

    统一接口：
      reset()  -> (observation_str, info)
      step(action_str) -> (observation_str, reward, done, truncated, info)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = 500,
        use_real_env: bool = False,
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.action_space_desc = get_action_space()
        self.available_actions = get_available_actions()

        if use_real_env and OMNIGIBSON_AVAILABLE:
            self.env = self._create_real_env()
        else:
            if use_real_env and not OMNIGIBSON_AVAILABLE:
                logger.warning("OmniGibson not available — using MockOmniGibsonEnv")
            self.env = MockOmniGibsonEnv(seed=seed)
            if hasattr(self.env, "_max_steps"):
                self.env._max_steps = max_steps

        self._last_obs: Optional[Dict] = None
        self._last_info: Optional[Dict] = None
        self._step_count = 0

    def _create_real_env(self):
        # 占位：真实 OmniGibson 需配置文件与场景
        logger.warning("Real OmniGibson env not configured — using mock")
        return MockOmniGibsonEnv(seed=self.seed)

    def reset(self) -> Tuple[str, Dict]:
        self._step_count = 0
        obs, info = self.env.reset()
        self._last_obs = obs
        self._last_info = info
        return Observer.get_observation(obs, info), info

    def step(self, action_str: str) -> Tuple[str, float, bool, bool, Dict]:
        self._step_count += 1
        try:
            raw = text_to_action(action_str)
        except ValueError as e:
            err_info = {
                "error": str(e),
                "valid_actions": self.available_actions,
            }
            return f"Invalid action: {action_str}. {e}", 0.0, False, False, err_info

        if isinstance(raw, dict):
            action_for_env = raw.get("action_index", 0)
            action_dict = dict(raw)
        else:
            action_for_env = raw
            action_dict = {"action_index": raw, "action_name": get_action_name(raw), "args": {}}
        action_dict.setdefault("func_name", action_dict.get("action_name", get_action_name(action_for_env)))

        obs, reward, done, info = self.env.step(action_for_env)
        self._last_obs = obs
        self._last_info = info
        info["last_action"] = action_str
        info["last_reward"] = reward
        info["step_count"] = self._step_count
        info["_action_dict"] = action_dict  # 供 StepHistory 使用

        obs_str = Observer.get_observation(obs, info)
        truncated = self._step_count >= self.max_steps
        return obs_str, reward, done, truncated, info

    def get_action_space_description(self) -> str:
        return self.action_space_desc

    def get_available_actions(self) -> List[str]:
        return self.available_actions

    def get_scenario_description(self) -> str:
        return (
            "OmniGibson — embodied simulation with a mobile manipulator in realistic "
            "3D indoor scenes. Move, turn, use arm and gripper, and interact with objects."
        )

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_omnigibson_env(seed: Optional[int] = None, **kwargs) -> OmniGibsonTask:
    """便捷工厂函数"""
    return OmniGibsonTask(seed=seed, **kwargs)
