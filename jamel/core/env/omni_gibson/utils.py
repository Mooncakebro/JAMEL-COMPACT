"""
OmniGibson 工具模块 - 定义 StepHistory 数据类用于标准化日志记录

具身环境特有字段：
- 机器人位姿（位置、朝向）
- 关节状态、夹爪状态
- 场景中可见/可交互物体列表
- 房间/区域标识（用于覆盖率）
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class StepHistory:
    """
    OmniGibson 单步历史记录

    具身场景字段：
    - 机器人位置、朝向
    - 关节/夹爪状态（若有）
    - 可见物体、交互对象
    - 房间/区域 ID
    """
    step: int
    obs: dict
    info: dict
    observation_str: str
    llm_completion: str
    memory_content: Any
    action: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: Any

    # 具身特有
    robot_pos: Optional[Tuple[float, float, float]] = None
    robot_orientation: Optional[Tuple[float, float, float, float]] = None  # quat or euler
    joint_positions: Optional[List[float]] = None
    gripper_state: Optional[float] = None  # 0=open, 1=closed
    room_id: Optional[str] = None
    visible_objects: Optional[List[Dict[str, Any]]] = None
    interacted_object_id: Optional[str] = None

    def __post_init__(self):
        if self.info:
            rp = self.info.get('robot_position')
            if self.robot_pos is None and rp is not None:
                if isinstance(rp, (list, tuple)) and len(rp) >= 3:
                    self.robot_pos = (float(rp[0]), float(rp[1]), float(rp[2]))
                elif isinstance(rp, np.ndarray):
                    self.robot_pos = tuple(float(x) for x in rp.flat[:3])
            ro = self.info.get('robot_orientation')
            if self.robot_orientation is None and ro is not None:
                if isinstance(ro, (list, tuple)):
                    self.robot_orientation = tuple(float(x) for x in ro[:4])
                elif isinstance(ro, np.ndarray):
                    self.robot_orientation = tuple(float(x) for x in ro.flat[:4])
            if self.room_id is None:
                self.room_id = self.info.get('room_id')
            if self.visible_objects is None:
                self.visible_objects = self.info.get('visible_objects', [])
            if self.gripper_state is None and 'gripper_state' in self.info:
                self.gripper_state = self.info.get('gripper_state')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observation_str": self.observation_str,
            "memory_content": self.memory_content,
            "llm_completion": self.llm_completion,
            "action": self.action,
            "result": self.result,
            "timestamp": str(self.timestamp),
            "robot_pos": list(self.robot_pos) if self.robot_pos else None,
            "robot_orientation": list(self.robot_orientation) if self.robot_orientation else None,
            "joint_positions": self.joint_positions,
            "gripper_state": self.gripper_state,
            "room_id": self.room_id,
            "visible_objects": self.visible_objects,
            "interacted_object_id": self.interacted_object_id,
        }


@dataclass
class ExplorationMetrics:
    """
    OmniGibson 探索评测指标

    维度：
    1. 空间覆盖 - 房间/网格覆盖率、位置多样性
    2. 状态多样性 - 状态熵、动作熵
    3. 新颖性 - 视觉新颖性、位置新颖性
    4. 交互多样性 - 接触物体种类、房间访问数
    """
    # 基础
    total_steps: int = 0
    total_reward: float = 0.0
    episode_length: int = 0

    # 空间覆盖
    visited_positions: int = 0
    map_coverage: float = 0.0
    room_coverage: float = 0.0
    unique_rooms_visited: int = 0
    exploration_speed: float = 0.0  # 新格子/步

    # 状态多样性
    state_entropy: float = 0.0
    unique_states: int = 0
    action_entropy: float = 0.0
    unique_actions_used: int = 0

    # 新颖性
    avg_visual_novelty: float = 0.0
    max_visual_novelty: float = 0.0
    cumulative_novelty: float = 0.0
    novelty_decay_rate: float = 0.0
    avg_position_novelty: float = 0.0

    # 交互
    unique_objects_interacted: int = 0
    interaction_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 4),
            "episode_length": self.episode_length,
            "visited_positions": self.visited_positions,
            "map_coverage": round(self.map_coverage, 6),
            "room_coverage": round(self.room_coverage, 4),
            "unique_rooms_visited": self.unique_rooms_visited,
            "exploration_speed": round(self.exploration_speed, 6),
            "state_entropy": round(self.state_entropy, 4),
            "unique_states": self.unique_states,
            "action_entropy": round(self.action_entropy, 4),
            "unique_actions_used": self.unique_actions_used,
            "avg_visual_novelty": round(self.avg_visual_novelty, 6),
            "max_visual_novelty": round(self.max_visual_novelty, 6),
            "cumulative_novelty": round(self.cumulative_novelty, 4),
            "novelty_decay_rate": round(self.novelty_decay_rate, 6),
            "avg_position_novelty": round(self.avg_position_novelty, 4),
            "unique_objects_interacted": self.unique_objects_interacted,
            "interaction_count": self.interaction_count,
        }
