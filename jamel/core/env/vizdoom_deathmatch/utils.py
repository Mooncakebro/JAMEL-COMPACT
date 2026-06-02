"""
VizDoom Deathmatch 工具模块 - 定义 StepHistory 数据类用于标准化日志记录

Deathmatch 场景特有字段：
- 击杀/死亡/武器/弹药等战斗相关状态
- 更丰富的位置和朝向追踪（用于覆盖率计算）
- 完整的战斗事件日志
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class StepHistory:
    """
    VizDoom Deathmatch 单步历史记录

    相比 basic 场景，增加了：
    - 击杀/死亡/碎片计数
    - 武器和弹药追踪
    - 战斗事件标记
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

    player_pos: Optional[Tuple[float, float, float]] = None
    player_angle: Optional[float] = None
    health: Optional[float] = None
    armor: Optional[float] = None
    frag_count: Optional[int] = None
    death_count: Optional[int] = None
    hit_count: Optional[int] = None

    def __post_init__(self):
        if self.info:
            gv = self.info.get('game_variables', {})
            if self.player_pos is None:
                px = gv.get('POSITION_X') or gv.get('position_x')
                py = gv.get('POSITION_Y') or gv.get('position_y')
                pz = gv.get('POSITION_Z') or gv.get('position_z', 0)
                if px is not None and py is not None:
                    self.player_pos = (float(px), float(py), float(pz) if pz else 0.0)
            if self.player_angle is None:
                angle = gv.get('ANGLE') or gv.get('angle')
                if angle is not None:
                    self.player_angle = float(angle)
            if self.health is None:
                self.health = gv.get('HEALTH', gv.get('health'))
            if self.armor is None:
                self.armor = gv.get('ARMOR', gv.get('armor'))
            if self.frag_count is None:
                self.frag_count = int(gv.get('FRAGCOUNT', gv.get('frag_count', 0)))
            if self.death_count is None:
                self.death_count = int(gv.get('DEATHCOUNT', gv.get('death_count', 0)))
            if self.hit_count is None:
                self.hit_count = int(gv.get('HITCOUNT', gv.get('hit_count', 0)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "observation_str": self.observation_str,
            "memory_content": self.memory_content,
            "llm_completion": self.llm_completion,
            "action": self.action,
            "result": self.result,
            "timestamp": str(self.timestamp),
            "player_pos": list(self.player_pos) if self.player_pos else None,
            "player_angle": self.player_angle,
            "health": self.health,
            "armor": self.armor,
            "frag_count": self.frag_count,
            "death_count": self.death_count,
            "hit_count": self.hit_count,
        }


@dataclass
class ExplorationMetrics:
    """
    Deathmatch 探索评测指标

    包含三大维度：
    1. 空间覆盖 - 地图覆盖率、位置多样性
    2. 状态多样性 - 状态熵、动作熵
    3. 新颖性 - 视觉新颖性、RND 新颖性
    4. 战斗表现 - 击杀、存活、伤害
    """
    # 基础
    total_steps: int = 0
    total_reward: float = 0.0
    episode_length: int = 0

    # 空间覆盖
    visited_positions: int = 0
    map_coverage: float = 0.0
    quadrant_coverage: float = 0.0
    exploration_speed: float = 0.0      # 新格子/步

    # 状态多样性
    state_entropy: float = 0.0
    unique_states: int = 0
    action_entropy: float = 0.0
    unique_actions_used: int = 0

    # 新颖性
    avg_visual_novelty: float = 0.0
    max_visual_novelty: float = 0.0
    cumulative_novelty: float = 0.0
    novelty_decay_rate: float = 0.0     # 新颖性随时间衰减速率

    # 战斗
    frags: int = 0
    deaths: int = 0
    kd_ratio: float = 0.0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    items_collected: int = 0
    weapons_picked_up: int = 0
    survival_time: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 4),
            "episode_length": self.episode_length,
            "visited_positions": self.visited_positions,
            "map_coverage": round(self.map_coverage, 6),
            "quadrant_coverage": round(self.quadrant_coverage, 4),
            "exploration_speed": round(self.exploration_speed, 6),
            "state_entropy": round(self.state_entropy, 4),
            "unique_states": self.unique_states,
            "action_entropy": round(self.action_entropy, 4),
            "unique_actions_used": self.unique_actions_used,
            "avg_visual_novelty": round(self.avg_visual_novelty, 6),
            "max_visual_novelty": round(self.max_visual_novelty, 6),
            "cumulative_novelty": round(self.cumulative_novelty, 4),
            "novelty_decay_rate": round(self.novelty_decay_rate, 6),
            "frags": self.frags,
            "deaths": self.deaths,
            "kd_ratio": round(self.kd_ratio, 2),
            "damage_dealt": round(self.damage_dealt, 2),
            "damage_taken": round(self.damage_taken, 2),
            "items_collected": self.items_collected,
            "weapons_picked_up": self.weapons_picked_up,
            "survival_time": self.survival_time,
        }
