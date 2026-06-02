"""
Vizdoom 工具模块 - 定义 StepHistory 数据类用于标准化日志记录
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class StepHistory:
    """
    Vizdoom 环境的单步历史记录
    
    用于记录每一步的完整信息，支持轨迹保存和分析
    """
    step: int                                    # 当前步数
    obs: dict                                    # 原始观察数据 (包含 image, game_variables 等)
    info: dict                                   # 环境返回的额外信息
    observation_str: str                         # LLM 可读的观察文本
    llm_completion: str                          # LLM 的输出 (如果使用 LLM agent)
    memory_content: Any                          # 记忆内容 (用于带记忆的 agent)
    action: Dict[str, Any]                       # 执行的动作 {"func_name": str, "args": dict}
    result: Dict[str, Any]                       # 执行结果 {"reward": float, "done": bool, ...}
    timestamp: Any                               # 时间戳
    
    # 探索相关的额外字段
    player_pos: Optional[Tuple[float, float, float]] = None   # (x, y, z) 坐标
    player_angle: Optional[float] = None                      # 玩家朝向角度
    visited_sectors: Optional[set] = None                     # 访问过的扇区 (用于地图覆盖)
    
    def __post_init__(self):
        """初始化后处理，从 info 中提取位置信息"""
        if self.player_pos is None and self.info:
            # 尝试从 game_variables 子字典中获取
            game_vars = self.info.get('game_variables', {})
            
            # 尝试多种可能的 key 名称
            pos_x = (game_vars.get('POSITION_X') or game_vars.get('position_x') or 
                     self.info.get('POSITION_X') or self.info.get('position_x'))
            pos_y = (game_vars.get('POSITION_Y') or game_vars.get('position_y') or 
                     self.info.get('POSITION_Y') or self.info.get('position_y'))
            pos_z = (game_vars.get('POSITION_Z') or game_vars.get('position_z') or 
                     self.info.get('POSITION_Z') or self.info.get('position_z') or 0)
            
            if pos_x is not None and pos_y is not None:
                self.player_pos = (float(pos_x), float(pos_y), float(pos_z) if pos_z else 0.0)
        
        if self.player_angle is None and self.info:
            game_vars = self.info.get('game_variables', {})
            angle = (game_vars.get('ANGLE') or game_vars.get('angle') or 
                     self.info.get('ANGLE') or self.info.get('angle'))
            if angle is not None:
                self.player_angle = float(angle)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为可序列化的字典
        只包含可以 JSON/Parquet 序列化的数据
        """
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
        }


@dataclass 
class ExplorationMetrics:
    """
    探索评测指标数据类
    """
    # 基础指标
    total_steps: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    
    # 覆盖率指标
    visited_positions: int = 0               # 访问的唯一位置数
    map_coverage: float = 0.0                # 地图覆盖率 (0-1)
    sector_coverage: float = 0.0             # 扇区覆盖率
    
    # 状态多样性指标
    state_entropy: float = 0.0               # 状态分布熵
    unique_states: int = 0                   # 唯一状态数量
    action_entropy: float = 0.0              # 动作分布熵
    unique_actions_used: int = 0             # 使用的唯一动作数
    
    # 新颖性指标
    avg_visual_novelty: float = 0.0          # 平均视觉新颖性
    max_visual_novelty: float = 0.0          # 最大视觉新颖性
    cumulative_novelty: float = 0.0          # 累计新颖性
    
    # 游戏特定指标
    kills: int = 0                           # 击杀数
    items_collected: int = 0                 # 收集物品数
    secrets_found: int = 0                   # 发现的秘密数
    damage_dealt: float = 0.0                # 造成伤害
    damage_taken: float = 0.0                # 受到伤害
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 4),
            "episode_length": self.episode_length,
            "visited_positions": self.visited_positions,
            "map_coverage": round(self.map_coverage, 6),
            "sector_coverage": round(self.sector_coverage, 6),
            "state_entropy": round(self.state_entropy, 4),
            "unique_states": self.unique_states,
            "action_entropy": round(self.action_entropy, 4),
            "unique_actions_used": self.unique_actions_used,
            "avg_visual_novelty": round(self.avg_visual_novelty, 6),
            "max_visual_novelty": round(self.max_visual_novelty, 6),
            "cumulative_novelty": round(self.cumulative_novelty, 4),
            "kills": self.kills,
            "items_collected": self.items_collected,
            "secrets_found": self.secrets_found,
            "damage_dealt": round(self.damage_dealt, 2),
            "damage_taken": round(self.damage_taken, 2),
        }
