"""
VizDoom Deathmatch 环境模块

VizDoom 中最具挑战性的场景：大型开放地图上的多人死亡竞赛。
"""

from jamel.core.env.vizdoom_deathmatch.task import DeathmatchTask, make_deathmatch_env
from jamel.core.env.vizdoom_deathmatch.observer import Observer
from jamel.core.env.vizdoom_deathmatch.action_space import (
    get_action_space,
    get_available_actions,
    text_to_action,
    ACTION_MAP,
)
from jamel.core.env.vizdoom_deathmatch.utils import StepHistory, ExplorationMetrics
from jamel.core.env.vizdoom_deathmatch.metrics import DeathmatchMetrics, compute_metrics

__all__ = [
    'DeathmatchTask',
    'make_deathmatch_env',
    'Observer',
    'get_action_space',
    'get_available_actions',
    'text_to_action',
    'ACTION_MAP',
    'StepHistory',
    'ExplorationMetrics',
    'DeathmatchMetrics',
    'compute_metrics',
]
