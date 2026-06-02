"""
OmniGibson 探索评测指标模块

指标体系：
┌──────────────────────────────────────────────────────────────┐
│ 1. 覆盖率 (Coverage)                                          │
│    - map_coverage: 离散网格覆盖率                              │
│    - room_coverage: 房间/区域覆盖率                            │
│    - exploration_speed: 新格子/步                             │
│                                                               │
│ 2. 状态熵 / 多样性 (State Entropy & Diversity)                 │
│    - state_entropy: H(s) = -Σ p(s) log p(s)                   │
│    - action_entropy: H(a) = -Σ p(a) log p(a)                  │
│    - unique_states / unique_actions_used                      │
│                                                               │
│ 3. 新颖性 (Novelty)                                           │
│    - visual_novelty: 帧间图像差异                              │
│    - position_novelty: 到历史位置最小距离                       │
│    - cumulative_novelty, novelty_decay_rate                  │
│                                                               │
│ 4. 交互多样性 (Interaction Diversity)                         │
│    - unique_objects_interacted, interaction_count             │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Set, Optional

import numpy as np

from jamel.core.env.omni_gibson.utils import StepHistory, ExplorationMetrics


class OmniGibsonMetrics:
    """OmniGibson 探索指标计算器"""

    def __init__(
        self,
        map_bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0),
        grid_resolution: float = 0.5,
        room_ids: Optional[List[str]] = None,
    ):
        self.map_bounds = map_bounds
        self.grid_resolution = grid_resolution
        self.room_ids = room_ids or []

    def compute_all(self, history: List[StepHistory]) -> Dict[str, Any]:
        """计算所有指标并返回合并字典"""
        if not history:
            return ExplorationMetrics().to_dict()

        result: Dict[str, Any] = {}
        result.update(self._basic_stats(history))
        result.update(self._coverage_metrics(history))
        result.update(self._entropy_metrics(history))
        result.update(self._novelty_metrics(history))
        result.update(self._interaction_metrics(history))
        return result

    def _basic_stats(self, history: List[StepHistory]) -> Dict[str, Any]:
        total_reward = sum(s.result.get("reward", 0) for s in history)
        return {
            "total_steps": len(history),
            "episode_length": len(history),
            "total_reward": round(total_reward, 4),
        }

    def _coverage_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        min_x, min_y, max_x, max_y = self.map_bounds
        gr = self.grid_resolution
        gw = max(1, int((max_x - min_x) / gr) + 1)
        gh = max(1, int((max_y - min_y) / gr) + 1)
        total_grid = gw * gh
        total_rooms = max(len(self.room_ids), 1)

        visited_grid: Set[Tuple[int, int]] = set()
        visited_rooms: Set[str] = set()
        new_cells_per_step: List[int] = []

        for s in history:
            new = 0
            if s.robot_pos:
                x, y = s.robot_pos[0], s.robot_pos[1]
                gx = max(0, min(int((x - min_x) / gr), gw - 1))
                gy = max(0, min(int((y - min_y) / gr), gh - 1))
                if (gx, gy) not in visited_grid:
                    new = 1
                visited_grid.add((gx, gy))
            if s.room_id:
                visited_rooms.add(s.room_id)
            new_cells_per_step.append(new)

        return {
            "visited_positions": len(visited_grid),
            "map_coverage": round(len(visited_grid) / total_grid, 6),
            "unique_rooms_visited": len(visited_rooms),
            "room_coverage": round(len(visited_rooms) / total_rooms, 4),
            "exploration_speed": round(sum(new_cells_per_step) / max(1, len(history)), 6),
        }

    def _entropy_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        min_x, min_y = self.map_bounds[0], self.map_bounds[1]
        gr = self.grid_resolution
        state_counts: Counter = Counter()
        action_counts: Counter = Counter()

        for s in history:
            pos_bin = (0, 0)
            if s.robot_pos:
                pos_bin = (
                    int((s.robot_pos[0] - min_x) / gr),
                    int((s.robot_pos[1] - min_y) / gr),
                )
            room = s.room_id or "unknown"
            state_key = (pos_bin, room)
            state_counts[state_key] += 1
            act = (
                s.action.get("func_name", s.action.get("action_name", "unknown"))
                if isinstance(s.action, dict)
                else str(s.action)
            )
            action_counts[act] += 1

        return {
            "unique_states": len(state_counts),
            "state_entropy": round(self._entropy(state_counts), 4),
            "unique_actions_used": len(action_counts),
            "action_entropy": round(self._entropy(action_counts), 4),
        }

    @staticmethod
    def _entropy(counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total + 1e-10) for c in counts.values())

    def _novelty_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        visual_scores: List[float] = []
        img_buf: List[np.ndarray] = []
        for s in history:
            screen = s.obs.get("rgb")
            if screen is None:
                screen = s.obs.get("image")
            if screen is None:
                screen = s.obs.get("screen")
            if screen is not None and isinstance(screen, np.ndarray):
                ds = screen[::4, ::4, :].astype(np.float32) if screen.ndim >= 3 else np.zeros((8, 8, 3), dtype=np.float32)
                if screen.ndim == 2:
                    ds = np.stack([screen[::4, ::4].astype(np.float32)] * 3, axis=-1)
                if img_buf:
                    diffs = [
                        np.mean((ds - h.astype(np.float32)) ** 2) / (255.0 ** 2 + 1e-10)
                        for h in img_buf[-50:]
                    ]
                    visual_scores.append(float(min(diffs)))
                img_buf.append(ds.copy())

        avg_vis = float(np.mean(visual_scores)) if visual_scores else 0.0
        max_vis = float(max(visual_scores)) if visual_scores else 0.0
        cum_vis = float(sum(visual_scores)) if visual_scores else 0.0
        decay_rate = 0.0
        if len(visual_scores) > 10:
            x = np.arange(len(visual_scores), dtype=float)
            y = np.array(visual_scores)
            x_mean, y_mean = np.mean(x), np.mean(y)
            decay_rate = float(
                (np.mean(x * y) - x_mean * y_mean) / (np.mean(x ** 2) - x_mean ** 2 + 1e-10)
            )

        pos_novelty_scores: List[float] = []
        pos_hist: List[Tuple[float, float]] = []
        for s in history:
            if s.robot_pos:
                cur = (s.robot_pos[0], s.robot_pos[1])
                if pos_hist:
                    min_d = min(
                        math.sqrt((cur[0] - p[0]) ** 2 + (cur[1] - p[1]) ** 2)
                        for p in pos_hist[-200:]
                    )
                    pos_novelty_scores.append(min_d)
                pos_hist.append(cur)
        avg_pos_nov = float(np.mean(pos_novelty_scores)) if pos_novelty_scores else 0.0

        return {
            "avg_visual_novelty": round(avg_vis, 6),
            "max_visual_novelty": round(max_vis, 6),
            "cumulative_novelty": round(cum_vis, 4),
            "novelty_decay_rate": round(decay_rate, 8),
            "avg_position_novelty": round(avg_pos_nov, 4),
        }

    def _interaction_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        interacted: Set[str] = set()
        count = 0
        for s in history:
            oid = None
            if isinstance(s.action, dict):
                oid = (s.action.get("args") or {}).get("object_id")
                if s.action.get("action_name") in ("interact", "grasp", "release"):
                    count += 1
            if s.interacted_object_id:
                oid = s.interacted_object_id
            if oid:
                interacted.add(str(oid))
        return {
            "unique_objects_interacted": len(interacted),
            "interaction_count": count,
        }


def compute_metrics(
    history: List[StepHistory],
    map_bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0),
    grid_resolution: float = 0.5,
    room_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """便捷函数：一步计算所有指标"""
    calculator = OmniGibsonMetrics(
        map_bounds=map_bounds,
        grid_resolution=grid_resolution,
        room_ids=room_ids,
    )
    return calculator.compute_all(history)
