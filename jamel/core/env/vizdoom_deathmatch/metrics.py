"""
VizDoom Deathmatch 探索评测指标模块

提供独立的指标计算器，可作为 Observer.calculate_metrics 的补充，
也可独立使用。

指标体系：
┌──────────────────────────────────────────────────────────────┐
│ 1. 覆盖率 (Coverage)                                        │
│    - map_coverage: 离散网格覆盖率                            │
│    - quadrant_coverage: 4 象限覆盖率                         │
│    - room_coverage: 房间/区域级覆盖（大网格）                 │
│                                                              │
│ 2. 状态熵 / 多样性 (State Entropy & Diversity)               │
│    - state_entropy: H(s) = -Σ p(s) log p(s)                │
│    - action_entropy: H(a) = -Σ p(a) log p(a)               │
│    - state_action_entropy: H(s,a)                            │
│    - unique_states / unique_actions                          │
│                                                              │
│ 3. 新颖性 (Novelty)                                         │
│    - visual_novelty: 帧间 MSE 差异                           │
│    - position_novelty: 当前位置到历史位置的最小距离           │
│    - cumulative_novelty: 新颖性积分                          │
│    - novelty_half_life: 新颖性降至一半所需步数               │
│                                                              │
│ 4. 战斗表现 (Combat Performance)                             │
│    - frags, deaths, K/D ratio                                │
│    - accuracy_estimate                                       │
│    - survival_rate                                           │
└──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Any, List, Tuple, Set, Optional

import numpy as np

from jamel.core.env.vizdoom_deathmatch.utils import StepHistory, ExplorationMetrics


class DeathmatchMetrics:
    """Deathmatch 探索指标计算器"""

    def __init__(
        self,
        map_bounds: Tuple[float, float, float, float] = (-2000, -2000, 2000, 2000),
        grid_resolution: float = 64.0,
        room_resolution: float = 256.0,
    ):
        self.map_bounds = map_bounds
        self.grid_resolution = grid_resolution
        self.room_resolution = room_resolution

    def compute_all(self, history: List[StepHistory]) -> Dict[str, Any]:
        """计算所有指标并返回合并字典"""
        if not history:
            return ExplorationMetrics().to_dict()

        result: Dict[str, Any] = {}
        result.update(self._basic_stats(history))
        result.update(self._coverage_metrics(history))
        result.update(self._entropy_metrics(history))
        result.update(self._novelty_metrics(history))
        result.update(self._combat_metrics(history))
        return result

    # ─────────────────── 1. Basic Stats ───────────────────

    def _basic_stats(self, history: List[StepHistory]) -> Dict[str, Any]:
        total_reward = sum(s.result.get('reward', 0) for s in history)
        return {
            'total_steps': len(history),
            'episode_length': len(history),
            'total_reward': round(total_reward, 4),
        }

    # ─────────────────── 2. Coverage ───────────────────

    def _coverage_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        min_x, min_y, max_x, max_y = self.map_bounds
        gr = self.grid_resolution
        rr = self.room_resolution

        gw = int((max_x - min_x) / gr) + 1
        gh = int((max_y - min_y) / gr) + 1
        total_grid = gw * gh

        rw = int((max_x - min_x) / rr) + 1
        rh = int((max_y - min_y) / rr) + 1
        total_rooms = rw * rh

        visited_grid: Set[Tuple[int, int]] = set()
        visited_rooms: Set[Tuple[int, int]] = set()
        visited_positions: Set[Tuple[float, float]] = set()
        quadrants: Set[Tuple[int, int]] = set()
        new_cells_per_step: List[int] = []

        for s in history:
            new = 0
            if s.player_pos:
                x, y = s.player_pos[0], s.player_pos[1]
                visited_positions.add((round(x, 1), round(y, 1)))

                gx = max(0, min(int((x - min_x) / gr), gw - 1))
                gy = max(0, min(int((y - min_y) / gr), gh - 1))
                if (gx, gy) not in visited_grid:
                    new = 1
                visited_grid.add((gx, gy))

                rx = max(0, min(int((x - min_x) / rr), rw - 1))
                ry = max(0, min(int((y - min_y) / rr), rh - 1))
                visited_rooms.add((rx, ry))

                quadrants.add((1 if x >= 0 else -1, 1 if y >= 0 else -1))
            new_cells_per_step.append(new)

        return {
            'visited_positions': len(visited_positions),
            'map_coverage': round(len(visited_grid) / max(1, total_grid), 6),
            'room_coverage': round(len(visited_rooms) / max(1, total_rooms), 4),
            'quadrant_coverage': round(len(quadrants) / 4.0, 4),
            'exploration_speed': round(sum(new_cells_per_step) / max(1, len(history)), 6),
        }

    # ─────────────────── 3. Entropy ───────────────────

    def _entropy_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        min_x, min_y = self.map_bounds[0], self.map_bounds[1]
        gr = self.grid_resolution

        state_counts: Counter = Counter()
        action_counts: Counter = Counter()
        sa_counts: Counter = Counter()

        for s in history:
            gv = s.info.get('game_variables', {})
            h_bin = int((gv.get('HEALTH', gv.get('health', 100)) or 0) // 25)
            a_bin = int(min((gv.get('AMMO2', gv.get('ammo_pistol', 0)) or 0), 100) // 20)
            armor_bin = int((gv.get('ARMOR', gv.get('armor', 0)) or 0) // 50)

            pos_bin = (0, 0)
            if s.player_pos:
                pos_bin = (int((s.player_pos[0] - min_x) / gr),
                           int((s.player_pos[1] - min_y) / gr))

            state_key = (pos_bin, h_bin, a_bin, armor_bin)
            act_key = s.action.get('func_name', 'unknown')

            state_counts[state_key] += 1
            action_counts[act_key] += 1
            sa_counts[(state_key, act_key)] += 1

        return {
            'unique_states': len(state_counts),
            'state_entropy': round(self._entropy(state_counts), 4),
            'unique_actions_used': len(action_counts),
            'action_entropy': round(self._entropy(action_counts), 4),
            'state_action_entropy': round(self._entropy(sa_counts), 4),
        }

    @staticmethod
    def _entropy(counts: Counter) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log(c / total + 1e-10) for c in counts.values())

    # ─────────────────── 4. Novelty ───────────────────

    def _novelty_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        # --- visual novelty ---
        visual_scores: List[float] = []
        img_buf: List[np.ndarray] = []
        for s in history:
            screen = s.obs.get('screen')
            if screen is not None and isinstance(screen, np.ndarray):
                ds = screen[::4, ::4, :]
                if img_buf:
                    diffs = [
                        np.mean((ds.astype(float) - h.astype(float)) ** 2) / (255.0 ** 2)
                        for h in img_buf[-50:]
                    ]
                    visual_scores.append(min(diffs))
                img_buf.append(ds.copy())

        avg_vis = float(np.mean(visual_scores)) if visual_scores else 0.0
        max_vis = float(max(visual_scores)) if visual_scores else 0.0
        cum_vis = float(sum(visual_scores)) if visual_scores else 0.0

        # novelty half-life: steps until avg novelty drops to half of initial
        half_life = len(history)
        if len(visual_scores) > 10:
            initial = np.mean(visual_scores[:5])
            threshold = initial / 2.0
            for i, v in enumerate(visual_scores):
                if v < threshold:
                    half_life = i
                    break

        # novelty decay rate (linear regression slope)
        decay_rate = 0.0
        if len(visual_scores) > 10:
            x = np.arange(len(visual_scores), dtype=float)
            y = np.array(visual_scores)
            x_mean, y_mean = np.mean(x), np.mean(y)
            decay_rate = float((np.mean(x * y) - x_mean * y_mean) /
                               (np.mean(x ** 2) - x_mean ** 2 + 1e-10))

        # --- position novelty ---
        pos_novelty_scores: List[float] = []
        pos_history: List[Tuple[float, float]] = []
        for s in history:
            if s.player_pos:
                cur = (s.player_pos[0], s.player_pos[1])
                if pos_history:
                    min_dist = min(math.sqrt((cur[0] - p[0]) ** 2 + (cur[1] - p[1]) ** 2)
                                   for p in pos_history[-200:])
                    pos_novelty_scores.append(min_dist)
                pos_history.append(cur)

        avg_pos_nov = float(np.mean(pos_novelty_scores)) if pos_novelty_scores else 0.0

        return {
            'avg_visual_novelty': round(avg_vis, 6),
            'max_visual_novelty': round(max_vis, 6),
            'cumulative_novelty': round(cum_vis, 4),
            'novelty_half_life': half_life,
            'novelty_decay_rate': round(decay_rate, 8),
            'avg_position_novelty': round(avg_pos_nov, 4),
        }

    # ─────────────────── 5. Combat ───────────────────

    def _combat_metrics(self, history: List[StepHistory]) -> Dict[str, Any]:
        last_gv = history[-1].info.get('game_variables', {})
        frags = int(last_gv.get('FRAGCOUNT', last_gv.get('frag_count', 0)))
        deaths = int(last_gv.get('DEATHCOUNT', last_gv.get('death_count', 0)))
        items = int(last_gv.get('ITEMCOUNT', last_gv.get('item_count', 0)))

        # count attack actions
        n_attacks = sum(1 for s in history if 'attack' in s.action.get('func_name', ''))
        accuracy = frags / max(1, n_attacks) if n_attacks else 0.0

        # damage tracking
        first_health = 100.0
        last_health = last_gv.get('HEALTH', last_gv.get('health', 100.0)) or 0
        damage_taken = max(0, first_health - last_health)

        # survival rate: proportion of steps alive
        total_deaths = deaths
        lives = total_deaths + 1
        avg_survival = len(history) / max(1, lives)

        return {
            'frags': frags,
            'deaths': deaths,
            'kd_ratio': round(frags / max(1, deaths), 2),
            'items_collected': items,
            'attack_count': n_attacks,
            'accuracy_estimate': round(accuracy, 4),
            'damage_taken': round(damage_taken, 2),
            'survival_time': len(history),
            'avg_survival_per_life': round(avg_survival, 1),
        }


def compute_metrics(
    history: List[StepHistory],
    map_bounds: Tuple[float, float, float, float] = (-2000, -2000, 2000, 2000),
    grid_resolution: float = 64.0,
) -> Dict[str, Any]:
    """便捷函数：一步计算所有指标"""
    calculator = DeathmatchMetrics(map_bounds=map_bounds, grid_resolution=grid_resolution)
    return calculator.compute_all(history)
