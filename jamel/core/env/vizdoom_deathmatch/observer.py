"""
VizDoom Deathmatch 观察者模块

将 VizDoom Deathmatch 场景的第一人称 3D 画面转化为 LLM 可读的结构化文本，
并提供轨迹保存（Parquet）和完整的探索评测指标计算。

Deathmatch 特有能力：
- 多敌人追踪与威胁等级评估
- 弹药/武器/护甲状态的战术描述
- 战斗事件检测（击杀、被击杀、拾取）
- 空间覆盖率、状态熵、视觉新颖性等探索指标
"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
import io
import os
import math
from PIL import Image
import numpy as np
import logging

pd = None
pq = None

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

from jamel.core.env.vizdoom_deathmatch.utils import StepHistory, ExplorationMetrics

try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# ========== VizDoom 物体分类 ==========

ENEMY_KEYWORDS = {
    'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'DoomImp', 'Demon',
    'Spectre', 'LostSoul', 'Cacodemon', 'HellKnight', 'BaronOfHell',
    'Arachnotron', 'PainElemental', 'Revenant', 'Mancubus', 'Archvile',
    'Cyberdemon', 'SpiderMastermind', 'WolfensteinSS',
    'Monster', 'Enemy', 'zombie', 'imp', 'demon',
}

WEAPON_KEYWORDS = {
    'Shotgun', 'SuperShotgun', 'Chaingun', 'RocketLauncher',
    'PlasmaRifle', 'BFG9000', 'Chainsaw',
}

HEALTH_KEYWORDS = {
    'Medikit', 'Stimpack', 'HealthBonus', 'Soulsphere', 'MegaSphere',
    'Berserk',
}

ARMOR_KEYWORDS = {
    'GreenArmor', 'BlueArmor', 'ArmorBonus',
}

AMMO_KEYWORDS = {
    'Clip', 'Shell', 'RocketAmmo', 'Cell', 'Backpack',
}

KEY_KEYWORDS = {
    'BlueCard', 'RedCard', 'YellowCard',
    'BlueSkull', 'RedSkull', 'YellowSkull',
}

POWERUP_KEYWORDS = {
    'InvulnerabilitySphere', 'BlurSphere', 'RadSuit',
    'Allmap', 'Infrared',
}

DECORATION_KEYWORDS = {
    'Column', 'TechPillar', 'Barrel', 'ExplosiveBarrel',
    'TallGreenColumn', 'ShortGreenColumn', 'TallRedColumn', 'ShortRedColumn',
    'Candelabra', 'Candlestick', 'HeadOnAStick', 'Gibs',
}


def _classify_object(name: str) -> str:
    nl = name.lower()
    for kw in ENEMY_KEYWORDS:
        if kw.lower() in nl:
            return "enemy"
    for kw in WEAPON_KEYWORDS:
        if kw.lower() in nl:
            return "weapon"
    for kw in HEALTH_KEYWORDS:
        if kw.lower() in nl:
            return "health"
    for kw in ARMOR_KEYWORDS:
        if kw.lower() in nl:
            return "armor"
    for kw in AMMO_KEYWORDS:
        if kw.lower() in nl:
            return "ammo"
    for kw in KEY_KEYWORDS:
        if kw.lower() in nl:
            return "key"
    for kw in POWERUP_KEYWORDS:
        if kw.lower() in nl:
            return "powerup"
    for kw in DECORATION_KEYWORDS:
        if kw.lower() in nl:
            return "decoration"
    if 'player' in nl:
        return "self"
    return "unknown"


def _screen_region(cx: float, sw: int) -> str:
    r = cx / sw
    if r < 0.2:
        return "far left"
    elif r < 0.4:
        return "left"
    elif r < 0.6:
        return "center"
    elif r < 0.8:
        return "right"
    return "far right"


def _estimate_distance(w: int, h: int, sw: int, sh: int) -> str:
    ratio = (w * h) / (sw * sh)
    if ratio > 0.15:
        return "very close"
    elif ratio > 0.05:
        return "close"
    elif ratio > 0.01:
        return "medium"
    elif ratio > 0.002:
        return "far"
    return "very far"


def _threat_level(proximity: str, obj_name: str) -> int:
    """Numeric threat score for sorting — higher = more dangerous."""
    base = 3
    high_threat = {'Cyberdemon', 'Archvile', 'Mancubus', 'Revenant', 'BaronOfHell'}
    if any(k.lower() in obj_name.lower() for k in high_threat):
        base = 6
    prox_map = {"very close": 5, "close": 4, "medium": 3, "far": 2, "very far": 1}
    return base + prox_map.get(proximity, 0)


@dataclass
class _DeathmatchObservation:
    screen: np.ndarray
    depth: Optional[np.ndarray]
    labels_buffer: Optional[np.ndarray]
    automap: Optional[np.ndarray]

    health: float = 100.0
    armor: float = 0.0
    ammo_pistol: float = 0.0
    ammo_shotgun: float = 0.0
    ammo_rocket: float = 0.0
    ammo_cell: float = 0.0
    selected_weapon: int = 0
    frag_count: int = 0
    death_count: int = 0

    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    angle: float = 0.0

    last_action: str = ""
    last_reward: float = 0.0

    @classmethod
    
    def from_dict(cls, obs: Dict, info: Dict = None) -> '_DeathmatchObservation':
        info = info or {}
        gv = info.get('game_variables', {})
        return cls(
            screen=obs.get('screen', np.zeros((240, 320, 3), dtype=np.uint8)),
            depth=obs.get('depth'),
            labels_buffer=obs.get('labels'),
            automap=obs.get('automap'),
            health=gv.get('HEALTH', gv.get('health', 100.0)),
            armor=gv.get('ARMOR', gv.get('armor', 0.0)),
            ammo_pistol=gv.get('AMMO2', gv.get('ammo_pistol', 0.0)),
            ammo_shotgun=gv.get('AMMO3', gv.get('ammo_shotgun', 0.0)),
            ammo_rocket=gv.get('AMMO4', gv.get('ammo_rocket', 0.0)),
            ammo_cell=gv.get('AMMO5', gv.get('ammo_cell', 0.0)),
            selected_weapon=int(gv.get('SELECTED_WEAPON', gv.get('selected_weapon', 0))),
            frag_count=int(gv.get('FRAGCOUNT', gv.get('frag_count', 0))),
            death_count=int(gv.get('DEATHCOUNT', gv.get('death_count', 0))),
            position_x=gv.get('POSITION_X', gv.get('position_x', 0.0)),
            position_y=gv.get('POSITION_Y', gv.get('position_y', 0.0)),
            position_z=gv.get('POSITION_Z', gv.get('position_z', 0.0)),
            angle=gv.get('ANGLE', gv.get('angle', 0.0)),
            last_action=info.get('last_action', ''),
            last_reward=info.get('last_reward', 0.0),
        )


class Observer:
    """VizDoom Deathmatch 观察者"""

    @staticmethod
    def describe_scene(obs: Dict, info: Dict = None) -> str:
        """
        将 VizDoom labels 转换为战术场景描述。

        Deathmatch 场景描述按重要性排序：
        1. 威胁（敌人） — 按威胁等级排序
        2. 战利品（武器、弹药、血包、护甲）
        3. 地形深度信息
        """
        info = info or {}
        labels = info.get('labels', [])
        screen = obs.get('screen')

        if not labels:
            return "No labeled objects detected — area appears clear."

        sh = screen.shape[0] if screen is not None else 240
        sw = screen.shape[1] if screen is not None else 320

        enemies, weapons, health_items, armor_items = [], [], [], []
        ammo_items, powerups, keys, other = [], [], [], []

        for lab in labels:
            obj_name = lab.get('object_name', 'Unknown')
            cat = _classify_object(obj_name)
            if cat == "self":
                continue

            cx = lab.get('x', 0) + lab.get('width', 0) / 2
            region = _screen_region(cx, sw)
            prox = _estimate_distance(lab.get('width', 0), lab.get('height', 0), sw, sh)

            entry = {'name': obj_name, 'region': region, 'proximity': prox, 'cx': cx}

            if cat == "enemy":
                entry['threat'] = _threat_level(prox, obj_name)
                enemies.append(entry)
            elif cat == "weapon":
                weapons.append(entry)
            elif cat == "health":
                health_items.append(entry)
            elif cat == "armor":
                armor_items.append(entry)
            elif cat == "ammo":
                ammo_items.append(entry)
            elif cat == "powerup":
                powerups.append(entry)
            elif cat == "key":
                keys.append(entry)
            elif cat != "decoration":
                other.append(entry)

        lines: List[str] = []

        if enemies:
            enemies.sort(key=lambda e: -e['threat'])
            descs = [f"{e['name']} at {e['region']} ({e['proximity']})" for e in enemies]
            lines.append(f"!! ENEMIES [{len(enemies)}]: {'; '.join(descs)}")
        else:
            lines.append("Enemies: none visible.")

        pickups: List[str] = []
        for label_text, group in [("Weapons", weapons), ("Health", health_items),
                                   ("Armor", armor_items), ("Ammo", ammo_items),
                                   ("Powerups", powerups), ("Keys", keys)]:
            if group:
                descs = [f"{it['name']} at {it['region']} ({it['proximity']})" for it in group]
                pickups.append(f"{label_text}: {'; '.join(descs)}")
        if pickups:
            lines.append("Pickups visible:\n  " + "\n  ".join(pickups))

        depth = obs.get('depth')
        if depth is not None and isinstance(depth, np.ndarray):
            h, w = depth.shape[:2]
            center = depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
            if center.size > 0:
                avg_d = float(np.mean(center))
                min_d = float(np.min(center))
                if min_d < 50:
                    lines.append(f"Depth: wall/obstacle VERY CLOSE (min {min_d:.0f})")
                elif avg_d < 200:
                    lines.append(f"Depth: tight corridor (avg {avg_d:.0f})")
                elif avg_d < 500:
                    lines.append(f"Depth: medium room (avg {avg_d:.0f})")
                else:
                    lines.append(f"Depth: open arena (avg {avg_d:.0f})")

        if other:
            lines.append(f"Other objects: {'; '.join(o['name'] for o in other[:3])}")

        return "\n".join(lines)

    @staticmethod
    def get_observation(obs: Dict, info: Dict = None) -> str:
        """将 Deathmatch 状态转换为 LLM 可读的完整观察文本"""
        d = _DeathmatchObservation.from_dict(obs, info)

        # --- status ---
        status_parts = [f"Health: {d.health:.0f}/100"]
        if d.armor > 0:
            status_parts.append(f"Armor: {d.armor:.0f}")
        status = ", ".join(status_parts)

        # --- ammo summary ---
        ammo_parts = []
        if d.ammo_pistol > 0:
            ammo_parts.append(f"Bullets: {d.ammo_pistol:.0f}")
        if d.ammo_shotgun > 0:
            ammo_parts.append(f"Shells: {d.ammo_shotgun:.0f}")
        if d.ammo_rocket > 0:
            ammo_parts.append(f"Rockets: {d.ammo_rocket:.0f}")
        if d.ammo_cell > 0:
            ammo_parts.append(f"Cells: {d.ammo_cell:.0f}")
        ammo_str = ", ".join(ammo_parts) if ammo_parts else "No ammo"

        # --- position ---
        pos_str = (f"Position: ({d.position_x:.1f}, {d.position_y:.1f}, "
                   f"{d.position_z:.1f}), Angle: {d.angle:.1f}°")

        # --- combat stats ---
        combat_str = f"Frags: {d.frag_count}, Deaths: {d.death_count}"

        # --- last action ---
        action_result = ""
        if d.last_action:
            action_result = f"\nLast Action: {d.last_action}, Reward: {d.last_reward:+.2f}"

        scene = Observer.describe_scene(obs, info)

        return f"""
Status: {status}
Ammo: {ammo_str}
{pos_str}
Combat: {combat_str}{action_result}

Visual Scene:
{scene}
"""

    # ======================== Trajectory I/O ========================

    @staticmethod
    def save_trajectory(
        history: List[StepHistory],
        history_dir: str,
        filename: str = None,
        metadata: dict = None,
    ) -> Optional[str]:
        global pd, pq
        try:
            if pd is None:
                import pandas as _pd
                pd = _pd
            if pq is None:
                import pyarrow.parquet as _pq
                pq = _pq
        except ImportError:
            logger.error("pandas/pyarrow not found – cannot save parquet.")
            return None

        try:
            os.makedirs(history_dir, exist_ok=True)
            if filename is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"deathmatch_trajectory_{ts}.parquet"
            filepath = os.path.join(history_dir, filename)

            records = []
            for h in history:
                screenshot_bytes = None
                screen = h.obs.get('screen')
                if screen is not None and isinstance(screen, np.ndarray):
                    img = Image.fromarray(screen.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    screenshot_bytes = buf.getvalue()

                records.append({
                    'step': h.step,
                    'observation_str': h.observation_str,
                    'screenshot': screenshot_bytes,
                    'action_name': h.action.get('func_name', 'unknown'),
                    'action_args': str(h.action.get('args', {})),
                    'reward': h.result.get('reward', 0),
                    'done': h.result.get('done', False),
                    'player_pos_x': h.player_pos[0] if h.player_pos else None,
                    'player_pos_y': h.player_pos[1] if h.player_pos else None,
                    'player_pos_z': h.player_pos[2] if h.player_pos else None,
                    'player_angle': h.player_angle,
                    'health': h.health,
                    'armor': h.armor,
                    'frag_count': h.frag_count,
                    'death_count': h.death_count,
                    'timestamp': str(h.timestamp),
                    'llm_completion': h.llm_completion,
                    'memory_content': str(h.memory_content) if h.memory_content else None,
                })

            df = pd.DataFrame(records)
            if metadata:
                df.attrs['metadata'] = metadata
            df.to_parquet(filepath)
            logger.info(f"Trajectory saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}", exc_info=True)
            return None

    @staticmethod
    def load_trajectory(parquet_file: str):
        global pd, pq
        try:
            if pd is None:
                import pandas as _pd
                pd = _pd
            if pq is None:
                import pyarrow.parquet as _pq
                pq = _pq
        except ImportError:
            logger.error("pandas/pyarrow not found – cannot load parquet.")
            return None
        try:
            return pq.read_table(parquet_file).to_pandas()
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}", exc_info=True)
            return None

    # =================== Exploration Metrics ===================

    @staticmethod
    def calculate_metrics(
        history: List[StepHistory],
        map_bounds: Tuple[float, float, float, float] = (-2000, -2000, 2000, 2000),
        grid_resolution: float = 64.0,
    ) -> Dict[str, Any]:
        """
        计算完整的 Deathmatch 探索评测指标

        指标体系：
        1. 空间覆盖率 (Map Coverage) — 网格化坐标覆盖
        2. 象限覆盖 (Quadrant Coverage) — 4 象限分别是否到达
        3. 探索速度 (Exploration Speed) — 新格子/步
        4. 状态熵 (State Entropy) — 离散状态的信息熵
        5. 动作熵 (Action Entropy) — 动作分布均匀性
        6. 视觉新颖性 (Visual Novelty) — 帧间 MSE
        7. 新颖性衰减率 — 随时间新颖性是否递减（正值=探索衰减）
        8. 战斗指标 — frags, deaths, K/D, damage
        """
        if not history:
            return ExplorationMetrics().to_dict()

        m = ExplorationMetrics()
        m.episode_length = len(history)
        m.total_steps = len(history)
        m.total_reward = sum(s.result.get('reward', 0) for s in history)

        min_x, min_y, max_x, max_y = map_bounds
        gw = int((max_x - min_x) / grid_resolution) + 1
        gh = int((max_y - min_y) / grid_resolution) + 1
        total_cells = gw * gh

        visited_cells: Set[Tuple[int, int]] = set()
        visited_positions: Set[Tuple[float, float]] = set()
        quadrants_visited: Set[Tuple[int, int]] = set()  # sign-based quadrant
        new_cells_per_step: List[int] = []

        for s in history:
            new = 0
            if s.player_pos:
                x, y = s.player_pos[0], s.player_pos[1]
                visited_positions.add((round(x, 1), round(y, 1)))
                gx = max(0, min(int((x - min_x) / grid_resolution), gw - 1))
                gy = max(0, min(int((y - min_y) / grid_resolution), gh - 1))
                if (gx, gy) not in visited_cells:
                    new = 1
                visited_cells.add((gx, gy))
                qx = 1 if x >= 0 else -1
                qy = 1 if y >= 0 else -1
                quadrants_visited.add((qx, qy))
            new_cells_per_step.append(new)

        m.visited_positions = len(visited_positions)
        m.map_coverage = len(visited_cells) / total_cells if total_cells else 0.0
        m.quadrant_coverage = len(quadrants_visited) / 4.0
        m.exploration_speed = sum(new_cells_per_step) / max(1, len(history))

        # --- state entropy ---
        state_counts: Counter = Counter()
        for s in history:
            gv = s.info.get('game_variables', {})
            h_bin = int((gv.get('HEALTH', gv.get('health', 100)) or 0) // 20)
            a_bin = int(min((gv.get('AMMO2', gv.get('ammo_pistol', 0)) or 0), 100) // 10)
            pos_bin = (0, 0)
            if s.player_pos:
                pos_bin = (int((s.player_pos[0] - min_x) / grid_resolution),
                           int((s.player_pos[1] - min_y) / grid_resolution))
            state_counts[(pos_bin, h_bin, a_bin)] += 1

        m.unique_states = len(state_counts)
        if state_counts:
            total = sum(state_counts.values())
            probs = [c / total for c in state_counts.values()]
            m.state_entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        # --- action entropy ---
        act_counts: Counter = Counter()
        for s in history:
            act_counts[s.action.get('func_name', 'unknown')] += 1
        m.unique_actions_used = len(act_counts)
        if act_counts:
            total = sum(act_counts.values())
            probs = [c / total for c in act_counts.values()]
            m.action_entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        # --- visual novelty ---
        novelty_scores: List[float] = []
        img_hist: List[np.ndarray] = []
        for s in history:
            screen = s.obs.get('screen')
            if screen is not None and isinstance(screen, np.ndarray):
                ds = screen[::4, ::4, :]
                if img_hist:
                    diffs = [np.mean((ds.astype(float) - h.astype(float)) ** 2) / (255.0 ** 2)
                             for h in img_hist[-50:]]
                    novelty_scores.append(min(diffs))
                img_hist.append(ds.copy())

        if novelty_scores:
            m.avg_visual_novelty = float(np.mean(novelty_scores))
            m.max_visual_novelty = float(max(novelty_scores))
            m.cumulative_novelty = float(sum(novelty_scores))
            # decay rate: linear regression slope of novelty over time
            if len(novelty_scores) > 10:
                x = np.arange(len(novelty_scores), dtype=float)
                y = np.array(novelty_scores)
                slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x ** 2) - np.mean(x) ** 2 + 1e-10)
                m.novelty_decay_rate = float(slope)

        # --- combat ---
        last_gv = history[-1].info.get('game_variables', {})
        m.frags = int(last_gv.get('FRAGCOUNT', last_gv.get('frag_count', 0)))
        m.deaths = int(last_gv.get('DEATHCOUNT', last_gv.get('death_count', 0)))
        m.kd_ratio = m.frags / max(1, m.deaths)
        m.items_collected = int(last_gv.get('ITEMCOUNT', last_gv.get('item_count', 0)))

        first_health = 100.0
        last_health = last_gv.get('HEALTH', last_gv.get('health', 100.0)) or 0
        if last_health < first_health:
            m.damage_taken = first_health - last_health

        m.survival_time = m.episode_length

        return m.to_dict()

    @staticmethod
    def calculate_novelty_rnd(
        history: List[StepHistory],
        feature_dim: int = 512,
        hidden_dim: int = 256,
    ) -> Dict[str, float]:
        """RND (Random Network Distillation) 新颖性计算"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return {"rnd_novelty": None, "error": "PyTorch not installed"}

        class _Fixed(nn.Module):
            def __init__(self, d_in, d_out):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_out))
                for p in self.parameters():
                    p.requires_grad = False
            def forward(self, x):
                return self.net(x)

        class _Pred(nn.Module):
            def __init__(self, d_in, d_out):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_in, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, d_out))
            def forward(self, x):
                return self.net(x)

        feats = []
        for s in history:
            screen = s.obs.get('screen')
            if screen is not None:
                small = screen[::16, ::16, :].flatten()
                if len(small) > 1024:
                    small = small[:1024]
                else:
                    small = np.pad(small, (0, 1024 - len(small)))
                feats.append(small.astype(np.float32) / 255.0)
            else:
                feats.append(np.zeros(1024, dtype=np.float32))

        if not feats:
            return {"rnd_avg_novelty": 0.0, "rnd_max_novelty": 0.0}

        feats_t = torch.tensor(np.array(feats), dtype=torch.float32)
        d_in = feats_t.shape[1]
        fixed = _Fixed(d_in, feature_dim)
        pred = _Pred(d_in, feature_dim)

        with torch.no_grad():
            tgt = fixed(feats_t)
            out = pred(feats_t)
            nov = torch.mean((tgt - out) ** 2, dim=1).numpy()

        return {
            "rnd_avg_novelty": float(np.mean(nov)),
            "rnd_max_novelty": float(np.max(nov)),
            "rnd_cumulative_novelty": float(np.sum(nov)),
        }
