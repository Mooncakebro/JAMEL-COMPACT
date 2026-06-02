"""
OmniGibson 观察者模块

将 OmniGibson 具身环境的原始观测（RGB、深度、本体感知、物体列表）
转换为 LLM 可读的结构化文本，并提供轨迹保存（Parquet）与探索评测指标计算。

核心能力：
- 场景描述：房间、可见物体、相对位置
- 机器人状态：位姿、夹爪、关节摘要
- 标准化轨迹 Parquet 输出
"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
import io
import os
import math
import logging
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np

pd = None
pq = None

from jamel.core.env.omni_gibson.utils import StepHistory, ExplorationMetrics

try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    from PIL import Image
except ImportError:
    Image = None


def _describe_scene(obs: Dict, info: Dict) -> str:
    """
    根据 obs/info 生成场景文本描述：房间、可见物体、距离/方位摘要。
    """
    info = info or {}
    visible = info.get('visible_objects', [])
    room_id = info.get('room_id', 'unknown')
    room_name = info.get('room_name', room_id)

    lines = [f"Current room: {room_name} (id: {room_id})"]
    if not visible:
        lines.append("Visible objects: none in view.")
    else:
        parts = []
        for i, obj in enumerate(visible[:20]):
            oid = obj.get('object_id', obj.get('id', f'object_{i}'))
            name = obj.get('name', obj.get('category', str(oid)))
            dist = obj.get('distance', obj.get('dist'))
            region = obj.get('region', obj.get('position_description', ''))
            if dist is not None:
                parts.append(f"{name} ({oid}) at {region}, distance ~{float(dist):.1f}m")
            else:
                parts.append(f"{name} ({oid}) at {region}")
        lines.append("Visible objects:\n  " + "\n  ".join(parts))
    return "\n".join(lines)


def _describe_robot_state(obs: Dict, info: Dict) -> str:
    """机器人状态摘要：位置、朝向、夹爪。"""
    info = info or {}
    pos = info.get('robot_position')
    ori = info.get('robot_orientation')
    gripper = info.get('gripper_state')
    joint_summary = info.get('joint_positions_summary')

    parts = []
    if pos is not None:
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            parts.append(f"Position: ({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})")
        elif isinstance(pos, np.ndarray) and pos.size >= 3:
            parts.append(f"Position: ({float(pos.flat[0]):.2f}, {float(pos.flat[1]):.2f}, {float(pos.flat[2]):.2f})")
    if ori is not None:
        parts.append("Orientation: (quat or euler as provided)")
    if gripper is not None:
        g = float(gripper)
        parts.append(f"Gripper: {'closed' if g > 0.5 else 'open'}")
    if joint_summary:
        parts.append(f"Arm: {joint_summary}")
    if not parts:
        return "Robot state: (no state in info)"
    return "Robot state: " + " | ".join(parts)


class Observer:
    """OmniGibson 观察者：原始观测 -> LLM 可读文本，轨迹 I/O，指标计算"""

    @staticmethod
    def get_observation(obs: Dict, info: Dict = None) -> str:
        """将 OmniGibson 单步观测与 info 转为 LLM 可读的完整观察文本。"""
        info = info or {}
        scene = _describe_scene(obs, info)
        robot = _describe_robot_state(obs, info)
        last_action = info.get('last_action', '')
        last_reward = info.get('last_reward', 0.0)
        step = info.get('step_count', 0)

        action_line = ""
        if last_action:
            action_line = f"\nLast action: {last_action}, Reward: {last_reward:+.2f}"
        header = f"[Step {step}]\n"
        return f"""{header}{robot}
{scene}{action_line}
"""

    @staticmethod
    def save_trajectory(
        history: List[StepHistory],
        history_dir: str,
        filename: str = None,
        metadata: dict = None,
    ) -> Optional[str]:
        """将 StepHistory 列表保存为 Parquet。"""
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
                filename = f"omnigibson_trajectory_{ts}.parquet"
            filepath = os.path.join(history_dir, filename)

            records = []
            for h in history:
                screenshot_bytes = None
                screen = h.obs.get('rgb')
                if screen is None:
                    screen = h.obs.get('image')
                if screen is None:
                    screen = h.obs.get('screen')
                if screen is not None and isinstance(screen, np.ndarray) and Image is not None:
                    img = Image.fromarray(screen.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    screenshot_bytes = buf.getvalue()

                action = h.action
                if isinstance(action, dict):
                    func_name = action.get('func_name', action.get('action_name', 'unknown'))
                    action_args = str(action.get('args', {}))
                else:
                    func_name = str(action)
                    action_args = ''

                rec = {
                    'step': h.step,
                    'observation_str': h.observation_str,
                    'screenshot': screenshot_bytes,
                    'action_name': func_name,
                    'action_args': action_args,
                    'reward': h.result.get('reward', 0),
                    'done': h.result.get('done', False),
                    'timestamp': str(h.timestamp),
                    'llm_completion': h.llm_completion,
                    'memory_content': str(h.memory_content) if h.memory_content else None,
                }
                if h.robot_pos:
                    rec['robot_pos_x'], rec['robot_pos_y'], rec['robot_pos_z'] = h.robot_pos
                else:
                    rec['robot_pos_x'] = rec['robot_pos_y'] = rec['robot_pos_z'] = None
                rec['room_id'] = h.room_id
                rec['visible_objects'] = str(h.visible_objects)[:2000] if h.visible_objects else None
                rec['gripper_state'] = h.gripper_state
                records.append(rec)

            df = pd.DataFrame(records)
            if metadata:
                df.attrs['metadata'] = metadata
            df.to_parquet(filepath)
            logger.info("Trajectory saved", filepath=filepath)
            return filepath
        except Exception as e:
            logger.error("Failed to save trajectory: %s", e, exc_info=True)
            return None

    @staticmethod
    def load_trajectory(parquet_file: str):
        """从 Parquet 加载轨迹 DataFrame。"""
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
            return pd.read_parquet(parquet_file)
        except Exception as e:
            logger.error("Failed to load trajectory: %s", e, exc_info=True)
            return None

    @staticmethod
    def calculate_metrics(
        history: List[StepHistory],
        map_bounds: Tuple[float, float, float, float] = (-10.0, -10.0, 10.0, 10.0),
        grid_resolution: float = 0.5,
        room_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        计算探索评测指标：覆盖率、状态熵、新颖性、交互多样性。
        room_ids: 若提供，则 room_coverage = 访问过的房间数 / len(room_ids)
        """
        if not history:
            return ExplorationMetrics().to_dict()

        m = ExplorationMetrics()
        m.episode_length = len(history)
        m.total_steps = len(history)
        m.total_reward = sum(s.result.get('reward', 0) for s in history)

        min_x, min_y, max_x, max_y = map_bounds
        gw = max(1, int((max_x - min_x) / grid_resolution) + 1)
        gh = max(1, int((max_y - min_y) / grid_resolution) + 1)
        total_cells = gw * gh

        visited_cells: Set[Tuple[int, int]] = set()
        visited_positions: Set[Tuple[float, float]] = set()
        rooms_visited: Set[str] = set()
        new_cells_per_step: List[int] = []

        for s in history:
            new = 0
            if s.robot_pos:
                x, y = s.robot_pos[0], s.robot_pos[1]
                visited_positions.add((round(x, 2), round(y, 2)))
                gx = max(0, min(int((x - min_x) / grid_resolution), gw - 1))
                gy = max(0, min(int((y - min_y) / grid_resolution), gh - 1))
                if (gx, gy) not in visited_cells:
                    new = 1
                visited_cells.add((gx, gy))
            if s.room_id:
                rooms_visited.add(s.room_id)
            new_cells_per_step.append(new)

        m.visited_positions = len(visited_positions)
        m.map_coverage = len(visited_cells) / total_cells if total_cells else 0.0
        total_rooms = len(room_ids) if room_ids else max(len(rooms_visited), 1)
        m.unique_rooms_visited = len(rooms_visited)
        m.room_coverage = len(rooms_visited) / total_rooms if total_rooms else 0.0
        m.exploration_speed = sum(new_cells_per_step) / max(1, len(history))

        # 状态熵（位置网格 + 房间）
        state_counts: Counter = Counter()
        action_counts: Counter = Counter()
        for s in history:
            pos_bin = (0, 0)
            if s.robot_pos:
                pos_bin = (
                    int((s.robot_pos[0] - min_x) / grid_resolution),
                    int((s.robot_pos[1] - min_y) / grid_resolution),
                )
            room = s.room_id or 'unknown'
            state_key = (pos_bin, room)
            state_counts[state_key] += 1
            act = s.action.get('func_name', s.action.get('action_name', 'unknown')) if isinstance(s.action, dict) else str(s.action)
            action_counts[act] += 1

        m.unique_states = len(state_counts)
        if state_counts:
            total = sum(state_counts.values())
            probs = [c / total for c in state_counts.values()]
            m.state_entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        m.unique_actions_used = len(action_counts)
        if action_counts:
            total = sum(action_counts.values())
            probs = [c / total for c in action_counts.values()]
            m.action_entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        # 视觉新颖性（帧间差异）
        novelty_scores: List[float] = []
        img_hist: List[np.ndarray] = []
        for s in history:
            screen = s.obs.get('rgb')
            if screen is None:
                screen = s.obs.get('image')
            if screen is None:
                screen = s.obs.get('screen')
            if screen is not None and isinstance(screen, np.ndarray):
                ds = screen[::4, ::4, :].astype(np.float32) if screen.ndim >= 3 else np.zeros((8, 8, 3))
                if screen.ndim == 2:
                    ds = np.stack([screen[::4, ::4]] * 3, axis=-1)
                if img_hist:
                    diffs = [
                        np.mean((ds - h.astype(np.float32)) ** 2) / (255.0 ** 2 + 1e-10)
                        for h in img_hist[-50:]
                    ]
                    novelty_scores.append(float(min(diffs)))
                img_hist.append(ds.copy())
        if novelty_scores:
            m.avg_visual_novelty = float(np.mean(novelty_scores))
            m.max_visual_novelty = float(max(novelty_scores))
            m.cumulative_novelty = float(sum(novelty_scores))
            if len(novelty_scores) > 10:
                x = np.arange(len(novelty_scores), dtype=float)
                y = np.array(novelty_scores)
                slope = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x ** 2) - np.mean(x) ** 2 + 1e-10)
                m.novelty_decay_rate = float(slope)

        # 位置新颖性
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
        if pos_novelty_scores:
            m.avg_position_novelty = float(np.mean(pos_novelty_scores))

        # 交互多样性
        interacted: Set[str] = set()
        for s in history:
            oid = s.interacted_object_id or (s.action.get('args') or {}).get('object_id') if isinstance(s.action, dict) else None
            if oid:
                interacted.add(str(oid))
            if isinstance(s.action, dict) and s.action.get('action_name') in ('interact', 'grasp', 'release'):
                m.interaction_count += 1
        m.unique_objects_interacted = len(interacted)

        return m.to_dict()
