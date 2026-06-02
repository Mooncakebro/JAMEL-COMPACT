"""
Vizdoom 观察者模块 - 负责将 Vizdoom 环境状态转化为 LLM 能理解的文本表示
并实现探索评测指标计算

核心能力：
- 解析 VizDoom labels (物体标签) 和 depth buffer (深度图)
- 将第一人称 3D 画面转化为结构化场景描述
"""

from __future__ import annotations
from datetime import datetime
import io
import os
from PIL import Image
import numpy as np
import logging
from collections import Counter

# Lazy import for pandas/pyarrow to avoid issues if libraries are broken
pd = None
pq = None

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

from jamel.core.env.vizdoom.utils import StepHistory, ExplorationMetrics

# Fallback logger if structlog/log_utils is not available
try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# ========== VizDoom 物体分类 ==========
# 根据常见的 VizDoom 物体名称进行分类
ENEMY_KEYWORDS = {
    'Zombieman', 'ShotgunGuy', 'ChaingunGuy', 'DoomImp', 'Demon',
    'Spectre', 'LostSoul', 'Cacodemon', 'HellKnight', 'BaronOfHell',
    'Arachnotron', 'PainElemental', 'Revenant', 'Mancubus', 'Archvile',
    'Cyberdemon', 'SpiderMastermind', 'WolfensteinSS',
    # 常见别名
    'Monster', 'Enemy', 'zombie', 'imp', 'demon',
}

ITEM_KEYWORDS = {
    'Medikit', 'Stimpack', 'HealthBonus', 'Soulsphere', 'MegaSphere',
    'GreenArmor', 'BlueArmor', 'ArmorBonus',
    'Clip', 'Shell', 'RocketAmmo', 'Cell', 'Backpack',
    'Shotgun', 'SuperShotgun', 'Chaingun', 'RocketLauncher',
    'PlasmaRifle', 'BFG9000', 'Chainsaw', 'Berserk',
    'InvulnerabilitySphere', 'BlurSphere', 'RadSuit', 'Allmap',
    'Infrared', 'BlueCard', 'RedCard', 'YellowCard',
    'BlueSkull', 'RedSkull', 'YellowSkull',
}

DECORATION_KEYWORDS = {
    'Column', 'TechPillar', 'Barrel', 'ExplosiveBarrel',
    'TallGreenColumn', 'ShortGreenColumn', 'TallRedColumn', 'ShortRedColumn',
    'Candelabra', 'Candlestick', 'HeadOnAStick', 'Gibs',
}


def _classify_object(object_name: str) -> str:
    """将 VizDoom 物体名称分类为 enemy / item / decoration / unknown"""
    for keyword in ENEMY_KEYWORDS:
        if keyword.lower() in object_name.lower():
            return "enemy"
    for keyword in ITEM_KEYWORDS:
        if keyword.lower() in object_name.lower():
            return "item"
    for keyword in DECORATION_KEYWORDS:
        if keyword.lower() in object_name.lower():
            return "decoration"
    if 'Player' in object_name or 'player' in object_name:
        return "self"
    return "unknown"


def _screen_region(center_x: float, screen_width: int) -> str:
    """根据物体在屏幕上的水平位置判断方位"""
    ratio = center_x / screen_width
    if ratio < 0.25:
        return "far left"
    elif ratio < 0.45:
        return "left"
    elif ratio < 0.55:
        return "center"
    elif ratio < 0.75:
        return "right"
    else:
        return "far right"


def _estimate_size(width: int, height: int, screen_width: int, screen_height: int) -> str:
    """根据物体在屏幕上的占比估算大小/距离"""
    area_ratio = (width * height) / (screen_width * screen_height)
    if area_ratio > 0.15:
        return "very close"
    elif area_ratio > 0.05:
        return "close"
    elif area_ratio > 0.01:
        return "medium distance"
    elif area_ratio > 0.002:
        return "far"
    else:
        return "very far"


@dataclass
class _VizdoomObservation:
    """
    Vizdoom 观察数据结构
    """
    # 核心观察
    screen: np.ndarray                          # RGB 屏幕图像
    depth: Optional[np.ndarray]                 # 深度图 (如果启用)
    labels_buffer: Optional[np.ndarray]         # 语义标签图 (如果启用)
    automap: Optional[np.ndarray]               # 自动地图 (如果启用)

    # 游戏变量
    health: float = 100.0
    ammo: float = 0.0
    armor: float = 0.0
    kill_count: int = 0
    item_count: int = 0
    secret_count: int = 0

    # 位置信息
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    angle: float = 0.0

    # 上一次动作
    last_action: str = ""
    last_reward: float = 0.0

    @classmethod
    def from_dict(cls, obs: Dict, info: Dict = None) -> _VizdoomObservation:
        """
        从字典创建观察对象

        Args:
            obs: 观察字典，包含 screen, depth 等
            info: 额外信息字典，包含 game_variables 等
        """
        info = info or {}
        game_vars = info.get('game_variables', {})

        return cls(
            screen=obs.get('screen', np.zeros((240, 320, 3), dtype=np.uint8)),
            depth=obs.get('depth'),
            labels_buffer=obs.get('labels'),
            automap=obs.get('automap'),
            health=game_vars.get('HEALTH', game_vars.get('health', 100.0)),
            ammo=game_vars.get('AMMO', game_vars.get('AMMO2', game_vars.get('ammo', 0.0))),
            armor=game_vars.get('ARMOR', game_vars.get('armor', 0.0)),
            kill_count=int(game_vars.get('KILLCOUNT', game_vars.get('kill_count', 0))),
            item_count=int(game_vars.get('ITEMCOUNT', game_vars.get('item_count', 0))),
            secret_count=int(game_vars.get('SECRETCOUNT', game_vars.get('secret_count', 0))),
            position_x=game_vars.get('POSITION_X', game_vars.get('position_x', 0.0)),
            position_y=game_vars.get('POSITION_Y', game_vars.get('position_y', 0.0)),
            position_z=game_vars.get('POSITION_Z', game_vars.get('position_z', 0.0)),
            angle=game_vars.get('ANGLE', game_vars.get('angle', 0.0)),
            last_action=info.get('last_action', ''),
            last_reward=info.get('last_reward', 0.0),
        )


class Observer:
    """Vizdoom 观察者，将状态转换为结构化文本"""

    @staticmethod
    def describe_scene(obs: Dict, info: Dict = None) -> str:
        """
        解析 VizDoom 的 labels 数据，生成第一人称视角的场景描述。

        VizDoom 提供的 labels 包含：
        - object_id: 物体唯一 ID
        - object_name: 物体名称（如 Zombieman, Medikit 等）
        - x, y, width, height: 物体在屏幕上的包围盒

        Args:
            obs: 包含 screen, depth, labels 等的观察字典
            info: 包含 game_variables, labels 等的信息字典

        Returns:
            场景的自然语言描述
        """
        info = info or {}
        labels = info.get('labels', [])
        screen = obs.get('screen')

        if not labels:
            return "No labeled objects detected in the current view."

        screen_height = screen.shape[0] if screen is not None else 240
        screen_width = screen.shape[1] if screen is not None else 320

        # 分类可见物体
        enemies: List[Dict] = []
        items: List[Dict] = []
        decorations: List[Dict] = []
        other: List[Dict] = []

        for label in labels:
            obj_name = label.get('object_name', 'Unknown')
            category = _classify_object(obj_name)

            if category == "self":
                continue  # 跳过玩家自身

            center_x = label.get('x', 0) + label.get('width', 0) / 2
            region = _screen_region(center_x, screen_width)
            proximity = _estimate_size(
                label.get('width', 0), label.get('height', 0),
                screen_width, screen_height
            )

            obj_info = {
                'name': obj_name,
                'region': region,
                'proximity': proximity,
                'center_x': center_x,
            }

            if category == "enemy":
                enemies.append(obj_info)
            elif category == "item":
                items.append(obj_info)
            elif category == "decoration":
                decorations.append(obj_info)
            else:
                other.append(obj_info)

        # ---- 组装描述 ----
        lines: List[str] = []

        # 1. 敌人（最重要）
        if enemies:
            enemy_descs = []
            for e in sorted(enemies, key=lambda x: x['center_x']):
                enemy_descs.append(f"{e['name']} at {e['region']} ({e['proximity']})")
            lines.append(f"!! ENEMIES IN VIEW: {'; '.join(enemy_descs)}")
        else:
            lines.append("Enemies: none visible.")

        # 2. 物品
        if items:
            item_descs = []
            for it in sorted(items, key=lambda x: x['center_x']):
                item_descs.append(f"{it['name']} at {it['region']} ({it['proximity']})")
            lines.append(f"Items visible: {'; '.join(item_descs)}")

        # 3. 环境物体（简要）
        if decorations:
            deco_names = [d['name'] for d in decorations[:5]]
            unique_names = list(dict.fromkeys(deco_names))  # 去重保序
            lines.append(f"Environment objects: {', '.join(unique_names)}")

        # 4. 其他物体
        if other:
            other_descs = [f"{o['name']} at {o['region']}" for o in other[:3]]
            lines.append(f"Other: {'; '.join(other_descs)}")

        # 5. 深度信息概要
        depth = obs.get('depth')
        if depth is not None and isinstance(depth, np.ndarray):
            h, w = depth.shape[:2]
            center_depth = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
            if center_depth.size > 0:
                avg_depth = float(np.mean(center_depth))
                min_depth = float(np.min(center_depth))
                if min_depth < 50:
                    lines.append(f"Depth: wall or obstacle very close ahead (min depth: {min_depth:.0f})")
                elif avg_depth < 200:
                    lines.append(f"Depth: enclosed space ahead (avg depth: {avg_depth:.0f})")
                else:
                    lines.append(f"Depth: open area ahead (avg depth: {avg_depth:.0f})")

        return "\n".join(lines)

    @staticmethod
    def get_observation(obs: Dict, info: Dict = None) -> str:
        """
        将 Vizdoom 观察转换为 LLM 可读的文本

        Args:
            obs: 原始观察数据
            info: 额外信息

        Returns:
            结构化的文本描述
        """
        vizdoom_obs = _VizdoomObservation.from_dict(obs, info)

        # 构建状态描述
        status_lines = []
        status_lines.append(f"Health: {vizdoom_obs.health:.0f}/100")
        if vizdoom_obs.ammo > 0:
            status_lines.append(f"Ammo: {vizdoom_obs.ammo:.0f}")
        if vizdoom_obs.armor > 0:
            status_lines.append(f"Armor: {vizdoom_obs.armor:.0f}")

        status_str = ", ".join(status_lines)

        # 位置信息
        position_str = f"Position: ({vizdoom_obs.position_x:.1f}, {vizdoom_obs.position_y:.1f}), Angle: {vizdoom_obs.angle:.1f}°"

        # 统计信息
        stats_lines = []
        if vizdoom_obs.kill_count > 0:
            stats_lines.append(f"Kills: {vizdoom_obs.kill_count}")
        if vizdoom_obs.item_count > 0:
            stats_lines.append(f"Items: {vizdoom_obs.item_count}")
        if vizdoom_obs.secret_count > 0:
            stats_lines.append(f"Secrets: {vizdoom_obs.secret_count}")

        stats_str = ", ".join(stats_lines) if stats_lines else "No stats yet"

        # 上一次动作结果
        action_result = ""
        if vizdoom_obs.last_action:
            action_result = f"\nLast Action: {vizdoom_obs.last_action}"
            action_result += f"\nLast Reward: {vizdoom_obs.last_reward:+.2f}"

        # 生成视觉场景描述（核心改进：从 labels 数据解析，而非占位符）
        scene_desc = Observer.describe_scene(obs, info)

        return f'''
Status: {status_str}
{position_str}
Statistics: {stats_str}
{action_result}

Visual Scene:
{scene_desc}
'''

    @staticmethod
    def save_trajectory(
        history: List[StepHistory], 
        history_dir: str, 
        filename: str = None, 
        metadata: dict = None
    ) -> str:
        """
        保存历史记录到 Parquet 文件
        
        Args:
            history: 步骤历史记录列表
            history_dir: 保存目录
            filename: 文件名（可选）
            metadata: 元数据字典（可选）
        
        Returns:
            保存的文件路径，失败返回 None
        """
        global pd, pq
        try:
            if pd is None:
                import pandas as _pd
                pd = _pd
            if pq is None:
                import pyarrow.parquet as _pq
                pq = _pq
        except ImportError:
            logger.error("Pandas or Pyarrow not found. Cannot save parquet.")
            return None
        except Exception:
            logger.error("Error importing pandas/pyarrow. Cannot save parquet.")
            return None

        try:
            # 创建历史数据目录
            os.makedirs(history_dir, exist_ok=True)

            # 生成文件名（使用时间戳）
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"vizdoom_trajectory_{timestamp}.parquet"
            filepath = os.path.join(history_dir, filename)
                
            # 准备历史记录数据
            history_records = []
            for step_record in history:
                # 处理截图
                screenshot_bytes = None
                screen = step_record.obs.get('screen')
                if screen is not None and isinstance(screen, np.ndarray):
                    img = Image.fromarray(screen.astype(np.uint8))
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    screenshot_bytes = img_byte_arr.getvalue()
                
                # 准备记录 - 注意将复杂对象转换为字符串以兼容 Parquet
                step_dict = step_record.to_dict()
                # 确保 action 和 result 是字符串形式
                step_dict['action'] = str(step_dict.get('action', {}))
                step_dict['result'] = str(step_dict.get('result', {}))
                
                record = {
                    'step': step_record.step,
                    "observation_str": step_record.observation_str,
                    "screenshot": screenshot_bytes,
                    "action_name": step_record.action.get("func_name", "unknown"),
                    "action_args": str(step_record.action.get("args", {})),
                    "reward": step_record.result.get("reward", 0),
                    "done": step_record.result.get("done", False),
                    "info_str": str(step_record.info),
                    "player_pos_x": step_record.player_pos[0] if step_record.player_pos else None,
                    "player_pos_y": step_record.player_pos[1] if step_record.player_pos else None,
                    "player_pos_z": step_record.player_pos[2] if step_record.player_pos else None,
                    "player_angle": step_record.player_angle,
                    "timestamp": str(step_record.timestamp),
                    "llm_completion": step_record.llm_completion,
                    "memory_content": str(step_record.memory_content) if step_record.memory_content else None,
                }
                history_records.append(record)

            # 创建 DataFrame
            df = pd.DataFrame(history_records)

            # 将元数据添加为 DataFrame 的属性
            if metadata:
                df.attrs['metadata'] = metadata

            # 保存为 Parquet 文件
            df.to_parquet(filepath)
            logger.info(f"历史记录已保存到: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存历史记录失败: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def calculate_metrics(
        history: List[StepHistory],
        map_bounds: Tuple[float, float, float, float] = (-1000, -1000, 1000, 1000),
        grid_resolution: float = 50.0
    ) -> Dict[str, Any]:
        """
        计算探索评测指标
        
        Args:
            history: 步骤历史记录列表
            map_bounds: 地图边界 (min_x, min_y, max_x, max_y)
            grid_resolution: 网格分辨率，用于计算覆盖率
        
        Returns:
            包含各项指标的字典
        """
        if not history:
            return ExplorationMetrics().to_dict()
        
        metrics = ExplorationMetrics()
        metrics.episode_length = len(history)
        metrics.total_steps = len(history)
        
        # ========== 1. 累计奖励 ==========
        total_reward = sum(step.result.get("reward", 0) for step in history)
        metrics.total_reward = total_reward
        
        # ========== 2. 地图覆盖率 (Map Coverage) ==========
        # 将连续坐标离散化为网格
        min_x, min_y, max_x, max_y = map_bounds
        grid_width = int((max_x - min_x) / grid_resolution) + 1
        grid_height = int((max_y - min_y) / grid_resolution) + 1
        total_cells = grid_width * grid_height
        
        visited_cells: Set[Tuple[int, int]] = set()
        visited_positions: Set[Tuple[float, float]] = set()
        
        for step in history:
            if step.player_pos:
                x, y = step.player_pos[0], step.player_pos[1]
                # 记录精确位置
                visited_positions.add((round(x, 1), round(y, 1)))
                # 计算网格坐标
                grid_x = int((x - min_x) / grid_resolution)
                grid_y = int((y - min_y) / grid_resolution)
                # 确保在边界内
                grid_x = max(0, min(grid_x, grid_width - 1))
                grid_y = max(0, min(grid_y, grid_height - 1))
                visited_cells.add((grid_x, grid_y))
        
        metrics.visited_positions = len(visited_positions)
        metrics.map_coverage = len(visited_cells) / total_cells if total_cells > 0 else 0.0
        
        # ========== 3. 状态多样性 (State Diversity / Entropy) ==========
        # 基于离散化的状态计算熵
        state_counts = Counter()
        for step in history:
            # 创建状态签名：位置网格 + 生命值区间 + 弹药区间
            game_vars = step.info.get('game_variables', {})
            health = game_vars.get('HEALTH', game_vars.get('health', 100))
            ammo = game_vars.get('AMMO', game_vars.get('AMMO2', game_vars.get('ammo', 0)))
            
            # 离散化
            health_bin = int(health // 20)  # 5 个区间: 0-20, 20-40, ...
            ammo_bin = int(min(ammo, 100) // 10)  # 10 个区间
            
            pos_bin = (0, 0)
            if step.player_pos:
                pos_bin = (
                    int((step.player_pos[0] - min_x) / grid_resolution),
                    int((step.player_pos[1] - min_y) / grid_resolution)
                )
            
            state_key = (pos_bin, health_bin, ammo_bin)
            state_counts[state_key] += 1
        
        metrics.unique_states = len(state_counts)
        
        # 计算状态熵
        if state_counts:
            total = sum(state_counts.values())
            probs = [count / total for count in state_counts.values()]
            state_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            metrics.state_entropy = state_entropy
        
        # ========== 4. 动作多样性 (Action Diversity) ==========
        action_counts = Counter()
        for step in history:
            action = step.action.get('func_name', 'unknown')
            action_counts[action] += 1
        
        metrics.unique_actions_used = len(action_counts)
        
        if action_counts:
            total = sum(action_counts.values())
            probs = [count / total for count in action_counts.values()]
            action_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            metrics.action_entropy = action_entropy
        
        # ========== 5. 视觉新颖性 (Visual Novelty) ==========
        novelty_scores = []
        image_history = []
        
        for step in history:
            screen = step.obs.get('screen')
            if screen is not None and isinstance(screen, np.ndarray):
                # 下采样以加速计算
                downsampled = screen[::4, ::4, :]  # 4x 下采样
                
                if len(image_history) > 0:
                    # 计算与所有历史帧的 MSE 差异
                    diffs = []
                    for hist_img in image_history[-50:]:  # 只比较最近50帧
                        diff = np.mean((downsampled.astype(float) - hist_img.astype(float)) ** 2)
                        normalized_diff = diff / (255.0 ** 2)
                        diffs.append(normalized_diff)
                    
                    # 最小差异作为新颖性度量
                    min_diff = min(diffs)
                    novelty_scores.append(min_diff)
                
                image_history.append(downsampled.copy())
        
        if novelty_scores:
            metrics.avg_visual_novelty = float(np.mean(novelty_scores))
            metrics.max_visual_novelty = float(max(novelty_scores))
            metrics.cumulative_novelty = float(sum(novelty_scores))
        
        # ========== 6. 游戏特定指标 ==========
        last_info = history[-1].info
        game_vars = last_info.get('game_variables', {})
        
        metrics.kills = int(game_vars.get('KILLCOUNT', game_vars.get('kill_count', 0)))
        metrics.items_collected = int(game_vars.get('ITEMCOUNT', game_vars.get('item_count', 0)))
        metrics.secrets_found = int(game_vars.get('SECRETCOUNT', game_vars.get('secret_count', 0)))
        
        # 计算伤害统计
        first_health = 100.0
        last_health = game_vars.get('HEALTH', game_vars.get('health', 100.0))
        if last_health < first_health:
            metrics.damage_taken = first_health - last_health
        
        return metrics.to_dict()

    @staticmethod
    def calculate_novelty_rnd(
        history: List[StepHistory],
        feature_dim: int = 512,
        hidden_dim: int = 256
    ) -> Dict[str, float]:
        """
        使用 Random Network Distillation (RND) 计算新颖性
        这是一种更高级的好奇心驱动方法
        
        Args:
            history: 步骤历史记录
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        
        Returns:
            RND 新颖性指标
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available for RND novelty calculation")
            return {"rnd_novelty": None, "error": "PyTorch not installed"}
        
        # 简单的随机网络和预测网络
        class RandomNetwork(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                # 固定随机网络权重
                for p in self.parameters():
                    p.requires_grad = False
            
            def forward(self, x):
                return self.net(x)
        
        class PredictorNetwork(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.net(x)
        
        # 提取观察特征（简化：使用下采样的图像像素）
        features = []
        for step in history:
            screen = step.obs.get('screen')
            if screen is not None:
                # 大幅下采样并展平
                small = screen[::16, ::16, :].flatten()
                # 截断或填充到固定大小
                if len(small) > 1024:
                    small = small[:1024]
                else:
                    small = np.pad(small, (0, 1024 - len(small)))
                features.append(small.astype(np.float32) / 255.0)
            else:
                features.append(np.zeros(1024, dtype=np.float32))
        
        if not features:
            return {"rnd_avg_novelty": 0.0, "rnd_max_novelty": 0.0}
        
        features = torch.tensor(np.array(features), dtype=torch.float32)
        
        # 初始化网络
        input_dim = features.shape[1]
        random_net = RandomNetwork(input_dim, feature_dim)
        predictor_net = PredictorNetwork(input_dim, feature_dim)
        
        # 计算 RND 新颖性（预测误差）
        with torch.no_grad():
            target_features = random_net(features)
            predicted_features = predictor_net(features)
            
            # 预测误差作为新颖性
            novelty = torch.mean((target_features - predicted_features) ** 2, dim=1)
            novelty = novelty.numpy()
        
        return {
            "rnd_avg_novelty": float(np.mean(novelty)),
            "rnd_max_novelty": float(np.max(novelty)),
            "rnd_cumulative_novelty": float(np.sum(novelty))
        }

    @staticmethod
    def load_trajectory(parquet_file: str):
        """
        从 Parquet 文件加载历史记录
        
        Args:
            parquet_file: Parquet 文件路径
        
        Returns:
            DataFrame 或 None
        """
        global pd, pq
        try:
            if pd is None:
                import pandas as _pd
                pd = _pd
            if pq is None:
                import pyarrow.parquet as _pq
                pq = _pq
        except ImportError:
            logger.error("Pandas or Pyarrow not found. Cannot load parquet.")
            return None

        try:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            logger.info(f"成功加载历史记录: {parquet_file}")
            return df
        except Exception as e:
            logger.error(f"加载历史记录失败: {str(e)}", exc_info=True)
            return None
