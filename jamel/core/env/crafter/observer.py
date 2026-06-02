"""
Crafter 观察者模块 - 负责将 Crafter 环境状态转化为 LLM 能理解的文本表示，并处理数据记录

核心能力：
- 解析 Crafter 语义地图 (semantic map)，将游戏画面转化为结构化文本描述
- 提供周围环境信息：地形、资源、威胁、生物、建筑等
"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
import io
import os
from PIL import Image
import numpy as np
import logging

# Lazy import for pandas/pyarrow to avoid segfaults if libraries are broken
pd = None
pq = None

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

from jamel.core.env.crafter.utils import StepHistory

# Fallback logger if structlog/log_utils is not available
try:
    from jamel.log import log_utils
    logger = log_utils.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# ========== Crafter 语义地图常量 ==========
# 语义地图 ID 到名称的映射（对应 crafter 库的内部编码）
SEMANTIC_IDS: Dict[int, str] = {
    0: 'unknown',
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'path',
    5: 'sand',
    6: 'tree',
    7: 'lava',
    8: 'coal',
    9: 'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
    13: 'player',
    14: 'cow',
    15: 'zombie',
    16: 'skeleton',
    17: 'arrow',
    18: 'plant',
}

# 分类集合，用于场景描述
TERRAIN_NAMES = {'grass', 'stone', 'path', 'sand'}
RESOURCE_NAMES = {'tree', 'coal', 'iron', 'diamond', 'plant'}
THREAT_NAMES = {'zombie', 'skeleton'}
CREATURE_NAMES = {'cow'}
STRUCTURE_NAMES = {'table', 'furnace'}
HAZARD_NAMES = {'lava', 'water'}
PROJECTILE_NAMES = {'arrow'}


def _get_direction(dx: int, dy: int) -> str:
    """根据相对偏移量获取方位描述（Crafter 中 y 轴向下为正）"""
    if dx == 0 and dy == 0:
        return "here"
    parts = []
    if dy < 0:
        parts.append("north")
    elif dy > 0:
        parts.append("south")
    if dx < 0:
        parts.append("west")
    elif dx > 0:
        parts.append("east")
    return "-".join(parts) if parts else "here"


@dataclass
class _CrafterObservation:
    inventory: Dict[str, int]
    status: Dict[str, Any]
    achievements: Dict[str, Any]
    screenshot: np.ndarray
    semantic_map: Optional[np.ndarray] = None   # 原始语义地图
    player_pos: Optional[Tuple[int, int]] = None  # 玩家坐标 (x, y)

    @classmethod
    def from_dict(cls, obs: Dict) -> _CrafterObservation:
        player_pos = obs.get("player_pos")
        if player_pos is not None:
            if isinstance(player_pos, np.ndarray):
                player_pos = tuple(player_pos.tolist())
            else:
                player_pos = tuple(player_pos)
        return cls(
            inventory=obs.get("inventory", {}),
            status=obs.get("status", {}),
            achievements=obs.get("achievements", {}),
            screenshot=obs.get("image", np.zeros((64, 64, 3), dtype=np.uint8)),
            semantic_map=obs.get("semantic_map"),
            player_pos=player_pos,
        )


class Observer:
    """Crafter 观察者，将状态转换为结构化文本"""

    @staticmethod
    def describe_scene(
        semantic_map: Optional[np.ndarray],
        player_pos: Optional[Tuple[int, int]],
        view_radius: int = 7,
    ) -> str:
        """
        解析 Crafter 语义地图，生成人类/LLM 可读的场景描述。

        Args:
            semantic_map: 语义地图 (H x W)，每个像素为物体类型 ID
            player_pos: 玩家在地图上的坐标 (x, y)
            view_radius: 视野半径（以 tile 为单位）

        Returns:
            场景的自然语言描述
        """
        if semantic_map is None or player_pos is None:
            return "No visual information available."

        h, w = semantic_map.shape
        px, py = int(player_pos[0]), int(player_pos[1])

        # 限定可视区域
        x_min = max(0, px - view_radius)
        x_max = min(w, px + view_radius + 1)
        y_min = max(0, py - view_radius)
        y_max = min(h, py + view_radius + 1)

        local_map = semantic_map[y_min:y_max, x_min:x_max]

        # 按类别统计
        terrain_counts: Counter = Counter()
        resources: List[Tuple[str, str, int]] = []   # (name, direction, distance)
        threats: List[Tuple[str, str, int]] = []
        creatures: List[Tuple[str, str, int]] = []
        structures: List[Tuple[str, str, int]] = []
        projectiles: List[Tuple[str, str, int]] = []
        hazard_counts: Counter = Counter()

        for dy in range(local_map.shape[0]):
            for dx in range(local_map.shape[1]):
                cell_id = int(local_map[dy, dx])
                cell_name = SEMANTIC_IDS.get(cell_id, f"unknown_{cell_id}")

                # 相对玩家的偏移
                rel_x = (x_min + dx) - px
                rel_y = (y_min + dy) - py
                direction = _get_direction(rel_x, rel_y)
                distance = max(abs(rel_x), abs(rel_y))

                if cell_name in TERRAIN_NAMES:
                    terrain_counts[cell_name] += 1
                elif cell_name in RESOURCE_NAMES:
                    resources.append((cell_name, direction, distance))
                elif cell_name in THREAT_NAMES:
                    threats.append((cell_name, direction, distance))
                elif cell_name in CREATURE_NAMES:
                    creatures.append((cell_name, direction, distance))
                elif cell_name in STRUCTURE_NAMES:
                    structures.append((cell_name, direction, distance))
                elif cell_name in PROJECTILE_NAMES:
                    projectiles.append((cell_name, direction, distance))
                elif cell_name in HAZARD_NAMES:
                    hazard_counts[cell_name] += 1

        # ---- 组装描述文本 ----
        lines: List[str] = []

        # 1. 地形概况
        if terrain_counts:
            dominant = terrain_counts.most_common(3)
            terrain_desc = ", ".join(f"{name}({count})" for name, count in dominant)
            lines.append(f"Terrain: {terrain_desc}")
        else:
            lines.append("Terrain: unclear")

        # 2. 威胁警告（最重要，放前面）
        if threats:
            threats.sort(key=lambda t: t[2])
            descs = [f"{n} to the {d} ({dist} tiles)" for n, d, dist in threats[:5]]
            lines.append(f"!! THREATS: {'; '.join(descs)}")
        else:
            lines.append("Threats: none detected nearby.")

        # 3. 飞行物
        if projectiles:
            projectiles.sort(key=lambda t: t[2])
            descs = [f"{n} to the {d} ({dist} tiles)" for n, d, dist in projectiles[:3]]
            lines.append(f"! Projectiles: {'; '.join(descs)}")

        # 4. 资源
        if resources:
            resources.sort(key=lambda t: t[2])
            # 按类型聚合
            res_by_type: Dict[str, List[Tuple[str, int]]] = {}
            for name, direction, dist in resources:
                res_by_type.setdefault(name, []).append((direction, dist))
            res_parts = []
            for rname, locs in res_by_type.items():
                closest = min(locs, key=lambda x: x[1])
                count = len(locs)
                res_parts.append(f"{rname} x{count} (nearest: {closest[0]}, {closest[1]} tiles)")
            lines.append(f"Resources: {'; '.join(res_parts)}")
        else:
            lines.append("Resources: none visible.")

        # 5. 建筑
        if structures:
            structures.sort(key=lambda t: t[2])
            descs = [f"{n} to the {d} ({dist} tiles)" for n, d, dist in structures]
            lines.append(f"Structures: {'; '.join(descs)}")

        # 6. 生物
        if creatures:
            creatures.sort(key=lambda t: t[2])
            descs = [f"{n} to the {d} ({dist} tiles)" for n, d, dist in creatures]
            lines.append(f"Creatures: {'; '.join(descs)}")

        # 7. 地形危险
        if hazard_counts:
            haz_parts = [f"{name}({count} tiles)" for name, count in hazard_counts.items()]
            lines.append(f"Hazards nearby: {', '.join(haz_parts)}")

        return "\n".join(lines)

    @staticmethod
    def get_observation(obs: Dict) -> str:
        crafter_obs = _CrafterObservation.from_dict(obs)

        # 构建文本描述
        status_str = ", ".join([f"{k}: {v}" for k, v in crafter_obs.status.items()])
        inv_str = ", ".join([f"{k}: {v}" for k, v in crafter_obs.inventory.items() if v > 0])
        if not inv_str:
            inv_str = "Empty"

        achievements_str = ", ".join([k for k, v in crafter_obs.achievements.items() if v > 0])
        if not achievements_str:
            achievements_str = "None"

        # 生成视觉场景描述（核心改进：从语义地图解析，而非占位符）
        scene_desc = Observer.describe_scene(
            crafter_obs.semantic_map,
            crafter_obs.player_pos,
        )

        return f'''
Status: {status_str}
Inventory: {inv_str}
Achievements: {achievements_str}

Visual Scene:
{scene_desc}
'''

    @staticmethod
    def save_trajectory(history: List[StepHistory], history_dir: str, filename: str=None, metadata: dict=None) -> str:
        """
        保存历史记录到 Parquet 文件
        """
        global pd, pq
        try:
            if pd is None:
                import pandas as pd
            if pq is None:
                import pyarrow.parquet as pq
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
                filename = f"crafter_agent_history_{timestamp}.parquet"
            filepath = os.path.join(history_dir, filename)
                
            # 准备历史记录数据
            history_records = []
            for step_record in history:

                # 处理截图
                screenshot_bytes = None
                if "image" in step_record.obs:
                    screenshot_data = step_record.obs["image"]
                    if isinstance(screenshot_data, np.ndarray):
                        img = Image.fromarray(screenshot_data.astype(np.uint8))
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        screenshot_bytes = img_byte_arr.getvalue()
                
                # 准备记录
                record = {
                    'step': step_record.step,
                    "observation_str": step_record.observation_str,
                    "screenshot": screenshot_bytes,
                    "action": str(step_record.action),
                    "reward": step_record.result.get("reward", 0),
                    "done": step_record.result.get("done", False),
                    "info": str(step_record.info),
                    **step_record.to_dict()
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
    def calculate_metrics(history: List[StepHistory], map_size: Tuple[int, int] = (64, 64)) -> Dict[str, float]:
        """
        计算探索评测指标
        
        Args:
            history: 步骤历史记录列表
            map_size: 地图大小，默认 (64, 64)
        
        Returns:
            包含各项指标的字典
        """
        if not history:
            return {}
        
        metrics = {}
        
        # ========== 1. 官方 Crafter Score (几何平均分) ==========
        # Crafter 论文使用成就解锁率的几何平均作为评分
        last_info = history[-1].info
        achievements = last_info.get("achievements", {})
        
        if achievements:
            # 计算每个成就的解锁率 (0 或 1)
            achievement_rates = []
            for name, count in achievements.items():
                # 成就解锁则为 1，否则为 0
                rate = 1.0 if count > 0 else 0.0
                achievement_rates.append(rate)
            
            # 几何平均 (加 epsilon 避免 0)
            # 官方公式: exp(mean(log(rate + 0.01))) - 0.01
            eps = 0.01
            log_rates = [np.log(r + eps) for r in achievement_rates]
            crafter_score = np.exp(np.mean(log_rates)) - eps
            crafter_score = max(0.0, crafter_score)  # 确保非负
            
            total_achievements = len(achievements)
            unlocked_achievements = sum(1 for v in achievements.values() if v > 0)
            achievement_coverage = unlocked_achievements / total_achievements if total_achievements > 0 else 0.0
        else:
            crafter_score = 0.0
            achievement_coverage = 0.0
            unlocked_achievements = 0
            total_achievements = 0
        
        metrics["crafter_score"] = round(crafter_score, 4)
        metrics["achievement_coverage"] = round(achievement_coverage, 4)
        metrics["unlocked_achievements"] = unlocked_achievements
        metrics["total_achievements"] = total_achievements
        
        # ========== 2. 地图探索覆盖率 (Map Exploration Coverage) ==========
        # 统计访问过的唯一坐标
        visited_positions = set()
        for step in history:
            pos = step.player_pos
            if pos is not None:
                visited_positions.add(pos)
        
        total_map_cells = map_size[0] * map_size[1]
        map_coverage = len(visited_positions) / total_map_cells
        
        metrics["visited_cells"] = len(visited_positions)
        metrics["map_coverage"] = round(map_coverage, 6)
        metrics["map_coverage_percent"] = round(map_coverage * 100, 2)
        
        # ========== 3. 生存时间 (Survival Time) ==========
        # 统计存活的步数
        survival_steps = len(history)
        
        # 检查是否死亡 (health <= 0)
        died = False
        for i, step in enumerate(history):
            status = step.info.get("status", {})
            health = status.get("health", 9)
            if health <= 0:
                died = True
                survival_steps = i + 1
                break
        
        metrics["survival_steps"] = survival_steps
        metrics["died"] = died
        
        # ========== 4. 视觉新颖性 (Visual Novelty) ==========
        # 计算每帧与历史帧的平均差异
        novelty_scores = []
        image_history = []
        
        for step in history:
            if "image" in step.obs and step.obs["image"] is not None:
                current_image = step.obs["image"]
                
                if len(image_history) > 0:
                    # 计算与所有历史帧的平均差异
                    diffs = []
                    for hist_img in image_history:
                        # 使用归一化的像素差异 (MSE)
                        diff = np.mean((current_image.astype(float) - hist_img.astype(float)) ** 2)
                        # 归一化到 [0, 1]
                        normalized_diff = diff / (255.0 ** 2)
                        diffs.append(normalized_diff)
                    
                    # 与最相似帧的差异 (越大越新颖)
                    min_diff = min(diffs)
                    avg_diff = np.mean(diffs)
                    novelty_scores.append(min_diff)
                
                image_history.append(current_image.copy())
        
        if novelty_scores:
            avg_visual_novelty = np.mean(novelty_scores)
            max_visual_novelty = max(novelty_scores)
            # 累计新颖性 (积分)
            cumulative_novelty = sum(novelty_scores)
        else:
            avg_visual_novelty = 0.0
            max_visual_novelty = 0.0
            cumulative_novelty = 0.0
        
        metrics["avg_visual_novelty"] = round(avg_visual_novelty, 6)
        metrics["max_visual_novelty"] = round(max_visual_novelty, 6)
        metrics["cumulative_visual_novelty"] = round(cumulative_novelty, 4)
        
        # ========== 5. 状态多样性 (State Diversity) ==========
        # 基于 Inventory 状态的多样性
        unique_inventory_states = set()
        for step in history:
            # 从 info 中获取 inventory（不包括 status）
            inv = step.info.get("inventory", {})
            # 过滤掉 status 字段 (health, food, drink, energy)
            filtered_inv = {k: v for k, v in inv.items() 
                          if k not in ('health', 'food', 'drink', 'energy')}
            state_tuple = tuple(sorted(filtered_inv.items()))
            unique_inventory_states.add(state_tuple)
        
        metrics["state_diversity"] = len(unique_inventory_states)
        
        # ========== 6. 动作多样性 (Action Diversity) ==========
        action_counts = {}
        for step in history:
            action = step.action.get("func_name", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        unique_actions = len(action_counts)
        # 动作熵
        if action_counts:
            total_actions = sum(action_counts.values())
            probs = [c / total_actions for c in action_counts.values()]
            action_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        else:
            action_entropy = 0.0
        
        metrics["unique_actions_used"] = unique_actions
        metrics["action_entropy"] = round(action_entropy, 4)
        
        # ========== 7. 资源收集效率 ==========
        # 统计收集到的资源总量
        final_inventory = last_info.get("inventory", {})
        resource_keys = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
        total_resources = sum(final_inventory.get(k, 0) for k in resource_keys)
        
        metrics["total_resources_collected"] = total_resources
        metrics["resources_per_step"] = round(total_resources / max(1, survival_steps), 4)
        
        return metrics
    
    @staticmethod
    def calculate_clip_novelty(history: List[StepHistory], model_name: str = "ViT-B/32") -> Dict[str, float]:
        """
        使用 CLIP 模型计算视觉新颖性 (可选，需要安装 clip)
        
        Args:
            history: 步骤历史记录列表
            model_name: CLIP 模型名称
        
        Returns:
            CLIP 新颖性指标
        """
        try:
            import torch
            import clip
            from PIL import Image as PILImage
        except ImportError:
            logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            return {"clip_novelty": None, "error": "CLIP not installed"}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)
        
        embeddings = []
        novelty_scores = []
        
        with torch.no_grad():
            for step in history:
                if "image" in step.obs and step.obs["image"] is not None:
                    # 转换为 PIL Image 并预处理
                    img = PILImage.fromarray(step.obs["image"].astype(np.uint8))
                    img_input = preprocess(img).unsqueeze(0).to(device)
                    
                    # 获取图像 embedding
                    embedding = model.encode_image(img_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # 归一化
                    
                    if len(embeddings) > 0:
                        # 计算与所有历史帧的余弦相似度
                        history_embeddings = torch.stack(embeddings)
                        similarities = (embedding @ history_embeddings.T).squeeze()
                        
                        # 新颖性 = 1 - 最大相似度
                        max_similarity = similarities.max().item()
                        novelty = 1.0 - max_similarity
                        novelty_scores.append(novelty)
                    
                    embeddings.append(embedding.squeeze())
        
        if novelty_scores:
            return {
                "clip_avg_novelty": round(np.mean(novelty_scores), 4),
                "clip_max_novelty": round(max(novelty_scores), 4),
                "clip_cumulative_novelty": round(sum(novelty_scores), 4)
            }
        
        return {"clip_avg_novelty": 0.0, "clip_max_novelty": 0.0, "clip_cumulative_novelty": 0.0}
