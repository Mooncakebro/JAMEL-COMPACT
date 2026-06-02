import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    import crafter
    CRAFTER_AVAILABLE = True
except ImportError:
    CRAFTER_AVAILABLE = False
    print("Crafter not found. Using MockCrafterEnv.")

from jamel.core.env.crafter.observer import Observer
from jamel.core.env.crafter.action_space import text_to_action, get_action_space


# ========== 语义地图颜色映射 (用于 Mock 环境渲染可视化图像) ==========
SEMANTIC_COLORS = {
    0: (20, 20, 20),       # unknown - 深灰
    1: (30, 100, 200),     # water - 蓝
    2: (60, 180, 60),      # grass - 绿
    3: (140, 140, 140),    # stone - 灰
    4: (180, 160, 120),    # path - 浅棕
    5: (220, 200, 120),    # sand - 黄
    6: (20, 120, 20),      # tree - 深绿
    7: (220, 80, 20),      # lava - 橙红
    8: (60, 60, 60),       # coal - 深灰
    9: (180, 180, 200),    # iron - 银
    10: (80, 220, 240),    # diamond - 青
    11: (140, 100, 60),    # table - 棕
    12: (160, 60, 40),     # furnace - 暗红
    13: (255, 255, 255),   # player - 白
    14: (200, 170, 120),   # cow - 棕
    15: (180, 40, 40),     # zombie - 红
    16: (220, 220, 200),   # skeleton - 骨白
    17: (240, 240, 40),    # arrow - 黄
    18: (100, 220, 100),   # plant - 浅绿
}

class MockCrafterEnv:
    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._player = type('obj', (object,), {
            'health': 9, 'food': 9, 'drink': 9, 'energy': 9
        })
        self._step_count = 0
        self._player_pos = np.array([32, 32])
        self._semantic = self._generate_semantic_map()
        self._wood_collected = 0

    def _generate_semantic_map(self) -> np.ndarray:
        """
        生成一张程序化的语义地图，包含多种地形和物体，
        使得场景描述有实际内容可以解析。
        """
        sem = np.ones((64, 64), dtype=np.uint8) * 2  # 默认: grass (ID=2)

        # 水域 (ID=1): 地图上方一条河流
        sem[0:4, :] = 1
        # 沙地 (ID=5): 靠近水域的河岸
        sem[4:6, :] = 5
        # 石头地形 (ID=3): 右下角区域
        sem[50:60, 50:60] = 3
        # 路径 (ID=4): 一条横穿地图的小路
        sem[30:32, :] = 4

        # 树木 (ID=6)
        tree_positions = self._rng.randint(6, 58, size=(40, 2))
        for x, y in tree_positions:
            if sem[y, x] == 2:  # 只在草地上放树
                sem[y, x] = 6

        # 煤矿 (ID=8)
        for _ in range(8):
            x, y = self._rng.randint(10, 54, size=2)
            if sem[y, x] in (2, 3):
                sem[y, x] = 8

        # 铁矿 (ID=9)
        for _ in range(4):
            x, y = self._rng.randint(10, 54, size=2)
            if sem[y, x] in (2, 3):
                sem[y, x] = 9

        # 钻石 (ID=10) - 稀有
        for _ in range(2):
            x, y = self._rng.randint(20, 44, size=2)
            if sem[y, x] in (3,):
                sem[y, x] = 10

        # 牛 (ID=14)
        for _ in range(3):
            x, y = self._rng.randint(10, 54, size=2)
            if sem[y, x] == 2:
                sem[y, x] = 14

        # 僵尸 (ID=15) - 在远离起始点的区域
        for _ in range(3):
            x, y = self._rng.randint(40, 60, size=2)
            if sem[y, x] in (2, 4):
                sem[y, x] = 15

        return sem

    def reset(self):
        self._step_count = 0
        self._player_pos = np.array([32, 32])
        self._wood_collected = 0
        self._semantic = self._generate_semantic_map()
        return self._render_from_semantic()

    def step(self, action):
        self._step_count += 1
        # Simulate movement
        if action == 1:  # move_left
            self._player_pos[0] = max(0, self._player_pos[0] - 1)
        elif action == 2:  # move_right
            self._player_pos[0] = min(63, self._player_pos[0] + 1)
        elif action == 3:  # move_up
            self._player_pos[1] = max(0, self._player_pos[1] - 1)
        elif action == 4:  # move_down
            self._player_pos[1] = min(63, self._player_pos[1] + 1)

        # 模拟 do() 动作: 如果面前有树，收集木头
        px, py = self._player_pos
        if action == 5:  # do
            if 0 <= py < 64 and 0 <= px < 64 and self._semantic[py, px] == 6:
                self._semantic[py, px] = 2  # 树变为草地
                self._wood_collected += 1

        # 随机移动僵尸
        zombie_positions = np.argwhere(self._semantic == 15)
        for zy, zx in zombie_positions:
            self._semantic[zy, zx] = 2
            nzx = max(0, min(63, zx + self._rng.randint(-1, 2)))
            nzy = max(0, min(63, zy + self._rng.randint(-1, 2)))
            if self._semantic[nzy, nzx] == 2:
                self._semantic[nzy, nzx] = 15

        obs = self._render_from_semantic()
        reward = 0.1
        done = False
        info = {
            'inventory': {
                'health': 9, 'food': 9, 'drink': 9, 'energy': 9,
                'wood': self._wood_collected, 'stone': 0
            },
            'achievements': {'place_stone': 0, 'collect_wood': min(1, self._wood_collected)},
            'discount': 1.0,
            'player_pos': self._player_pos.copy(),
            'semantic': self._semantic.copy(),
            'reward': reward
        }
        return obs, reward, done, info

    def _render_from_semantic(self, view_radius: int = 9) -> np.ndarray:
        """
        从语义地图渲染一张以玩家为中心的俯视图像。
        每个 tile 渲染为 tile_size x tile_size 像素，生成 64x64 的最终图像。
        """
        px, py = int(self._player_pos[0]), int(self._player_pos[1])
        view_size = 2 * view_radius + 1  # 可见 tile 数
        tile_px = max(1, 64 // view_size)  # 每个 tile 的像素大小
        img_size = tile_px * view_size

        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        for dy in range(view_size):
            for dx in range(view_size):
                world_x = px - view_radius + dx
                world_y = py - view_radius + dy
                # 边界检查
                if 0 <= world_x < 64 and 0 <= world_y < 64:
                    cell_id = int(self._semantic[world_y, world_x])
                else:
                    cell_id = 0  # 地图外 = unknown

                color = SEMANTIC_COLORS.get(cell_id, (20, 20, 20))
                y1 = dy * tile_px
                y2 = y1 + tile_px
                x1 = dx * tile_px
                x2 = x1 + tile_px
                img[y1:y2, x1:x2] = color

        # 标记玩家位置（中心）为白色十字
        center = view_radius * tile_px + tile_px // 2
        cross_size = max(1, tile_px // 3)
        for offset in range(-cross_size, cross_size + 1):
            cy = center + offset
            cx = center + offset
            if 0 <= cy < img_size:
                img[cy, center] = (255, 255, 255)
            if 0 <= cx < img_size:
                img[center, cx] = (255, 255, 255)

        # 缩放到 64x64 (简单最近邻)
        if img_size != 64:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img)
            pil_img = pil_img.resize((64, 64), PILImage.NEAREST)
            img = np.array(pil_img)

        return img

    def close(self):
        pass

class CrafterTask:
    """
    Crafter 环境任务封装
    """
    def __init__(self, seed: int = None):
        if CRAFTER_AVAILABLE:
            self.env = crafter.Env(seed=seed)
        else:
            self.env = MockCrafterEnv(seed=seed)
            
        self.action_space_desc = get_action_space()
        self._last_obs = None
        self._last_info = None

    def reset(self) -> Tuple[str, Dict]:
        obs = self.env.reset() # obs is image
        # Crafter reset doesn't return info by default in old gym versions, 
        # but let's check or assume we can get initial info or wait for first step.
        # However, to conform to Observer, we need initial info.
        # We can simulate a no-op step or manually construct initial info.
        
        # Initial status
        self._last_info = {
            "inventory": {k: 0 for k in [
                'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 
                'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'wood_sword', 
                'stone_sword', 'iron_sword'
            ]},
            "status": {
                "health": 9,
                "food": 9,
                "drink": 9,
                "energy": 9
            },
            "achievements": {} 
        }
        
        enhanced_obs = self._enhance_obs(obs, self._last_info)
        obs_str = Observer.get_observation(enhanced_obs)
        return obs_str, self._last_info

    def step(self, action_str: str) -> Tuple[str, float, bool, bool, Dict]:
        try:
            action_idx = text_to_action(action_str)
        except ValueError as e:
            # Handle invalid action
            return f"Invalid action: {action_str}", 0.0, False, False, {"error": str(e)}

        obs, reward, done, info = self.env.step(action_idx)
        
        # Crafter info contains inventory and achievements
        self._last_info = info
        # Extract status explicitly if not in info (Crafter usually puts them in observation dict if using specific wrapper, 
        # but standard env returns image obs. Info contains 'inventory', 'achievements'.
        # Player status (health etc) is visualized in the bar, but also available in internal state.
        # We can access env._player.health etc if needed, but let's rely on info for now.
        # Actually crafter info keys: 'inventory', 'achievements', 'discount'.
        # We might need to hack to get health/food if not in info.
        
        # Extract status from inventory (crafter puts health/food/drink/energy in inventory dict)
        inventory = info.get('inventory', {})
        self._last_info['status'] = {
            'health': inventory.get('health', 9),
            'food': inventory.get('food', 9),
            'drink': inventory.get('drink', 9),
            'energy': inventory.get('energy', 9)
        }
        
        # Add raw image to info for recording
        self._last_info['image'] = obs
        
        # Expose player_pos and semantic map for metrics calculation
        # player_pos is already in info from crafter
        # semantic is a 64x64 map of terrain types
        
        enhanced_obs = self._enhance_obs(obs, self._last_info)
        obs_str = Observer.get_observation(enhanced_obs)
        
        return obs_str, reward, done, False, self._last_info

    def _enhance_obs(self, image_obs: np.ndarray, info: Dict) -> Dict:
        """
        将原始图像观察与 info 中的语义数据合并，供 Observer 解析。
        关键改进：传入语义地图 (semantic map) 和玩家坐标 (player_pos)，
        使 Observer 能从游戏引擎数据生成真实的场景描述，而非使用占位符。
        """
        return {
            "image": image_obs,
            "inventory": info.get("inventory", {}),
            "status": info.get("status", {}),
            "achievements": info.get("achievements", {}),
            "semantic_map": info.get("semantic"),       # 语义地图 (H x W ndarray)
            "player_pos": info.get("player_pos"),       # 玩家坐标 (x, y)
        }

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
