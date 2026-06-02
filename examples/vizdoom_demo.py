#!/usr/bin/env python3
"""
Vizdoom 环境演示脚本

演示如何：
1. 实例化 Vizdoom 环境
2. 运行 Random Agent
3. 展示增强后的视觉场景描述（Labels -> Text）
4. 保存 trajectory (parquet) 和 GIF 动画
"""

import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from PIL import Image

# 导入 Vizdoom 环境模块
from jamel.core.env.vizdoom import (
    VizdoomTask,
    Observer,
    get_action_space,
    StepHistory,
)
from jamel.core.env.vizdoom.action_space import get_available_actions


def save_frames_as_gif(
    frames: list,
    output_path: str,
    duration: int = 100,
    scale: int = 2,
):
    """
    将 numpy 数组帧序列保存为 GIF 动画

    Args:
        frames: list of np.ndarray (H, W, 3)
        output_path: 输出路径
        duration: 每帧持续时间 (毫秒)
        scale: 放大倍数
    """
    if not frames:
        print("  No frames to save.")
        return None

    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        if scale > 1:
            new_size = (img.width * scale, img.height * scale)
            img = img.resize(new_size, Image.NEAREST)
        pil_frames.append(img)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"  GIF saved: {output_path} ({len(pil_frames)} frames, {img.width}x{img.height})")
    return output_path


class RandomAgent:
    """
    随机智能体 - 随机选择文本动作
    
    用于测试环境和基线评估
    """
    
    def __init__(self, scenario: str = "basic", seed: int = None):
        """
        初始化随机智能体
        
        Args:
            scenario: Vizdoom 场景名称
            seed: 随机种子
        """
        self.scenario = scenario
        self.available_actions = get_available_actions(scenario)
        self.rng = random.Random(seed)
        
        print(f"RandomAgent initialized for scenario: {scenario}")
        print(f"Available actions: {self.available_actions}")
    
    def select_action(self, observation: str) -> str:
        """
        根据观察选择动作（随机选择）
        
        Args:
            observation: LLM 可读的观察文本
        
        Returns:
            文本形式的动作，如 "move_left()"
        """
        action_name = self.rng.choice(self.available_actions)
        return f"{action_name}()"
    
    def reset(self):
        """重置智能体状态（随机智能体无状态）"""
        pass


def run_episode(
    env: VizdoomTask,
    agent: RandomAgent,
    max_steps: int = 500,
    verbose: bool = True,
    show_scene_desc: bool = True,
    collect_frames: bool = False,
) -> tuple:
    """
    运行一个完整的 episode

    Args:
        env: Vizdoom 环境
        agent: 智能体
        max_steps: 最大步数
        verbose: 是否打印详细信息
        show_scene_desc: 是否展示详细的场景描述（前几步）
        collect_frames: 是否收集帧用于 GIF

    Returns:
        (history, frames): 步骤历史记录列表, 帧列表
    """
    history: List[StepHistory] = []
    frames: List[np.ndarray] = []

    # 重置环境和智能体
    obs_str, info = env.reset()
    agent.reset()

    # 收集初始帧
    if collect_frames and env._last_obs is not None:
        screen = env._last_obs.get('screen')
        if screen is not None:
            frames.append(screen.copy())

    if verbose:
        print("\n" + "=" * 60)
        print("Episode started")
        print("=" * 60)
        print(f"\nInitial observation (LLM sees this):\n{obs_str}")

    total_reward = 0.0

    for step in range(max_steps):
        # 选择动作
        action_str = agent.select_action(obs_str)

        # 执行动作
        next_obs_str, reward, done, truncated, next_info = env.step(action_str)

        total_reward += reward

        # 收集帧
        if collect_frames and env._last_obs is not None:
            screen = env._last_obs.get('screen')
            if screen is not None:
                frames.append(screen.copy())

        # 记录历史
        step_record = StepHistory(
            step=step,
            obs=env._last_obs,
            info=next_info,
            observation_str=obs_str,
            llm_completion="",
            memory_content=None,
            action={"func_name": action_str.rstrip("()"), "args": {}},
            result={"reward": reward, "done": done, "truncated": truncated},
            timestamp=datetime.now(),
        )
        history.append(step_record)

        # 展示前几步的详细场景描述
        if verbose and show_scene_desc and step < 5:
            print(f"\n{'='*50}")
            print(f" Step {step}: Action={action_str}, Reward={reward:+.2f}")
            print(f"{'='*50}")
            print(next_obs_str)
        elif verbose and step == 5:
            print(f"\n... (remaining steps omitted for brevity) ...\n")
        elif verbose and step % 100 == 0 and step > 5:
            game_vars = next_info.get('game_variables', {})
            print(f"Step {step}: Reward={reward:.2f}, Total={total_reward:.2f}, "
                  f"Health={game_vars.get('HEALTH', 'N/A')}")

        # 检查终止条件
        if done or truncated:
            if verbose:
                print(f"\nEpisode ended at step {step}")
                print(f"Done: {done}, Truncated: {truncated}")
            break

        obs_str = next_obs_str

    if verbose:
        print(f"\nTotal reward: {total_reward:.2f}")
        print(f"Total steps: {len(history)}")

    return history, frames


def evaluate_exploration(history: List[StepHistory]) -> Dict[str, Any]:
    """
    评估探索性能
    
    Args:
        history: 步骤历史记录
    
    Returns:
        探索指标字典
    """
    # 使用 Observer 的指标计算方法
    metrics = Observer.calculate_metrics(history)
    return metrics


def print_metrics(metrics: Dict[str, Any], title: str = "Exploration Metrics"):
    """
    格式化打印指标
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    # 分组打印
    groups = {
        "基础指标": ["total_steps", "episode_length", "total_reward"],
        "覆盖率指标": ["visited_positions", "map_coverage", "sector_coverage"],
        "多样性指标": ["state_entropy", "unique_states", "action_entropy", "unique_actions_used"],
        "新颖性指标": ["avg_visual_novelty", "max_visual_novelty", "cumulative_novelty"],
        "游戏指标": ["kills", "items_collected", "secrets_found", "damage_taken"],
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")


def main():
    """主函数"""
    print("=" * 60)
    print(" Vizdoom Environment Demo")
    print(" Visual Scene Description (Labels -> Text)")
    print(" Random Agent + Exploration Metrics")
    print("=" * 60)
    
    # 配置
    SCENARIO = "basic"  # 可选: "basic", "health_gathering", "my_way_home" 等
    SEED = 42
    MAX_STEPS = 300
    NUM_EPISODES = 3
    
    # 打印动作空间
    print(f"\nScenario: {SCENARIO}")
    print("\nAction Space:")
    print(get_action_space(SCENARIO))
    
    # 创建环境和智能体
    env = VizdoomTask(scenario=SCENARIO, seed=SEED)
    agent = RandomAgent(scenario=SCENARIO, seed=SEED)
    
    all_metrics = []
    all_frames = []

    # 运行多个 episodes
    for episode in range(NUM_EPISODES):
        print(f"\n{'='*60}")
        print(f" Episode {episode + 1}/{NUM_EPISODES}")
        print(f"{'='*60}")

        # 运行 episode（最后一个 episode 收集帧用于 GIF）
        collect = (episode == NUM_EPISODES - 1)
        history, frames = run_episode(
            env, agent, max_steps=MAX_STEPS, verbose=True,
            collect_frames=collect
        )
        if collect:
            all_frames = frames

        # 计算指标
        metrics = evaluate_exploration(history)
        all_metrics.append(metrics)

        # 打印指标
        print_metrics(metrics, f"Episode {episode + 1} Metrics")
    
    # 打印汇总统计
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print(" Aggregate Statistics (across all episodes)")
        print("=" * 60)
        
        # 计算平均值
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if m[key] is not None]
            if values and isinstance(values[0], (int, float)):
                avg_metrics[key] = np.mean(values)
                std_metrics = np.std(values)
                print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")
    
    # ========== 保存输出 ==========
    print("\n" + "=" * 60)
    print(" SAVING OUTPUTS")
    print("=" * 60)

    output_dir = "./vizdoom_trajectories"

    # GIF
    if all_frames:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 每隔几帧取一帧，避免 GIF 太大
        step_size = max(1, len(all_frames) // 100)
        sampled_frames = all_frames[::step_size]
        gif_path = os.path.join(output_dir, f"vizdoom_replay_{timestamp}.gif")
        save_frames_as_gif(sampled_frames, gif_path, duration=100, scale=2)

        # 保存关键帧 PNG
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        sample_indices = [0, len(all_frames)//4, len(all_frames)//2, 3*len(all_frames)//4, len(all_frames)-1]
        for idx in sample_indices:
            if 0 <= idx < len(all_frames):
                img = Image.fromarray(all_frames[idx].astype(np.uint8))
                img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
                img.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        print(f"  Key frames saved to: {frames_dir}/")

    # Trajectory (parquet)
    if history:
        save_path = Observer.save_trajectory(
            history,
            history_dir=output_dir,
            metadata={
                "scenario": SCENARIO,
                "agent": "RandomAgent",
                "seed": SEED,
                "max_steps": MAX_STEPS,
            }
        )
        if save_path:
            print(f"  Trajectory saved: {save_path}")

    # ========== 场景描述独立演示 ==========
    print("\n" + "=" * 60)
    print(" Scene Description Demo (standalone)")
    print("=" * 60)
    print("\nUsing Observer.describe_scene() directly on the last state:")

    if env._last_obs is not None and env._last_info is not None:
        scene_text = Observer.describe_scene(env._last_obs, env._last_info)
        print(f"\n{scene_text}")
    else:
        print("  No observation available for standalone demo.")

    # 清理
    env.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
