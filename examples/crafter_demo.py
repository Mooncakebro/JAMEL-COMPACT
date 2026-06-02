"""
Crafter 环境演示脚本

演示如何：
1. 实例化 Crafter 环境
2. 运行 Random Agent
3. 展示增强后的视觉场景描述（语义地图 → 文本）
4. 保存 trajectory (parquet) 和 GIF 动画
"""

import os
import random
import time
import numpy as np
from PIL import Image

from jamel.core.env.crafter import CrafterTask, Observer, get_action_space, StepHistory
from jamel.core.env.crafter.action_space import ACTION_MAP


def random_agent_policy():
    """随机选择一个动作"""
    actions = list(ACTION_MAP.keys())
    return f"{random.choice(actions)}()"


def save_frames_as_gif(
    frames: list,
    output_path: str,
    duration: int = 200,
    scale: int = 4,
):
    """
    将 numpy 数组帧序列保存为 GIF 动画

    Args:
        frames: list of np.ndarray (H, W, 3)
        output_path: 输出路径
        duration: 每帧持续时间 (毫秒)
        scale: 放大倍数（Crafter 原始是 64x64 太小了）
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


def main():
    print("=" * 60)
    print(" Crafter Environment Demo")
    print(" Visual Scene Description + Trajectory + GIF")
    print("=" * 60)

    # 1. 实例化 Crafter 环境
    try:
        task = CrafterTask(seed=42)
        print("CrafterTask initialized")
        obs_str, info = task.reset()
        print("Task reset done")
    except Exception as e:
        print(f"Error initializing task: {e}")
        return

    print("\n" + "-" * 40)
    print(" Initial Observation (LLM sees this):")
    print("-" * 40)
    print(obs_str)

    history = []
    frames = []       # 收集每帧图像用于 GIF
    max_steps = 20

    # 保存初始帧
    initial_image = info.get("image")
    if initial_image is None and hasattr(task.env, '_render_from_semantic'):
        initial_image = task.env._render_from_semantic()
    if initial_image is not None:
        frames.append(initial_image.copy())

    print(f"\nRunning {max_steps} steps with Random Agent...\n")

    for i in range(max_steps):
        # 2. 选择动作
        action_str = random_agent_policy()

        # 3. 执行动作
        next_obs_str, reward, done, _, next_info = task.step(action_str)

        # 4. 收集帧图像
        frame_image = next_info.get("image")
        if frame_image is not None:
            frames.append(frame_image.copy())

        # 5. 展示前几步的详细观察
        if i < 5:
            print(f"{'='*50}")
            print(f" Step {i+1}/{max_steps}: Action -> {action_str}")
            print(f"{'='*50}")
            print(next_obs_str)
            print(f"  -> Reward: {reward:.2f}")
        elif i == 5:
            print(f"\n... (remaining steps omitted for brevity) ...\n")

        # 6. 记录历史
        step_record = StepHistory(
            step=i,
            obs={"image": frame_image},
            info=next_info,
            observation_str=next_obs_str,
            llm_completion=f"I choose {action_str} randomly.",
            memory_content=None,
            action={"func_name": action_str},
            result={"reward": reward, "done": done},
            timestamp=time.time()
        )
        history.append(step_record)

        if done:
            print(f"Episode finished at step {i+1}.")
            break

    # 7. 计算指标
    print("\n" + "=" * 60)
    print("           EXPLORATION METRICS")
    print("=" * 60)

    metrics = Observer.calculate_metrics(history)

    print(f"\n[Crafter Score]")
    print(f"  Crafter Score: {metrics.get('crafter_score', 0):.4f}")
    print(f"  Achievement Coverage: {metrics.get('achievement_coverage', 0):.2%}")
    print(f"  Unlocked: {metrics.get('unlocked_achievements', 0)}/{metrics.get('total_achievements', 0)}")

    print(f"\n[Map Exploration]")
    print(f"  Visited Cells: {metrics.get('visited_cells', 0)}")
    print(f"  Map Coverage: {metrics.get('map_coverage_percent', 0):.2f}%")

    print(f"\n[Survival]")
    print(f"  Steps: {metrics.get('survival_steps', 0)}, Died: {'Yes' if metrics.get('died', False) else 'No'}")

    print(f"\n[Action Diversity]")
    print(f"  Unique Actions: {metrics.get('unique_actions_used', 0)}, Entropy: {metrics.get('action_entropy', 0):.4f}")

    # 8. 保存 GIF
    print("\n" + "=" * 60)
    print(" SAVING OUTPUTS")
    print("=" * 60)

    output_dir = "crafter_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    # GIF
    if frames:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(output_dir, f"crafter_replay_{timestamp}.gif")
        save_frames_as_gif(frames, gif_path, duration=300, scale=6)

        # 也保存关键帧为 PNG
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
        for idx in sample_indices:
            if 0 <= idx < len(frames):
                img = Image.fromarray(frames[idx].astype(np.uint8))
                img = img.resize((img.width * 6, img.height * 6), Image.NEAREST)
                img.save(os.path.join(frames_dir, f"frame_{idx:03d}.png"))
        print(f"  Key frames saved to: {frames_dir}/")

    # Trajectory (parquet)
    trajectory_path = Observer.save_trajectory(
        history,
        history_dir=output_dir,
        metadata={
            "agent": "RandomAgent",
            "seed": 42,
            "max_steps": max_steps,
            "total_steps": len(history),
        }
    )
    if trajectory_path:
        print(f"  Trajectory saved: {trajectory_path}")
    else:
        print("  Trajectory save skipped (pandas/pyarrow not available).")

    # 9. 场景描述独立演示
    print("\n" + "=" * 60)
    print(" Scene Description Demo")
    print("=" * 60)

    semantic_map = next_info.get("semantic")
    player_pos = next_info.get("player_pos")
    if semantic_map is not None and player_pos is not None:
        print(f"  Player at ({player_pos[0]}, {player_pos[1]}), view radius=7:")
        scene_text = Observer.describe_scene(semantic_map, player_pos, view_radius=7)
        print(f"\n{scene_text}")

    task.close()
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
