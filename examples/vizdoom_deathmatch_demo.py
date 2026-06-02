#!/usr/bin/env python3
"""
VizDoom Deathmatch 环境演示脚本

演示：
1. 实例化 Deathmatch 环境（VizDoom 最难场景）
2. Random Agent 随机选择文本动作
3. 收集轨迹并保存 Parquet + GIF
4. 计算完整的探索评测指标（覆盖率、状态熵、新颖性、战斗表现）
"""

import os
import random
import time
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from jamel.core.env.vizdoom_deathmatch import (
    DeathmatchTask,
    Observer,
    get_action_space,
    get_available_actions,
    StepHistory,
    DeathmatchMetrics,
    compute_metrics,
)


def save_frames_as_gif(
    frames: list,
    output_path: str,
    duration: int = 100,
    scale: int = 2,
):
    if not frames:
        print("  No frames to save.")
        return None

    pil_frames = []
    for f in frames:
        img = Image.fromarray(f.astype(np.uint8))
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        pil_frames.append(img)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0,
    )
    print(f"  GIF saved: {output_path} ({len(pil_frames)} frames)")
    return output_path


class RandomAgent:
    """随机智能体 - 在 Deathmatch 动作空间中随机选择"""

    def __init__(self, seed: int = None):
        self.available_actions = get_available_actions()
        self.rng = random.Random(seed)
        print(f"RandomAgent initialized for DEATHMATCH")
        print(f"Available actions ({len(self.available_actions)}): {self.available_actions}")

    def select_action(self, observation: str) -> str:
        name = self.rng.choice(self.available_actions)
        return f"{name}()"

    def reset(self):
        pass


def run_episode(
    env: DeathmatchTask,
    agent: RandomAgent,
    max_steps: int = 500,
    verbose: bool = True,
    collect_frames: bool = False,
) -> tuple:
    history: List[StepHistory] = []
    frames: List[np.ndarray] = []

    obs_str, info = env.reset()
    agent.reset()

    if collect_frames and env._last_obs is not None:
        screen = env._last_obs.get('screen')
        if screen is not None:
            frames.append(screen.copy())

    if verbose:
        print("\n" + "=" * 70)
        print(" DEATHMATCH Episode Started")
        print("=" * 70)
        print(f"\nInitial observation (LLM sees this):\n{obs_str}")

    total_reward = 0.0

    for step in range(max_steps):
        action_str = agent.select_action(obs_str)
        next_obs_str, reward, done, truncated, next_info = env.step(action_str)
        total_reward += reward

        if collect_frames and env._last_obs is not None:
            screen = env._last_obs.get('screen')
            if screen is not None:
                frames.append(screen.copy())

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

        if verbose and step < 3:
            print(f"\n{'─' * 60}")
            print(f" Step {step}: Action={action_str}, Reward={reward:+.3f}")
            print(f"{'─' * 60}")
            print(next_obs_str)
        elif verbose and step == 3:
            print(f"\n... (showing summary every 100 steps) ...\n")
        elif verbose and step % 100 == 0 and step > 3:
            gv = next_info.get('game_variables', {})
            print(f"  Step {step:4d} | R={reward:+.3f} | Total={total_reward:+.2f} | "
                  f"HP={gv.get('HEALTH', 'N/A'):.0f} | "
                  f"Frags={gv.get('FRAGCOUNT', 0)} | Deaths={gv.get('DEATHCOUNT', 0)} | "
                  f"Pos=({gv.get('POSITION_X', 0):.0f}, {gv.get('POSITION_Y', 0):.0f})")

        if done or truncated:
            if verbose:
                print(f"\n  Episode ended at step {step} (done={done}, truncated={truncated})")
            break

        obs_str = next_obs_str

    if verbose:
        print(f"\n  Total reward: {total_reward:+.3f}")
        print(f"  Total steps:  {len(history)}")

    return history, frames


def print_metrics(metrics: Dict[str, Any], title: str = "Exploration Metrics"):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    groups = {
        "基础指标 (Basic)": [
            "total_steps", "episode_length", "total_reward",
        ],
        "覆盖率 (Coverage)": [
            "visited_positions", "map_coverage", "room_coverage",
            "quadrant_coverage", "exploration_speed",
        ],
        "多样性 (Diversity / Entropy)": [
            "unique_states", "state_entropy",
            "unique_actions_used", "action_entropy",
            "state_action_entropy",
        ],
        "新颖性 (Novelty)": [
            "avg_visual_novelty", "max_visual_novelty", "cumulative_novelty",
            "novelty_half_life", "novelty_decay_rate", "avg_position_novelty",
        ],
        "战斗表现 (Combat)": [
            "frags", "deaths", "kd_ratio",
            "items_collected", "attack_count", "accuracy_estimate",
            "damage_taken", "survival_time", "avg_survival_per_life",
        ],
    }

    for group_name, keys in groups.items():
        present = [k for k in keys if k in metrics]
        if not present:
            continue
        print(f"\n  {group_name}:")
        for key in present:
            val = metrics[key]
            if isinstance(val, float):
                print(f"    {key:30s}: {val:.6f}")
            else:
                print(f"    {key:30s}: {val}")


def main():
    print("=" * 70)
    print(" VizDoom DEATHMATCH Demo")
    print(" The Hardest VizDoom Scenario — Full Combat + Exploration Metrics")
    print("=" * 70)

    SEED = 42
    MAX_STEPS = 500
    NUM_EPISODES = 3

    # Print action space
    print("\n" + "─" * 70)
    print(" Action Space:")
    print("─" * 70)
    print(get_action_space())

    env = DeathmatchTask(seed=SEED)
    agent = RandomAgent(seed=SEED)

    # Metrics calculator
    metrics_calc = DeathmatchMetrics()

    all_metrics = []
    all_frames = []

    for ep in range(NUM_EPISODES):
        print(f"\n{'=' * 70}")
        print(f" Episode {ep + 1}/{NUM_EPISODES}")
        print(f"{'=' * 70}")

        collect = (ep == NUM_EPISODES - 1)
        history, frames = run_episode(
            env, agent, max_steps=MAX_STEPS, verbose=True, collect_frames=collect,
        )
        if collect:
            all_frames = frames

        metrics = metrics_calc.compute_all(history)
        all_metrics.append(metrics)

        print_metrics(metrics, f"Episode {ep + 1} Exploration Metrics")

    # ========== Aggregate ==========
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print(" Aggregate Statistics (mean ± std)")
        print("=" * 70)

        numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for key in sorted(numeric_keys):
            vals = [m[key] for m in all_metrics if m.get(key) is not None]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                print(f"  {key:35s}: {mean_v:>12.4f} ± {std_v:.4f}")

    # ========== Save Outputs ==========
    output_dir = "./deathmatch_trajectories"
    print(f"\n{'=' * 70}")
    print(f" Saving Outputs → {output_dir}")
    print(f"{'=' * 70}")

    # GIF
    if all_frames:
        ts = time.strftime("%Y%m%d_%H%M%S")
        step_size = max(1, len(all_frames) // 120)
        sampled = all_frames[::step_size]
        gif_path = os.path.join(output_dir, f"deathmatch_replay_{ts}.gif")
        save_frames_as_gif(sampled, gif_path, duration=80, scale=2)

        # Key frames
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        indices = [0, len(all_frames) // 4, len(all_frames) // 2,
                   3 * len(all_frames) // 4, len(all_frames) - 1]
        for idx in indices:
            if 0 <= idx < len(all_frames):
                img = Image.fromarray(all_frames[idx].astype(np.uint8))
                img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
                img.save(os.path.join(frames_dir, f"frame_{idx:04d}.png"))
        print(f"  Key frames saved to: {frames_dir}/")

    # Trajectory (Parquet)
    if history:
        save_path = Observer.save_trajectory(
            history,
            history_dir=output_dir,
            metadata={
                "scenario": "deathmatch",
                "agent": "RandomAgent",
                "seed": SEED,
                "max_steps": MAX_STEPS,
                "num_episodes": NUM_EPISODES,
            },
        )
        if save_path:
            print(f"  Trajectory saved: {save_path}")

    # ========== Scene Description Demo ==========
    print(f"\n{'=' * 70}")
    print(" Scene Description Demo (standalone)")
    print(f"{'=' * 70}")

    if env._last_obs is not None and env._last_info is not None:
        scene = Observer.describe_scene(env._last_obs, env._last_info)
        print(f"\n{scene}")
    else:
        print("  No observation available.")

    env.close()
    print("\nDeathmatch demo completed!")


if __name__ == "__main__":
    main()
