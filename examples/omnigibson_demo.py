#!/usr/bin/env python3
"""
OmniGibson 环境演示脚本

演示：
1. 实例化 OmniGibson 环境（无 Isaac Sim 时使用 Mock）
2. Random Agent 随机选择文本动作
3. 收集轨迹并保存 Parquet
4. 计算探索评测指标：覆盖率、状态熵、新颖性、交互多样性
"""

import os
import random
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

from jamel.core.env.omni_gibson import (
    OmniGibsonTask,
    make_omnigibson_env,
    Observer,
    get_action_space,
    get_available_actions,
    StepHistory,
    OmniGibsonMetrics,
    compute_metrics,
)


# 仅无参动作，便于 Random Agent 直接随机选择
BASE_ACTION_NAMES = [
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right",
    "arm_up", "arm_down", "arm_forward", "arm_back",
    "grasp", "release",
    "noop",
]


class RandomAgent:
    """随机智能体 - 在 OmniGibson 动作空间中随机选择（无参动作 + 偶尔带参）"""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.base_actions = BASE_ACTION_NAMES
        # 带参动作示例（随机时可选）
        self.param_actions = [
            'interact("cabinet_0")', 'interact("table_0")', 'interact("door_0")',
            'navigate_to("kitchen")', 'navigate_to("living_room")',
        ]
        print("RandomAgent initialized for OmniGibson")
        print(f"  Base actions: {len(self.base_actions)}")
        print(f"  Param actions (sample): {self.param_actions[:2]}...")

    def select_action(self, observation: str) -> str:
        if self.rng.random() < 0.85:
            name = self.rng.choice(self.base_actions)
            return f"{name}()"
        return self.rng.choice(self.param_actions)

    def reset(self):
        pass


def run_episode(
    env: OmniGibsonTask,
    agent: RandomAgent,
    max_steps: int = 200,
    verbose: bool = True,
) -> List[StepHistory]:
    history: List[StepHistory] = []
    obs_str, info = env.reset()
    agent.reset()

    if verbose:
        print("\n" + "=" * 70)
        print(" OmniGibson Episode Started")
        print("=" * 70)
        print(f"\nInitial observation (LLM sees this):\n{obs_str[:800]}...")

    total_reward = 0.0

    for step in range(max_steps):
        action_str = agent.select_action(obs_str)
        next_obs_str, reward, done, truncated, next_info = env.step(action_str)
        total_reward += reward

        action_dict = next_info.get("_action_dict")
        if action_dict is None:
            action_dict = {"action_name": "unknown", "func_name": "unknown", "args": {}}
        if "func_name" not in action_dict:
            action_dict["func_name"] = action_dict.get("action_name", "unknown")

        step_record = StepHistory(
            step=step,
            obs=env._last_obs,
            info=next_info,
            observation_str=obs_str,
            llm_completion="",
            memory_content=None,
            action=action_dict,
            result={"reward": reward, "done": done, "truncated": truncated},
            timestamp=datetime.now(),
        )
        history.append(step_record)

        if verbose and step < 3:
            print(f"\n{'─' * 60}")
            print(f" Step {step}: Action={action_str}, Reward={reward:+.3f}")
            print(f"{'─' * 60}")
            print(next_obs_str[:500] + "...")
        elif verbose and step == 3:
            print("\n... (showing summary every 50 steps) ...\n")
        elif verbose and step % 50 == 0 and step > 3:
            rp = next_info.get("robot_position", [0, 0, 0])
            room = next_info.get("room_id", "?")
            print(f"  Step {step:4d} | R={reward:+.3f} | Total={total_reward:+.2f} | "
                  f"Pos=({rp[0]:.2f}, {rp[1]:.2f}) | Room={room}")

        if done or truncated:
            if verbose:
                print(f"\n  Episode ended at step {step} (done={done}, truncated={truncated})")
            break

        obs_str = next_obs_str

    if verbose:
        print(f"\n  Total reward: {total_reward:+.3f}")
        print(f"  Total steps:  {len(history)}")

    return history


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
            "unique_rooms_visited", "exploration_speed",
        ],
        "多样性 (Diversity / Entropy)": [
            "unique_states", "state_entropy",
            "unique_actions_used", "action_entropy",
        ],
        "新颖性 (Novelty)": [
            "avg_visual_novelty", "max_visual_novelty", "cumulative_novelty",
            "novelty_decay_rate", "avg_position_novelty",
        ],
        "交互 (Interaction)": [
            "unique_objects_interacted", "interaction_count",
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
    print(" OmniGibson Embodied Environment Demo")
    print(" Mock env + Random Agent + Exploration Metrics")
    print("=" * 70)

    SEED = 42
    MAX_STEPS = 200
    NUM_EPISODES = 2

    print("\n" + "─" * 70)
    print(" Action Space:")
    print("─" * 70)
    print(get_action_space())

    env = make_omnigibson_env(seed=SEED, max_steps=MAX_STEPS)
    agent = RandomAgent(seed=SEED)
    metrics_calc = OmniGibsonMetrics(room_ids=["living_room", "kitchen", "bedroom", "bathroom", "office"])

    all_metrics = []

    for ep in range(NUM_EPISODES):
        print(f"\n{'=' * 70}")
        print(f" Episode {ep + 1}/{NUM_EPISODES}")
        print(f"{'=' * 70}")

        history = run_episode(env, agent, max_steps=MAX_STEPS, verbose=True)
        metrics = metrics_calc.compute_all(history)
        all_metrics.append(metrics)
        print_metrics(metrics, f"Episode {ep + 1} Exploration Metrics")

        # 保存最后一轮的轨迹为 Parquet
        if ep == NUM_EPISODES - 1 and history:
            out_dir = os.path.join(os.path.dirname(__file__), "..", "omnigibson_trajectories")
            path = Observer.save_trajectory(history, out_dir, metadata={"seed": SEED, "episode": ep + 1})
            if path:
                print(f"\n  Trajectory saved: {path}")

    # 使用 Observer.calculate_metrics 再算一次（与 metrics 模块一致）
    if all_metrics:
        print("\n" + "=" * 70)
        print(" Summary (Episode 1 metrics via Observer.calculate_metrics)")
        print("=" * 70)
        first_history = run_episode(env, agent, max_steps=50, verbose=False)
        if first_history:
            obs_metrics = Observer.calculate_metrics(
                first_history,
                map_bounds=(-10.0, -10.0, 10.0, 10.0),
                grid_resolution=0.5,
                room_ids=["living_room", "kitchen", "bedroom", "bathroom", "office"],
            )
            print_metrics(obs_metrics, "Observer.calculate_metrics (short run)")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
