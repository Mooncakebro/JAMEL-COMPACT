"""
Demo 命令
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from jamel.core.models.openai_model import OpenAIModel
from jamel.web.automation.web_agent import WebAgent
import structlog


async def run_web_demo(url: str, goal: str):
    """运行 Web 演示"""
    print(f"🌐 开始 Web 自动化演示")
    print(f"   URL: {url}")
    print(f"   目标: {goal}\n")

    # 创建模型
    model = OpenAIModel("DeepSeek-V3.2")  # 或其他模型

    # 创建 Agent
    agent = WebAgent(model, headless=False, max_steps=10)

    # 运行任务
    result = await agent.run(url, goal)

    # 输出结果
    if result.get("success"):
        print(f"\n✅ 演示成功: {result.get('message')}")
        print(f"   完成步数: {result.get('steps_completed')}")
    else:
        print(f"\n❌ 演示失败: {result.get('error')}")
        print(f"   完成步数: {result.get('steps_completed')}")


def run_rl_demo():
    """运行 RL 演示"""
    print("🎮 开始强化学习演示")

    try:
        import gymnasium as gym
        import torch
        import torch.nn as nn
        import random
        import numpy as np

        env = gym.make("CartPole-v1")
        state, _ = env.reset()

        print(f"环境: {env.spec.name}")
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")

        total_reward = 0
        for step in range(100):
            # 随机动作
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"回合结束: 步数={step}, 总奖励={total_reward}")
                break

        env.close()

    except ImportError:
        print("❌ 缺少依赖，请安装: pip install gymnasium torch")


async def main(args):
    """Demo 主函数"""
    if args.type == "web":
        await run_web_demo(args.url, args.goal)
    elif args.type == "rl":
        run_rl_demo()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["web", "rl"], default="web")
    parser.add_argument("--url", default="https://news.ycombinator.com/")
    parser.add_argument("--goal", default="浏览热门新闻")
    args = parser.parse_args()

    if args.type == "web":
        asyncio.run(main(args))
    else:
        asyncio.run(main(args))