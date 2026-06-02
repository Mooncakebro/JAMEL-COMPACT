"""
Train 命令
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import structlog

from jamel.log import log_utils
logger = log_utils.get_logger(__name__)



class SimplePolicy(nn.Module):
    """简单的策略网络"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_rl(env_name: str, episodes: int = 1000):
    """训练 RL 智能体"""
    print(f"🎮 开始强化学习训练")
    print(f"   环境: {env_name}")
    print(f"   回合数: {episodes}\n")

    # 创建环境
    env = gym.make(env_name)

    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    logger.info("环境信息",
               state_dim=state_dim,
               action_dim=action_dim)

    # 创建策略网络
    policy = SimplePolicy(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    # 训练循环
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 将状态转换为 tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 获取动作概率
            action_probs = torch.softmax(policy(state_tensor), dim=-1)

            # 采样动作
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += reward

            done = terminated or truncated

            # 计算损失（简化版 REINFORCE）
            if done:
                # 计算优势
                advantage = reward - np.mean(rewards[-10:] if rewards else 0)

                # 计算策略损失
                log_prob = torch.log(action_probs[0, action.item()])
                loss = -log_prob * advantage

                # 更新网络
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                state = next_state

        rewards.append(episode_reward)

        # 每100回合输出一次
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"回合 {episode+1:4d} | 平均奖励: {avg_reward:6.2f}")

            logger.info("训练进度",
                      episode=episode+1,
                      avg_reward=avg_reward)

    env.close()

    # 输出最终结果
    final_avg = np.mean(rewards[-100:])
    print(f"\n✅ 训练完成")
    print(f"   最终平均奖励: {final_avg:.2f}")
    print(f"   总回合数: {episodes}")


def main(args):
    """训练主函数"""
    try:
        train_rl(args.env, args.episodes)
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install gymnasium torch")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        logger.error("训练错误", error=str(e))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", help="环境名称")
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    args = parser.parse_args()
    main(args)