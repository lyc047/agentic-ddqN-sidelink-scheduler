"""
训练主脚本: Agentic DDQN Sidelink 调度器
基于 Chou et al., arXiv:2509.06775, 2025.

用法:
  python train.py                    # 单次训练
  python train.py --sweep            # 扫描不同授权带宽
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

import config as cfg
from environment import SidelinkEnv
from ddqn_agent import DDQNAgent
from baselines import threshold_policy, random_policy, evaluate_policy


def train_ddqn(licensed_bw=500.0, arrival_rate=0.5, dynamic_channel=True,
               num_episodes=None, seed=42):
    """
    训练 DDQN 调度器并返回训练历史.

    返回: dict with keys:
      - episode_rewards, episode_blocking, episode_throughput
      - epsilon_history, loss_history
    """
    if num_episodes is None:
        num_episodes = cfg.NUM_EPISODES

    env = SidelinkEnv(
        licensed_bw_total=licensed_bw,
        arrival_rate=arrival_rate,
        seed=seed,
        dynamic_channel=dynamic_channel,
    )

    agent = DDQNAgent(cfg.STATE_DIM, cfg.ACTION_DIM, cfg)

    history = defaultdict(list)

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        for step in range(cfg.MAX_STEPS_PER_EP):
            # 动作选择
            action = agent.select_action(state, training=True)

            # 环境交互
            next_state, reward, done, info = env.step(action)

            # 存储经验并更新
            loss = agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward

            if loss is not None:
                history["loss"].append(loss)

            if done:
                break

        # ---- 记录 episode 统计 ----
        history["episode_rewards"].append(ep_reward)
        history["episode_blocking"].append(env.get_blocking_rate())
        history["episode_throughput"].append(env.get_throughput())
        history["epsilon"].append(agent.epsilon)

        # 学习率衰减 (线性插值)
        progress = ep / num_episodes
        agent.update_lr(progress)

        # 日志 (过夜版每10%输出一次)
        log_interval = max(1, num_episodes // 10)
        if (ep + 1) % log_interval == 0:
            recent_block = np.mean(history["episode_blocking"][-100:])
            recent_reward = np.mean(history["episode_rewards"][-100:])
            print(f"[Ep {ep+1:5d}/{num_episodes}] "
                  f"ε={agent.epsilon:.3f}  "
                  f"blocking(100avg)={recent_block:.4f}  "
                  f"reward(100avg)={recent_reward:.2f}  "
                  f"lr={agent.optimizer.param_groups[0]['lr']:.2e}")

    return history, agent, env


def run_sweep(bw_values=None, arrival_rate=0.5, num_episodes=None):
    """
    扫描不同授权带宽值, 对比 DDQN vs 基线.
    复现论文 Fig. 2.
    """
    if bw_values is None:
        bw_values = np.arange(
            cfg.LICENSED_BW_MIN_MBPS,
            cfg.LICENSED_BW_MAX_MBPS + 1,
            cfg.LICENSED_BW_STEP_MBPS,
        )

    results = {"bw": [], "ddqn_blocking": [], "threshold_blocking": [],
               "random_blocking": [], "ddqn_tput": [], "threshold_tput": [],
               "random_tput": []}

    for bw in bw_values:
        print(f"\n{'='*60}")
        print(f"扫描: 授权带宽 = {bw:.0f} Mbps")
        print(f"{'='*60}")

        # ---- DDQN ----
        print(f"\n--- 训练 DDQN (bw={bw:.0f}) ---")
        history, agent, train_env = train_ddqn(
            licensed_bw=bw, arrival_rate=arrival_rate,
            num_episodes=num_episodes, seed=int(bw),
        )

        # 在独立环境上评估训练后的DDQN
        eval_env = SidelinkEnv(
            licensed_bw_total=bw, arrival_rate=arrival_rate,
            seed=int(bw) + 1000, dynamic_channel=True,
        )
        ddqn_block, ddqn_tput = evaluate_policy(
            eval_env, lambda e: agent.select_action(e._get_state(), training=False),
            num_episodes=50,
        )

        # ---- 基线 ----
        base_env_t = SidelinkEnv(
            licensed_bw_total=bw, arrival_rate=arrival_rate,
            seed=int(bw) + 2000, dynamic_channel=True,
        )
        thresh_block, thresh_tput = evaluate_policy(
            base_env_t, threshold_policy, num_episodes=50,
        )

        base_env_r = SidelinkEnv(
            licensed_bw_total=bw, arrival_rate=arrival_rate,
            seed=int(bw) + 3000, dynamic_channel=True,
        )
        rand_block, rand_tput = evaluate_policy(
            base_env_r, random_policy, num_episodes=50,
        )

        # ---- 记录 ----
        results["bw"].append(bw)
        results["ddqn_blocking"].append(ddqn_block)
        results["threshold_blocking"].append(thresh_block)
        results["random_blocking"].append(rand_block)
        results["ddqn_tput"].append(ddqn_tput)
        results["threshold_tput"].append(thresh_tput)
        results["random_tput"].append(rand_tput)

        print(f"\n  bw={bw:.0f}: DDQN blocking={ddqn_block:.4f}, "
              f"threshold={thresh_block:.4f}, random={rand_block:.4f}")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    np.savez("results/sweep_results.npz", **results)
    print("\n结果已保存至 results/sweep_results.npz")

    return results


def main():
    parser = argparse.ArgumentParser(description="Agentic DDQN Scheduler")
    parser.add_argument("--sweep", action="store_true",
                        help="扫描不同授权带宽")
    parser.add_argument("--bw", type=float, default=500.0,
                        help="授权带宽 (Mbps), 用于单次训练")
    parser.add_argument("--episodes", type=int, default=None,
                        help="训练 episode 数 (覆盖config)")
    parser.add_argument("--arrival", type=float, default=0.5,
                        help="Poisson 到达率")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(arrival_rate=args.arrival, num_episodes=args.episodes)
    else:
        print(f"单次训练: 授权带宽={args.bw:.0f} Mbps")
        history, agent, env = train_ddqn(
            licensed_bw=args.bw,
            arrival_rate=args.arrival,
            num_episodes=args.episodes,
        )
        final_block = np.mean(history["episode_blocking"][-100:])
        print(f"\n最终100ep平均阻塞率: {final_block:.4f}")


if __name__ == "__main__":
    main()
