"""
基线调度策略 v2.0: 阈值策略 与 随机策略 (多UE版)
基于 Chou et al., arXiv:2509.06775, 2025.

v2.0: policy_fn 签名为 (env, ue_id) → action, 每个UE独立决策
"""

import numpy as np
from tqdm import tqdm


def threshold_policy(env, ue_id):
    """
    阈值策略 (论文 baseline 2, 参考 [15]).

    每个UE独立判断:
      1. Wi-Fi空闲 + 队列>50% → CC-28G 高速排空
      2. Wi-Fi空闲 + 队列≤50% → SL-U-5G 卸载
      3. Wi-Fi忙 → CC-28G (仅授权频段)
    """
    wifi_idle = env.wifi.is_idle()
    queue = env.queues[ue_id]
    q_ratio = len(queue) / max(env.queue_max, 1)

    if wifi_idle:
        if q_ratio > 0.5:
            return 0  # CC-28G
        else:
            return 4  # SL-U-5G
    else:
        return 0  # CC-28G


def random_policy(env, ue_id):
    """随机策略 (论文 baseline 3): 等概率选择5种模式."""
    return np.random.randint(0, 5)


def make_ddqn_policy(agent):
    """将训练好的DDQN Agent包装为 (env, ue_id) → action 策略函数."""
    def policy(env, ue_id):
        state = env._get_state(ue_id)
        return agent.select_action(state, training=False)
    return policy


def evaluate_policy(env, policy_fn, num_episodes=50, steps_per_ep=100):
    """
    评估给定策略 (多UE版).

    参数:
      env: SidelinkEnv 实例 (评估前应已构造好, 含num_ue)
      policy_fn: (env, ue_id) → action
      num_episodes: 评估episode数
      steps_per_ep: 每episode步数

    返回: (avg_blocking_rate, avg_throughput_mbps)
    """
    blocking_rates = []
    throughputs = []

    for _ in tqdm(range(num_episodes), desc="  评估中", unit="ep",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        env.reset()
        for _ in range(steps_per_ep):
            actions = [policy_fn(env, i) for i in range(env.num_ue)]
            env.step(actions)
        blocking_rates.append(env.get_blocking_rate())
        throughputs.append(env.get_throughput())

    return np.mean(blocking_rates), np.mean(throughputs)
