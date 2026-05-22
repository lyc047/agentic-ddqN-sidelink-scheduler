"""
基线调度策略: 阈值策略 与 随机策略
基于 Chou et al., arXiv:2509.06775, 2025.
"""

import numpy as np


def threshold_policy(env, wifi_idle_threshold=0.5):
    """
    阈值策略 (论文 baseline 2, 参考 [15]).

    规则:
      1. Wi-Fi空闲概率 > 50% 时允许SL-U, 否则仅授权频段
      2. 队列占用率 > 50% → 优先高速CC模式排空队列
      3. 队列占用率 ≤ 50% → 可用SL-U卸载到非授权频段
    """
    wifi_idle = env.wifi.is_idle()
    q_ratio = len(env.queue) / max(env.queue_max, 1)

    if wifi_idle:
        if q_ratio > 0.5:
            return 0  # CC-28G, 高速排空队列
        else:
            return 4  # SL-U-5G, 卸载到非授权频段
    else:
        # Wi-Fi忙: 仅授权频段, 选CC-28G
        return 0


def random_policy(env):
    """随机策略 (论文 baseline 3): 等概率选择5种模式."""
    return np.random.randint(0, 5)


def evaluate_policy(env, policy_fn, num_episodes=100, steps_per_ep=200):
    """
    评估给定策略.
    返回: (avg_blocking_rate, avg_throughput)
    """
    all_blocking = []
    all_throughput = []

    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(steps_per_ep):
            if callable(policy_fn):
                action = policy_fn(env)
            else:
                action = policy_fn
            _, _, _, _ = env.step(action)
        all_blocking.append(env.get_blocking_rate())
        all_throughput.append(env.get_throughput())

    return np.mean(all_blocking), np.mean(all_throughput)
