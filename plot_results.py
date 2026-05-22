"""
可视化脚本: 复现论文 Fig. 2 (阻塞概率 vs 授权带宽)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_blocking_vs_bandwidth(results_path="results/sweep_results.npz"):
    """
    绘制阻塞概率 vs. 授权带宽 (论文 Fig. 2).
    """
    if not os.path.exists(results_path):
        print(f"结果文件 {results_path} 不存在, 请先运行: python train.py --sweep")
        return

    data = np.load(results_path, allow_pickle=True)

    bw = data["bw"]
    ddqn = data["ddqn_blocking"]
    thresh = data["threshold_blocking"]
    rand = data["random_blocking"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bw, ddqn, "o-", color="tab:blue", linewidth=2,
            markersize=8, label="Proposed DDQN")
    ax.plot(bw, thresh, "s--", color="tab:orange", linewidth=2,
            markersize=8, label="Threshold-based [15]")
    ax.plot(bw, rand, "^:", color="tab:green", linewidth=2,
            markersize=8, label="Random Selection")

    ax.set_xlabel("Bandwidth of Licensed Band (Mbps)", fontsize=12)
    ax.set_ylabel("Blocking Probability", fontsize=12)
    ax.set_title("Blocking Probability vs. Licensed Bandwidth ($B_U$ = 100 Mbps)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/fig2_blocking_vs_bw.png", dpi=150)
    print("Fig. 2 已保存至 results/fig2_blocking_vs_bw.png")
    plt.show()


def plot_training_curves(history):
    """
    绘制训练曲线: 奖励、阻塞率、ε值、损失.
    history: train_ddqn 返回的 history dict (或从 .npz 加载).
    """
    if isinstance(history, str):
        history = dict(np.load(history, allow_pickle=True))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 平滑函数
    def smooth(y, window=50):
        if len(y) < window:
            return y
        kernel = np.ones(window) / window
        return np.convolve(y, kernel, mode="valid")

    # (a) Episode 奖励
    ax = axes[0, 0]
    rewards = history.get("episode_rewards", [])
    ax.plot(rewards, alpha=0.3, color="tab:blue", linewidth=0.5)
    if len(rewards) > 50:
        ax.plot(smooth(rewards), color="tab:blue", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Episode 阻塞率
    ax = axes[0, 1]
    blocking = history.get("episode_blocking", [])
    ax.plot(blocking, alpha=0.3, color="tab:red", linewidth=0.5)
    if len(blocking) > 50:
        ax.plot(smooth(blocking), color="tab:red", linewidth=2, label="Smoothed")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Blocking Probability")
    ax.set_title("Blocking Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) ε 衰减
    ax = axes[1, 0]
    eps = history.get("epsilon", [])
    ax.plot(eps, color="tab:green", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.set_title("Exploration Rate (ε)")
    ax.grid(True, alpha=0.3)

    # (d) Loss
    ax = axes[1, 1]
    loss = history.get("loss", [])
    if len(loss) > 0:
        ax.plot(loss, alpha=0.3, color="tab:purple", linewidth=0.3)
        if len(loss) > 500:
            ax.plot(smooth(loss, 200), color="tab:purple", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Bellman Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/training_curves.png", dpi=150)
    print("训练曲线已保存至 results/training_curves.png")
    plt.show()


def plot_blocking_reduction_bar(results_path="results/sweep_results.npz"):
    """
    柱状图: DDQN相对基线的阻塞率降低百分比.
    """
    if not os.path.exists(results_path):
        print(f"结果文件 {results_path} 不存在.")
        return

    data = np.load(results_path, allow_pickle=True)
    bw = data["bw"]
    ddqn = data["ddqn_blocking"]
    thresh = data["threshold_blocking"]
    rand = data["random_blocking"]

    # 计算降低百分比 (取最差带宽点)
    idx_min = np.argmin(bw)  # 500 Mbps 处 (最受限)

    reduction_vs_threshold = (1 - ddqn[idx_min] / (thresh[idx_min] + 1e-10)) * 100
    reduction_vs_random = (1 - ddqn[idx_min] / (rand[idx_min] + 1e-10)) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["vs Threshold", "vs Random"],
                  [reduction_vs_threshold, reduction_vs_random],
                  color=["tab:orange", "tab:green"], alpha=0.7, edgecolor="black")
    ax.set_ylabel("Blocking Reduction (%)")
    ax.set_title(f"DDQN Blocking Reduction at {bw[idx_min]:.0f} Mbps Licensed BW")
    ax.axhline(y=87.5, color="tab:red", linestyle="--", linewidth=1, alpha=0.7,
               label="Paper: 87.5% (vs Threshold)")
    ax.axhline(y=82.5, color="tab:purple", linestyle="--", linewidth=1, alpha=0.7,
               label="Paper: 82.5% (vs Random)")

    for bar, val in zip(bars, [reduction_vs_threshold, reduction_vs_random]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig("results/blocking_reduction.png", dpi=150)
    print("阻塞率降低对比已保存至 results/blocking_reduction.png")
    plt.show()


if __name__ == "__main__":
    plot_blocking_vs_bandwidth()
