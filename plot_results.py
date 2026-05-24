"""
可视化脚本 v2.0: 复现论文 Fig. 2 + 训练诊断 + CMDP β收敛
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def load_results(path="results/sweep_results_v2.npz"):
    """加载扫描结果，若v2不存在则回退v1."""
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    fallback = "results/sweep_results.npz"
    if os.path.exists(fallback):
        print(f"未找到 {path}, 使用 {fallback}")
        return dict(np.load(fallback, allow_pickle=True))
    print(f"结果文件不存在, 请先运行: python train.py --sweep")
    return None


def plot_blocking_vs_bandwidth(results_path="results/sweep_results_v2.npz"):
    """绘制阻塞概率 vs. 授权带宽 (论文 Fig. 2)."""
    data = load_results(results_path)
    if data is None:
        return

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
    绘制训练曲线: 奖励、阻塞率、β收敛、ε值、损失 (5面板).
    history: train_ddqn 返回的 dict (或 .npz 路径).
    """
    if isinstance(history, str) and os.path.exists(history):
        history = dict(np.load(history, allow_pickle=True))
    elif isinstance(history, str):
        print(f"文件 {history} 不存在")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

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
    ax.set_ylabel("Avg Reward/UE")
    ax.set_title("Training Reward")
    ax.grid(True, alpha=0.3)

    # (b) Episode 阻塞率
    ax = axes[0, 1]
    blocking = history.get("episode_blocking", [])
    ax.plot(blocking, alpha=0.3, color="tab:red", linewidth=0.5)
    if len(blocking) > 50:
        ax.plot(smooth(blocking), color="tab:red", linewidth=2, label="Smoothed")
    # 目标阻塞率参考线
    if blocking:
        ax.axhline(y=0.02, color="gray", linestyle="--", alpha=0.5, label="Target (2%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Blocking Probability")
    ax.set_title("Blocking Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) CMDP β 收敛
    ax = axes[0, 2]
    beta_hist = history.get("beta", [])
    if len(beta_hist) > 0:
        ax.plot(beta_hist, color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("β (Lagrange Multiplier)")
    ax.set_title("CMDP Dual Variable")
    ax.grid(True, alpha=0.3)

    # (d) 温度衰减
    ax = axes[1, 0]
    temps = history.get("temperature", history.get("epsilon", []))
    ax.plot(temps, color="tab:green", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("τ")
    ax.set_title("Temperature")
    ax.grid(True, alpha=0.3)

    # (e) Loss
    ax = axes[1, 1]
    loss = history.get("loss", [])
    if len(loss) > 0:
        ax.plot(loss, alpha=0.2, color="tab:purple", linewidth=0.2)
        if len(loss) > 500:
            ax.plot(smooth(loss, 200), color="tab:purple", linewidth=1.5, label="Smoothed")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Bellman Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # (f) 吞吐量 vs 阻塞率散点 (帕累托前沿)
    ax = axes[1, 2]
    tput = history.get("episode_throughput", [])
    if len(tput) > 0 and len(blocking) > 0:
        n_pts = min(len(tput), len(blocking))
        colors = np.arange(n_pts)
        scatter = ax.scatter(blocking[:n_pts], tput[:n_pts],
                             c=colors, cmap="viridis", alpha=0.5, s=10)
        ax.set_xlabel("Blocking Rate")
        ax.set_ylabel("Throughput (Mbps)")
        ax.set_title("Throughput vs Blocking")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Episode")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Throughput vs Blocking")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/training_curves.png", dpi=150)
    print("训练曲线已保存至 results/training_curves.png")
    plt.show()


def plot_blocking_reduction_bar(results_path="results/sweep_results_v2.npz"):
    """柱状图: DDQN在每个BW点相对基线的阻塞率降低百分比."""
    data = load_results(results_path)
    if data is None:
        return

    bw = np.array(data["bw"])
    ddqn = np.array(data["ddqn_blocking"])
    thresh = np.array(data["threshold_blocking"])
    rand = np.array(data["random_blocking"])

    x = np.arange(len(bw))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))

    red_vs_thresh = np.clip((1 - ddqn / (thresh + 1e-10)) * 100, -100, 100)
    red_vs_rand = np.clip((1 - ddqn / (rand + 1e-10)) * 100, -100, 100)

    bars1 = ax.bar(x - width / 2, red_vs_thresh, width, color="tab:orange",
                   alpha=0.7, edgecolor="black", label="vs Threshold")
    bars2 = ax.bar(x + width / 2, red_vs_rand, width, color="tab:green",
                   alpha=0.7, edgecolor="black", label="vs Random")

    for bar, val in zip(bars1, red_vs_thresh):
        y_pos = bar.get_height() if val >= 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for bar, val in zip(bars2, red_vs_rand):
        y_pos = bar.get_height() if val >= 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.0f}" for b in bw])
    ax.set_xlabel("Licensed Bandwidth (Mbps)")
    ax.set_ylabel("Blocking Reduction (%)")
    ax.set_title("DDQN Blocking Reduction vs Baselines (v2.0 Multi-UE)")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig("results/blocking_reduction.png", dpi=150)
    print("阻塞率降低对比已保存至 results/blocking_reduction.png")
    plt.show()


if __name__ == "__main__":
    plot_blocking_vs_bandwidth()
