"""
训练主脚本 v2.0: Agentic DDQN Sidelink 调度器 (多UE竞争 + CMDP)
基于 Chou et al., arXiv:2509.06775, 2025.

用法:
  python train.py                         # 单次训练 (bw=500)
  python train.py --sweep                 # 扫描6个带宽 (串行, ~23h)
  python train.py --sweep --parallel 3    # 扫描6个带宽 (3并行, ~8h)
  python train.py --sweep --parallel 6    # 扫描6个带宽 (6并行, ~4h)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

import config as cfg
from environment import SidelinkEnv
from ddqn_agent import DDQNAgent
from baselines import threshold_policy, random_policy, evaluate_policy, make_ddqn_policy


def train_ddqn(licensed_bw=500.0, arrival_rate=1.5, dynamic_channel=True,
               num_episodes=None, seed=42):
    """训练 DDQN 调度器 (多UE竞争版). 返回 (history, agent, env)."""
    if num_episodes is None:
        num_episodes = cfg.NUM_EPISODES

    env = SidelinkEnv(
        num_ue=cfg.NUM_UE,
        licensed_bw_total=licensed_bw,
        arrival_rate=arrival_rate,
        seed=seed,
        dynamic_channel=dynamic_channel,
    )

    agent = DDQNAgent(cfg.STATE_DIM, cfg.ACTION_DIM, cfg)
    history = defaultdict(list)

    tag = f"[bw={licensed_bw:.0f}] " if cfg.NUM_UE > 1 else ""
    pbar = tqdm(total=num_episodes, desc=f"{tag}训练进度", unit="ep",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for ep in range(num_episodes):
        states = env.reset()
        ep_rewards = np.zeros(env.num_ue)

        for step in range(cfg.MAX_STEPS_PER_EP):
            actions = [agent.select_action(s, training=True) for s in states]
            next_states, rewards, done, infos = env.step(actions)

            for i in range(env.num_ue):
                loss = agent.update(states[i], actions[i], rewards[i],
                                    next_states[i], done)
                if loss is not None:
                    history["loss"].append(loss)

            states = next_states
            ep_rewards += np.array(rewards)

            if done:
                break

        avg_reward = ep_rewards.mean()
        blocking = env.get_blocking_rate()
        throughput = env.get_throughput()

        history["episode_rewards"].append(avg_reward)
        history["episode_blocking"].append(blocking)
        history["episode_throughput"].append(throughput)
        history["temperature"].append(agent.temperature)
        history["beta"].append(agent.beta)

        agent.update_beta(blocking, ep)
        env.beta = agent.beta

        progress = ep / num_episodes
        agent.update_lr(progress)

        log_interval = max(1, num_episodes // 10)
        if (ep + 1) % log_interval == 0:
            recent_block = np.mean(history["episode_blocking"][-100:])
            recent_reward = np.mean(history["episode_rewards"][-100:])
            avg_qlen = env.get_avg_queue_length()
            pbar.write(f"{tag}[Ep {ep+1:5d}/{num_episodes}] "
                       f"τ={agent.temperature:.3f}  β={agent.beta:.2f}  "
                       f"block(100avg)={recent_block:.4f}  "
                       f"rew(100avg)={recent_reward:.2f}  "
                       f"qlen={avg_qlen:.1f}  "
                       f"lr={agent.optimizer.param_groups[0]['lr']:.2e}")

        pbar.update(1)

    pbar.close()

    return history, agent, env


# ============================================================
# 并行扫描 worker (模块级函数, Windows spawn 兼容)
# ============================================================
def _train_single_bw(bw, arrival_rate, num_episodes):
    """
    单个带宽值的完整训练+评估 (独立进程入口).
    返回: dict with bw, ddqn_blocking, threshold_blocking, random_blocking, ...
    增量保存: 完成后立即写入 results/bw_{bw:.0f}.npz
    """
    # 限制每个worker的线程数, 避免多进程争用CPU
    torch.set_num_threads(1)

    tag = f"[bw={bw:.0f}]"
    print(f"{tag} 启动训练...")

    # 训练 DDQN
    history, agent, _ = train_ddqn(
        licensed_bw=bw, arrival_rate=arrival_rate,
        num_episodes=num_episodes, seed=int(bw),
    )

    # 评估 DDQN
    eval_env = SidelinkEnv(
        num_ue=cfg.NUM_UE, licensed_bw_total=bw,
        arrival_rate=arrival_rate, seed=int(bw) + 1000,
    )
    ddqn_policy = make_ddqn_policy(agent)
    ddqn_block, ddqn_tput = evaluate_policy(eval_env, ddqn_policy, num_episodes=50)

    # 评估阈值策略
    th_env = SidelinkEnv(
        num_ue=cfg.NUM_UE, licensed_bw_total=bw,
        arrival_rate=arrival_rate, seed=int(bw) + 2000,
    )
    th_block, th_tput = evaluate_policy(th_env, threshold_policy, num_episodes=50)

    # 评估随机策略
    rand_env = SidelinkEnv(
        num_ue=cfg.NUM_UE, licensed_bw_total=bw,
        arrival_rate=arrival_rate, seed=int(bw) + 3000,
    )
    rand_block, rand_tput = evaluate_policy(rand_env, random_policy, num_episodes=50)

    print(f"{tag} 完成: "
          f"DDQN blocking={ddqn_block:.4f}, tput={ddqn_tput:.1f} | "
          f"Threshold blocking={th_block:.4f}, tput={th_tput:.1f} | "
          f"Random blocking={rand_block:.4f}, tput={rand_tput:.1f}")

    result = {
        "bw": bw,
        "ddqn_blocking": ddqn_block, "ddqn_tput": ddqn_tput,
        "threshold_blocking": th_block, "threshold_tput": th_tput,
        "random_blocking": rand_block, "random_tput": rand_tput,
        # 保存训练历史用于绘图
        "train_rewards": np.array(history["episode_rewards"]),
        "train_blocking": np.array(history["episode_blocking"]),
        "train_throughput": np.array(history["episode_throughput"]),
        "train_beta": np.array(history["beta"]),
        "train_temperature": np.array(history["temperature"]),
    }

    # 保存模型
    os.makedirs("results", exist_ok=True)
    torch.save(agent.online_net.state_dict(), f"results/model_bw_{bw:.0f}.pth")

    # 增量保存: 每个bw立即写入独立文件
    np.savez(f"results/bw_{bw:.0f}.npz", **result)
    print(f"{tag} 结果已保存至 results/bw_{bw:.0f}.npz")

    return result


def run_sweep_serial(bw_values, arrival_rate, num_episodes):
    """串行扫描 (带总体进度条)."""
    results = {"bw": [], "ddqn_blocking": [], "threshold_blocking": [],
               "random_blocking": [], "ddqn_tput": [], "threshold_tput": [],
               "random_tput": []}

    sweep_pbar = tqdm(bw_values, desc="扫描总进度", unit="bw",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for bw in sweep_pbar:
        sweep_pbar.set_postfix_str(f"当前 bw={bw:.0f}")
        r = _train_single_bw(bw, arrival_rate, num_episodes)
        for k in results:
            results[k].append(r[k])

    return results


def run_sweep_parallel(bw_values, arrival_rate, num_episodes, n_workers):
    """并行扫描 (多进程, 带总体进度)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = {"bw": [], "ddqn_blocking": [], "threshold_blocking": [],
               "random_blocking": [], "ddqn_tput": [], "threshold_tput": [],
               "random_tput": []}

    total = len(bw_values)
    print(f"[{time.strftime('%H:%M:%S')}] 启动并行扫描: {total} 个带宽, {n_workers} 个并行worker")
    print(f"带宽列表: {[f'{b:.0f}' for b in bw_values]}")
    print()

    sweep_pbar = tqdm(total=total, desc="扫描总进度", unit="bw",
                      bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_train_single_bw, float(bw), arrival_rate, num_episodes): bw
            for bw in bw_values
        }
        for future in as_completed(futures):
            bw = futures[future]
            try:
                r = future.result()
                for k in results:
                    results[k].append(r[k])
                sweep_pbar.set_postfix_str(f"完成 bw={r['bw']:.0f}")
                sweep_pbar.update(1)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] [主进程] bw={bw:.0f} 失败: {e}")

    sweep_pbar.close()

    # 按bw排序
    order = np.argsort(results["bw"])
    for k in results:
        results[k] = [results[k][i] for i in order]

    return results


def run_sweep(bw_values=None, arrival_rate=5.0, num_episodes=None, n_workers=1):
    """扫描不同授权带宽, 对比 DDQN vs 基线."""
    if bw_values is None:
        bw_values = np.arange(
            cfg.LICENSED_BW_MIN_MBPS,
            cfg.LICENSED_BW_MAX_MBPS + 1,
            cfg.LICENSED_BW_STEP_MBPS,
        )

    if n_workers > 1:
        results = run_sweep_parallel(bw_values, arrival_rate, num_episodes, n_workers)
    else:
        results = run_sweep_serial(bw_values, arrival_rate, num_episodes)

    # 保存
    os.makedirs("results", exist_ok=True)
    np.savez("results/sweep_results_v2.npz", **results)
    print(f"\n[{time.strftime('%H:%M:%S')}] 结果已保存至 results/sweep_results_v2.npz")

    # 汇总
    print(f"\n{'='*70}")
    print(f"  带宽      DDQN阻塞   阈值阻塞   随机阻塞   DDQN vs阈值")
    print(f"{'='*70}")
    for i, bw in enumerate(results["bw"]):
        d, t, r = results["ddqn_blocking"][i], results["threshold_blocking"][i], results["random_blocking"][i]
        reduction = (1 - d / (t + 1e-10)) * 100
        print(f"  {bw:6.0f}    {d:.4f}     {t:.4f}     {r:.4f}     {reduction:+.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Agentic DDQN Scheduler v2.0")
    parser.add_argument("--sweep", action="store_true",
                        help="扫描不同授权带宽")
    parser.add_argument("--parallel", "-j", type=int, default=1, metavar="N",
                        help="并行worker数 (需配合--sweep, 建议2-4)")
    parser.add_argument("--bw", type=float, default=500.0,
                        help="授权带宽 (Mbps), 用于单次训练")
    parser.add_argument("--episodes", type=int, default=None,
                        help="训练 episode 数 (覆盖config)")
    parser.add_argument("--arrival", type=float, default=1.5,
                        help="Poisson 到达率")
    args = parser.parse_args()

    start_time = time.time()
    if args.sweep:
        print(f"[{time.strftime('%H:%M:%S')}] 开始扫描训练")
        run_sweep(arrival_rate=args.arrival, num_episodes=args.episodes,
                  n_workers=args.parallel)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 单次训练: 授权带宽={args.bw:.0f} Mbps, UE数={cfg.NUM_UE}")
        history, agent, env = train_ddqn(
            licensed_bw=args.bw,
            arrival_rate=args.arrival,
            num_episodes=args.episodes,
        )
        final_block = np.mean(history["episode_blocking"][-100:])
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] 最终100ep平均阻塞率: {final_block:.4f}")
        print(f"[{time.strftime('%H:%M:%S')}] 最终β: {agent.beta:.2f}")
        print(f"[{time.strftime('%H:%M:%S')}] 总耗时: {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    main()
