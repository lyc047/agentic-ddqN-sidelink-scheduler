"""
无线仿真环境 v2.0: 多UE竞争 + Rician信道 + 3GPP UMi路径损耗 + M/M/1队列 + Wi-Fi LBT共存
基于 Chou et al., arXiv:2509.06775, 2025.

v2.0 新增:
  - 多UE竞争: N个V2V pair共享同一资源池, 同模式带宽均分+干扰
  - 动态剩余资源比: 状态中的5个模式可用性根据上一轮使用量实时变化
  - HOL延迟约束: 超时丢包 (deadline=50ms), 计入阻塞率
  - CMDP奖励: r = 实际吞吐 − β·I(阻塞事件), β通过对偶梯度更新
"""

import numpy as np
from config import (
    CARRIER_FREQ, SPEED_OF_LIGHT, H_BS, H_UT,
    DIST_CC, DIST_SL, DIST_WIFI,
    SNR_CC_DB, SNR_SL_DB, SNR_SLU_DB,
    TX_POWER_CC_DBM, TX_POWER_SL_DBM, TX_POWER_SLU_DBM,
    NOISE_FIGURE_DB, RICIAN_K, MMWAVE_NLOS_L,
    UNLICENSED_BW_MBPS, NUM_LICENSED_MODES,
    PACKET_SIZE_BITS, QUEUE_MAX_CAPACITY, EPOCH_DURATION,
    WIFI_TX_PROB, WIFI_IDLE_CORRELATION,
    ACTION_DIM, STATE_DIM, NUM_UE, DEADLINE_STEPS,
    INTERFERENCE_COUPLING, COORD_PENALTY_WEIGHT,
)

# Boltzmann常数
K_B = 1.380649e-23  # J/K
T0  = 290.0         # K (标准温度)


# ============================================================
# 辅助: dB ↔ linear
# ============================================================
def db_to_linear(db_val):
    return 10.0 ** (db_val / 10.0)


def linear_to_db(lin_val):
    return 10.0 * np.log10(lin_val + 1e-30)


# ============================================================
# 3GPP TR 38.901 UMi Street Canyon 路径损耗模型
# ============================================================
def umi_path_loss(d_3d, f_c, h_t, h_r):
    """
    计算 UMi-Street Canyon LOS/NLOS 路径损耗 (dB).
    返回: (PL_LOS_dB, PL_NLOS_dB)
    """
    # 断点距离 (m)
    d_bp = 4.0 * h_t * h_r * f_c / SPEED_OF_LIGHT

    # ---- LOS ----
    if d_3d < d_bp:
        pl_los = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(f_c / 1e9)
    else:
        pl_los = (32.4 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(f_c / 1e9)
                  - 9.5 * np.log10(d_bp ** 2 + (h_t - h_r) ** 2))

    # ---- NLOS ----
    pl_nlos_emp = (22.4 + 35.3 * np.log10(d_3d)
                   + 21.3 * np.log10(f_c / 1e9)
                   - 0.3 * (h_r - 1.5))
    pl_nlos = max(pl_los, pl_nlos_emp)

    return pl_los, pl_nlos


# ============================================================
# Rician 衰落信道
# ============================================================
def rician_fading(K, is_mmwave=True):
    """
    生成 Rician 衰落信道系数 H.
    K: Rician K-factor
    is_mmwave: True → NLOS方差 1/L (散射少); False → NLOS方差 1 (散射丰富)
    返回: 复数值 H
    """
    sqrt_k_ratio = np.sqrt(K / (K + 1))
    sqrt_1_ratio = np.sqrt(1.0 / (K + 1))

    # LOS: 随机相位
    phase = np.random.uniform(0, 2 * np.pi)
    h_los = np.exp(-1j * phase)

    # NLOS: 复高斯
    if is_mmwave:
        nlos_var = 1.0 / MMWAVE_NLOS_L
    else:
        nlos_var = 1.0
    h_nlos = np.sqrt(nlos_var / 2.0) * (
        np.random.randn() + 1j * np.random.randn()
    )

    h = sqrt_k_ratio * h_los + sqrt_1_ratio * h_nlos
    return h


# ============================================================
# 传输模式定义
# ============================================================
# 动作索引 → (模式名, 载频key, 类型, 平均SNR_dB)
ACTION_INFO = {
    0: ("CC-28G",   "28G", "cc",  SNR_CC_DB),
    1: ("CC-26G",   "26G", "cc",  SNR_CC_DB),
    2: ("SL-L-28G", "28G", "sl",  SNR_SL_DB),
    3: ("SL-L-26G", "26G", "sl",  SNR_SL_DB),
    4: ("SL-U-5G",  "5G",  "slu", SNR_SLU_DB),
}


def get_mode_bandwidth(action, licensed_bw_total, num_users=1):
    """返回指定动作的每UE带宽 (Mbps), 考虑同模式用户均分."""
    if action in (0, 1, 2, 3):
        return licensed_bw_total / NUM_LICENSED_MODES / num_users
    else:  # action == 4 (SL-U-5G)
        return UNLICENSED_BW_MBPS / num_users


def get_mode_snr_db(action):
    """返回指定动作的平均 SNR (dB)."""
    return ACTION_INFO[action][3]


# ============================================================
# Wi-Fi 共存模型 (一阶马尔可夫 ON/OFF 过程)
# ============================================================
class WiFiCoexistence:
    """
    5 GHz 非授权频段 Wi-Fi 共存模型 (非对称马尔可夫链).
    Wi-Fi 状态: 0 = 空闲, 1 = 占用.
    LBT: 当 Wi-Fi 占用时 SL-U 必须等待.

    稳态: P(idle) = 1 - tx_prob = 0.7 (默认)
    转移: P(idle→busy) = p_ib, P(busy→idle) = p_bi
          其中 p_bi / (p_ib + p_bi) = P(idle)
    """

    def __init__(self, tx_prob=WIFI_TX_PROB, persistence=WIFI_IDLE_CORRELATION):
        self.tx_prob = tx_prob
        self.p_idle = 1.0 - tx_prob   # 稳态空闲概率

        # 非对称转移概率
        self.p_ib = (1.0 - persistence) * 0.5  # idle → busy
        # 由稳态条件解出: p_bi = p_ib * p_idle / (1 - p_idle)
        self.p_bi = self.p_ib * self.p_idle / max(1.0 - self.p_idle, 1e-10)
        self.state = 0  # 初始空闲

    def step(self):
        """更新 Wi-Fi 状态."""
        if self.state == 0:
            self.state = 1 if np.random.random() < self.p_ib else 0
        else:
            self.state = 0 if np.random.random() < self.p_bi else 1

    def is_idle(self):
        return self.state == 0


# ============================================================
# 主环境类 — 多UE竞争版
# ============================================================
class SidelinkEnv:
    """
    NR Sidelink 调度仿真环境 (多UE竞争).

    状态 (8维):
      [0] 归一化队列占用率 (0~1)
      [1] CC-28G 剩余资源比 (基于上轮使用量, 动态)
      [2] CC-26G 剩余资源比
      [3] SL-L-28G 剩余资源比
      [4] SL-L-26G 剩余资源比
      [5] SL-U-5G 剩余资源比
      [6] Wi-Fi 空闲指示 (0/1)
      [7] HOL等待时间/截止时间 (归一化紧急度)

    动作 (5): CC-28G, CC-26G, SL-L-28G, SL-L-26G, SL-U-5G

    多UE机制:
      - N个UE各自维护独立队列, 每个epoch独立选择动作
      - 同模式多用户 → 带宽均分 + 干扰惩罚
      - 上轮模式使用量决定下轮状态的剩余资源比
      - 数据包超时 (DEADLINE_STEPS) → 丢弃计入阻塞

    CMDP奖励:
      r_i = actual_throughput_i − β · I(blocking_event_i)
      β 由外部通过对偶梯度更新 (见 ddqn_agent.update_beta)
    """

    def __init__(self, num_ue=NUM_UE, licensed_bw_total=500.0, arrival_rate=1.5,
                 seed=None, dynamic_channel=True):
        self.num_ue = num_ue
        self.licensed_bw = licensed_bw_total
        self.arrival_rate = arrival_rate
        self.dynamic_channel = dynamic_channel
        self.rng = np.random.RandomState(seed)
        self.wifi = WiFiCoexistence()
        self.queue_max = QUEUE_MAX_CAPACITY

        # ---- Per-UE 队列: 每元素 = arrival_step ----
        self.queues = [[] for _ in range(num_ue)]

        # ---- Per-UE 统计 ----
        self.total_arrivals = np.zeros(num_ue, dtype=int)
        self.total_blocked = np.zeros(num_ue, dtype=int)
        self.total_transmitted = np.zeros(num_ue, dtype=int)

        # ---- 上轮模式使用量 (用于构建状态的剩余资源比) ----
        self.prev_mode_usage = np.zeros(5, dtype=int)

        # ---- CMDP 拉格朗日乘子 ----
        self.beta = 0.0

        # ---- 当前步计数 (用于HOL计算) ----
        self.current_step = 0

        # ---- 每UE每步奖励记录 ----
        self.step_rewards = []

    def reset(self):
        """重置环境状态 (每个episode开始). 返回所有UE的初始状态列表."""
        self.queues = [[] for _ in range(self.num_ue)]
        self.total_arrivals = np.zeros(self.num_ue, dtype=int)
        self.total_blocked = np.zeros(self.num_ue, dtype=int)
        self.total_transmitted = np.zeros(self.num_ue, dtype=int)
        self.prev_mode_usage = np.zeros(5, dtype=int)
        self.wifi.state = 0
        self.current_step = 0
        self.step_rewards = []
        return [self._get_state(i) for i in range(self.num_ue)]

    def _get_state(self, ue_id):
        """构建单个UE的8维观测状态."""
        q_ratio = len(self.queues[ue_id]) / self.queue_max if self.queue_max > 0 else 0.0

        # 动态剩余资源比: 基于上轮模式使用量
        # 剩余比 = 1 / prev_users, 限制在 [0.1, 1.0]
        residual_ratios = np.ones(5, dtype=np.float32)
        for m in range(5):
            if self.prev_mode_usage[m] > 0:
                residual_ratios[m] = max(0.1, 1.0 / self.prev_mode_usage[m])
        # SL-U额外受Wi-Fi影响
        residual_ratios[4] *= max(0.3, 1.0 - self.wifi.tx_prob)

        wifi_idle = float(self.wifi.is_idle())

        # HOL等待时间 (归一化)
        if len(self.queues[ue_id]) > 0:
            hol_arrival = self.queues[ue_id][0]
            hol_wait = min(1.0, (self.current_step - hol_arrival) / DEADLINE_STEPS)
        else:
            hol_wait = 0.0

        return np.array([
            q_ratio,
            *residual_ratios,
            wifi_idle,
            hol_wait,
        ], dtype=np.float32)

    def _generate_arrivals(self):
        """为所有UE生成Poisson到达. 返回每UE的 (到达数, 阻塞数)."""
        arrivals = np.zeros(self.num_ue, dtype=int)
        blocked = np.zeros(self.num_ue, dtype=int)
        for ue_id in range(self.num_ue):
            n = self.rng.poisson(self.arrival_rate)
            arrivals[ue_id] = n
            for _ in range(n):
                if len(self.queues[ue_id]) < self.queue_max:
                    self.queues[ue_id].append(self.current_step)
                else:
                    blocked[ue_id] += 1
            self.total_arrivals[ue_id] += n
            self.total_blocked[ue_id] += blocked[ue_id]
        return arrivals, blocked

    def _compute_sinr(self, action, num_users_on_mode):
        """
        计算考虑多用户干扰的瞬时SINR (线性值).
        SINR_eff = SNR_base × |H|² / (1 + α·(K-1))
        其中 K = 同模式用户数, α = 干扰耦合系数.
        """
        _, freq_key, mode_type, snr_mean_db = ACTION_INFO[action]
        is_mmwave = (freq_key != "5G")

        h = rician_fading(RICIAN_K, is_mmwave)
        channel_gain = np.abs(h) ** 2
        snr_mean_lin = db_to_linear(snr_mean_db)
        sinr_linear = snr_mean_lin * channel_gain

        # 多用户干扰: SINR随用户数增加而衰减
        if num_users_on_mode > 1:
            sinr_linear /= (1.0 + INTERFERENCE_COUPLING * (num_users_on_mode - 1))

        return max(sinr_linear, 1e-10)

    def step(self, actions):
        """
        执行一个决策epoch (所有UE同时).
        actions: list of N ints (0~4)
        返回: (next_states, rewards, done, infos)
        """
        if len(actions) != self.num_ue:
            raise ValueError(f"需要 {self.num_ue} 个动作, 收到 {len(actions)}")

        # ---- 1. Poisson 到达 ----
        arrivals, arrival_blocked = self._generate_arrivals()

        # ---- 2. Wi-Fi 状态更新 ----
        self.wifi.step()

        # ---- 3. 统计本轮模式选择 ----
        mode_usage = np.zeros(5, dtype=int)
        for a in actions:
            mode_usage[a] += 1

        # ---- 4. 处理每个UE的传输 ----
        rewards = [0.0] * self.num_ue
        infos = []

        for ue_id, action in enumerate(actions):
            queue = self.queues[ue_id]
            info = {
                "arrivals": int(arrivals[ue_id]),
                "arrival_blocked": int(arrival_blocked[ue_id]),
                "deadline_drops": 0,
                "n_served": 0,
                "action": action,
            }

            if len(queue) == 0:
                infos.append(info)
                continue

            _, _, mode_type, _ = ACTION_INFO[action]

            # LBT: SL-U在Wi-Fi忙时被阻塞
            if mode_type == "slu" and not self.wifi.is_idle():
                infos.append(info)
                continue

            # 计算SINR和速率 (考虑同模式竞争)
            n_users = int(mode_usage[action])
            sinr_lin = self._compute_sinr(action, n_users)
            bw_mbps = get_mode_bandwidth(action, self.licensed_bw, n_users)
            bw_hz = bw_mbps * 1e6
            rate_bps = bw_hz * np.log2(1.0 + sinr_lin)

            # 可服务包数
            max_packets = rate_bps * EPOCH_DURATION / PACKET_SIZE_BITS
            n_candidate = int(np.floor(max_packets))
            frac = max_packets - n_candidate
            if frac > 0 and self.rng.random() < frac:
                n_candidate += 1
            n_candidate = min(n_candidate, len(queue))

            # 逐包处理: 检查HOL超时
            served = 0
            dropped_deadline = 0
            for _ in range(n_candidate):
                if len(queue) == 0:
                    break
                arrival_step = queue[0]
                if self.current_step - arrival_step >= DEADLINE_STEPS:
                    queue.pop(0)
                    dropped_deadline += 1
                else:
                    queue.pop(0)
                    served += 1

            self.total_transmitted[ue_id] += served
            self.total_blocked[ue_id] += dropped_deadline
            info["n_served"] = served
            info["deadline_drops"] = dropped_deadline

            # 奖励: 实际吞吐 − 协调惩罚 (打破多UE羊群效应)
            actual_throughput_mbps = (served * PACKET_SIZE_BITS) / EPOCH_DURATION / 1e6
            n_users = int(mode_usage[action])
            if n_users > 1:
                congestion = (n_users - 1) / (self.num_ue - 1)
                actual_throughput_mbps -= COORD_PENALTY_WEIGHT * congestion
            rewards[ue_id] = actual_throughput_mbps

            infos.append(info)

        # ---- 5. 保存本轮模式使用量供下轮状态使用 ----
        self.prev_mode_usage = mode_usage

        # ---- 6. 下一状态 ----
        self.current_step += 1
        self.step_rewards.append(np.mean(rewards))
        next_states = [self._get_state(i) for i in range(self.num_ue)]

        # ---- 7. 终止条件: episode固定长度, 不提前终止 ----
        # 阻塞惩罚已通过CMDP奖励体现, 提前终止会缩短探索、阻碍学习
        done = False

        return next_states, rewards, done, infos

    # ============================================================
    # 统计接口
    # ============================================================
    def get_blocking_rate(self, ue_id=None):
        """返回系统平均阻塞率 (或指定UE)."""
        if ue_id is not None:
            if self.total_arrivals[ue_id] == 0:
                return 0.0
            return self.total_blocked[ue_id] / self.total_arrivals[ue_id]
        total_a = self.total_arrivals.sum()
        if total_a == 0:
            return 0.0
        return self.total_blocked.sum() / total_a

    def get_throughput(self):
        """返回系统平均吞吐量 (Mbps)."""
        total_bits = self.total_transmitted.sum() * PACKET_SIZE_BITS
        total_time = max(self.current_step, 1) * EPOCH_DURATION
        return total_bits / total_time / 1e6

    def get_avg_queue_length(self):
        """返回所有UE的平均队列长度."""
        return np.mean([len(q) for q in self.queues])
