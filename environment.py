"""
无线仿真环境: Rician信道 + 3GPP UMi路径损耗 + M/M/1队列 + Wi-Fi LBT共存
基于 Chou et al., arXiv:2509.06775, 2025.
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
    ACTION_DIM, STATE_DIM,
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


def get_mode_bandwidth(action, licensed_bw_total):
    """返回指定动作对应的带宽 (Mbps)."""
    if action in (0, 1, 2, 3):
        return licensed_bw_total / NUM_LICENSED_MODES
    else:  # action == 4 (SL-U-5G)
        return UNLICENSED_BW_MBPS


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
# 主环境类
# ============================================================
class SidelinkEnv:
    """
    NR Sidelink 调度仿真环境.
    状态 (7维):
      [0] 归一化队列占用率 (0~1)
      [1] CC-28G 剩余资源比
      [2] CC-26G 剩余资源比
      [3] SL-L-28G 剩余资源比
      [4] SL-L-26G 剩余资源比
      [5] SL-U-5G 剩余资源比
      [6] Wi-Fi 空闲概率

    动作 (5): CC-28G, CC-26G, SL-L-28G, SL-L-26G, SL-U-5G
    """

    def __init__(self, licensed_bw_total=500.0, arrival_rate=0.5,
                 seed=None, dynamic_channel=True):
        """
        licensed_bw_total: 授权频段总带宽 (Mbps)
        arrival_rate: Poisson 到达率 λ
        dynamic_channel: True→每包新信道; False→每episode固定信道
        """
        self.licensed_bw = licensed_bw_total
        self.arrival_rate = arrival_rate
        self.dynamic_channel = dynamic_channel

        # RNG
        self.rng = np.random.RandomState(seed)

        # Wi-Fi 共存
        self.wifi = WiFiCoexistence()

        # 队列
        self.queue = []
        self.queue_max = QUEUE_MAX_CAPACITY

        # 统计
        self.total_arrivals = 0
        self.total_blocked = 0
        self.total_transmitted = 0
        self.episode_rewards = []

    def reset(self):
        """重置环境状态 (每个episode开始)."""
        self.queue = []
        self.total_arrivals = 0
        self.total_blocked = 0
        self.total_transmitted = 0
        self.episode_rewards = []
        self.wifi.state = 0
        return self._get_state()

    def _get_state(self):
        """构建7维状态向量."""
        q_ratio = len(self.queue) / self.queue_max if self.queue_max > 0 else 0.0

        # 剩余资源比: 简化版设为1.0 (无带内竞争)
        residual_ratios = np.ones(5, dtype=np.float32)
        # SL-U 的剩余比率受 Wi-Fi 状态影响
        residual_ratios[4] = 1.0 - self.wifi.tx_prob * 0.5

        wifi_idle = float(self.wifi.is_idle())

        state = np.array([
            q_ratio,
            *residual_ratios,
            wifi_idle,
        ], dtype=np.float32)
        return state

    def _generate_arrivals(self):
        """Poisson 到达生成."""
        n_arrivals = self.rng.poisson(self.arrival_rate)
        n_blocked = 0
        for _ in range(n_arrivals):
            if len(self.queue) < self.queue_max:
                self.queue.append(PACKET_SIZE_BITS)
            else:
                n_blocked += 1
        self.total_arrivals += n_arrivals
        self.total_blocked += n_blocked
        return n_arrivals, n_blocked

    def _compute_sinr(self, action):
        """
        计算瞬时 SINR (线性值).
        基于平均SNR + Rician衰落 + 路径损耗变化.
        """
        _, freq_key, mode_type, snr_mean_db = ACTION_INFO[action]
        is_mmwave = (freq_key != "5G")

        # Rician 信道增益 |H|^2
        h = rician_fading(RICIAN_K, is_mmwave)
        channel_gain = np.abs(h) ** 2

        # 平均 SNR (线性)
        snr_mean_lin = db_to_linear(snr_mean_db)

        # 瞬时 SINR = 平均SNR × |H|^2 (衰落围绕均值波动)
        sinr_linear = snr_mean_lin * channel_gain
        return max(sinr_linear, 1e-10)

    def step(self, action):
        """
        执行一个决策epoch.
        action: 0-4
        返回: (next_state, reward, done, info)
        """
        info = {"blocked_arrival": 0, "blocked_transmit": 0, "n_served": 0}

        # ---- 1. Poisson 到达 ----
        _, blocked_arrival = self._generate_arrivals()
        info["blocked_arrival"] = blocked_arrival

        # ---- 2. Wi-Fi 状态更新 ----
        self.wifi.step()

        # ---- 3. 传输决策 ----
        reward = 0.0
        if len(self.queue) > 0:
            _, _, mode_type, _ = ACTION_INFO[action]

            # SL-U: 检查LBT
            if mode_type == "slu" and not self.wifi.is_idle():
                info["blocked_transmit"] = 1
            else:
                # 计算瞬时速率
                sinr_lin = self._compute_sinr(action)
                bw_mbps = get_mode_bandwidth(action, self.licensed_bw)
                bw_hz = bw_mbps * 1e6
                rate_bps = bw_hz * np.log2(1.0 + sinr_lin)

                # 服务包数 = floor(rate * duration / packet_size) + 概率化分数部分
                max_packets_per_epoch = rate_bps * EPOCH_DURATION / PACKET_SIZE_BITS
                n_served = int(np.floor(max_packets_per_epoch))
                frac = max_packets_per_epoch - n_served
                if frac > 0 and self.rng.random() < frac:
                    n_served += 1
                n_served = min(n_served, len(self.queue))

                for _ in range(n_served):
                    self.queue.pop(0)
                self.total_transmitted += n_served
                info["n_served"] = n_served

                # 奖励 = B·log₂(1+SINR) / 1e6 (Mbps尺度)
                reward = rate_bps / 1e6

        self.episode_rewards.append(reward)

        # ---- 4. 下一状态 ----
        next_state = self._get_state()

        # ---- 5. 终止判断 ----
        done = (len(self.queue) >= self.queue_max)

        return next_state, reward, done, info

    def get_blocking_rate(self):
        """计算当前episode的阻塞率."""
        if self.total_arrivals == 0:
            return 0.0
        return self.total_blocked / self.total_arrivals

    def get_throughput(self):
        """计算平均吞吐量 (Mbps)."""
        if len(self.episode_rewards) == 0:
            return 0.0
        return np.mean(self.episode_rewards)
