"""
Agentic DDQN Sidelink Scheduler — 超参数配置
基于: Chou et al., "Agentic DDQN-Based Scheduling for Licensed and
Unlicensed Band Allocation in Sidelink Networks," arXiv:2509.06775, 2025.
"""

# ============================================================
# 物理层参数
# ============================================================
CARRIER_FREQ = {
    "28G": 28e9,   # 28 GHz (FR2)
    "26G": 26e9,   # 26 GHz (FR2)
    "5G":  5e9,    # 5 GHz  (unlicensed)
}

SPEED_OF_LIGHT = 3e8  # m/s

# 天线高度 (m) — 3GPP UMi
H_BS = 10.0   # gNB 天线高度
H_UT = 1.5    # UE 天线高度

# 收发距离 (m)
DIST_CC = 100.0   # gNB ↔ UE (CC链路)
DIST_SL = 5.0     # UE ↔ UE  (SL链路)
DIST_WIFI = 5.0   # Wi-Fi节点间距

# 平均 SNR (dB) — 论文指定值
SNR_CC_DB  = 40.0   # CC链路
SNR_SL_DB  = 13.0   # SL-L链路
SNR_SLU_DB = 0.0    # SL-U链路 (高竞争场景，对齐3GPP TR 38.889)

# 发射功率 (dBm)
TX_POWER_CC_DBM  = 30.0   # gNB
TX_POWER_SL_DBM  = 23.0   # UE (SL)
TX_POWER_SLU_DBM = 23.0   # UE (SL-U)

# 噪声系数 (dB)
NOISE_FIGURE_DB = 7.0

# Rician K 因子
RICIAN_K = 10.0
# mmWave NLOS 缩放参数 (L > 1 表示散射更少)
MMWAVE_NLOS_L = 10.0

# ============================================================
# 带宽配置
# ============================================================
UNLICENSED_BW_MBPS = 100.0      # 5 GHz 非授权频段固定带宽 (Mbps)
LICENSED_BW_MIN_MBPS = 500.0    # 授权频段总带宽下限
LICENSED_BW_MAX_MBPS = 1000.0   # 授权频段总带宽上限
LICENSED_BW_STEP_MBPS = 100.0   # 扫描步长

# 每个授权模式平均分配带宽
# CC-28G, CC-26G, SL-L-28G, SL-L-26G 各占 1/4
NUM_LICENSED_MODES = 4

# ============================================================
# 流量与队列参数 (3GPP FTP 模型)
# ============================================================
PACKET_SIZE_BYTES = 0.5e6       # 每包 0.5 MBytes
PACKET_SIZE_BITS  = PACKET_SIZE_BYTES * 8
POISSON_ARRIVAL_RATE = 5.0      # λ: 高负载下授权频段也不够用
QUEUE_MAX_CAPACITY = 50         # 有限队列容量

# ============================================================
# Wi-Fi 共存参数 (5 GHz 非授权频段)
# ============================================================
WIFI_TX_PROB = 0.3              # Wi-Fi 传输概率 → 空闲概率 ≈ 0.7
WIFI_IDLE_CORRELATION = 0.8     # Wi-Fi 状态时间相关性 (一阶马尔可夫)

# ============================================================
# DDQN 网络超参数
# ============================================================
STATE_DIM  = 7   # 队列占用率 + 5个频段剩余资源比 + Wi-Fi空闲概率
ACTION_DIM = 5   # CC-28G, CC-26G, SL-L-28G, SL-L-26G, SL-U-5G
HIDDEN_DIM_1 = 64    # 快速版用64
HIDDEN_DIM_2 = 32    # 快速版用32
DROPOUT_RATE = 0.3

# ============================================================
# 训练超参数
# ============================================================
LEARNING_RATE  = 1e-7    # 初始学习率 (论文: 1e-7 → 1e-9)
LR_FINAL       = 1e-9    # 最终学习率
BATCH_SIZE     = 64
GAMMA          = 0.99    # 折扣因子
REPLAY_CAPACITY = 200_000  # 长训练用更大缓冲区
TARGET_UPDATE_FREQ = 500   # 目标网络更新频率 (步)

# ε-greedy 探索 (指数衰减: ε = max(ε_min, ε₀·e^(-t/K_decay)))
EPSILON_START    = 1.0
EPSILON_MIN      = 0.05
EPSILON_K_DECAY  = 400000    # 1.2M步后ε≈0.05

# 每epoch时长 (s), 用于将速率转换为服务概率
EPOCH_DURATION = 0.01

# 训练规模 (简化版)
NUM_EPISODES       = 15000   # 过夜训练 (~5小时)
MAX_STEPS_PER_EP   = 100     # 每episode最大步数
