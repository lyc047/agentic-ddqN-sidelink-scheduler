"""
Agentic DDQN Sidelink Scheduler — 超参数配置 v2.0
基于: Chou et al., "Agentic DDQN-Based Scheduling for Licensed and
Unlicensed Band Allocation in Sidelink Networks," arXiv:2509.06775, 2025.

v2.0 新增: 多UE竞争 + CMDP拉格朗日对偶 + HOL延迟约束
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
POISSON_ARRIVAL_RATE = 1.5      # λ: 多UE下需调度但随机探索也能存活 (v2.0校准)
QUEUE_MAX_CAPACITY = 100        # 增大队列避免过早饱和

# ============================================================
# 多UE竞争参数 (v2.0 新增)
# ============================================================
NUM_UE = 3                      # V2V pair 数量
INTERFERENCE_COUPLING = 0.5     # 同模式干扰耦合系数 (0~1)
DEADLINE_STEPS = 5              # 数据包过期时间 (epoch数), 5×10ms=50ms

# ============================================================
# CMDP 拉格朗日对偶参数 (v2.0 新增)
# ============================================================
TARGET_BLOCKING = 0.02          # 目标阻塞率 ξ
LAGRANGIAN_LR = 0.005           # 对偶变量学习率 η
BETA_UPDATE_FREQ = 100          # β更新所需的阻塞率窗口大小 (episodes)

# 多UE协调惩罚 (v2.0 打破羊群效应)
# ============================================================
COORD_PENALTY_WEIGHT = 100.0    # 同模式拥挤惩罚权重 (v2.1: 降低以适配低BW场景)

# ============================================================
# Wi-Fi 共存参数 (5 GHz 非授权频段)
# ============================================================
WIFI_TX_PROB = 0.3              # Wi-Fi 传输概率 → 空闲概率 ≈ 0.7
WIFI_IDLE_CORRELATION = 0.8     # Wi-Fi 状态时间相关性 (一阶马尔可夫)

# ============================================================
# DDQN 网络超参数
# ============================================================
STATE_DIM  = 8   # 队列占用率 + 5个频段剩余资源比(动态) + Wi-Fi空闲 + HOL紧急度
ACTION_DIM = 5   # CC-28G, CC-26G, SL-L-28G, SL-L-26G, SL-U-5G
HIDDEN_DIM_1 = 32
HIDDEN_DIM_2 = 16
DROPOUT_RATE = 0.1

# ============================================================
# 训练超参数
# ============================================================
LEARNING_RATE  = 1e-4    # 初始学习率 (多UE需更高LR才能有效学习)
LR_FINAL       = 1e-6    # 最终学习率
BATCH_SIZE     = 128
GAMMA          = 0.99
REPLAY_CAPACITY = 500_000  # 更大缓冲区保留好经验更久
TARGET_UPDATE_FREQ = 200   # 更频繁同步目标网络

# ε-greedy 探索 (指数衰减: ε = max(ε_min, ε₀·e^(-t/K_decay)))
# Softmax探索 (温度衰减): τ = max(τ_min, τ_start·e^(-t/K_decay))
# 天然打破多UE羊群效应: 即使Q值相似, softmax采样也会分散动作
TEMPERATURE_START = 10.0     # 初始高温 → 近似均匀随机
TEMPERATURE_MIN   = 0.5      # 最低温度 → 基本贪婪但不完全确定
TEMPERATURE_K_DECAY = 300000

# 每epoch时长 (s), 用于将速率转换为服务概率
EPOCH_DURATION = 0.01

# 训练规模
NUM_EPISODES       = 15000   # 过夜训练 (~5小时)
MAX_STEPS_PER_EP   = 100     # 每episode最大步数
