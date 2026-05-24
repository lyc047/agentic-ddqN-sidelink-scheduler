"""
DDQN 智能体 v2.0: 双网络结构 + 经验回放 + ε-greedy探索 + CMDP拉格朗日对偶
基于 Chou et al., arXiv:2509.06775, 2025.

v2.0 新增: 拉格朗日乘子 β 用于CMDP约束 — 阻塞率超过目标时自动增大惩罚权重
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ============================================================
# Q 网络定义
# ============================================================
class QNetwork(nn.Module):
    """全连接 Q 网络: 输入状态 → 输出每个动作的 Q 值."""

    def __init__(self, state_dim, action_dim, hidden1, hidden2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.LeakyReLU(0.01),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden2, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 初始化 (与论文 kernel_initializer='glorot_uniform' 对应)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.net(state)


# ============================================================
# 经验回放缓冲区
# ============================================================
class ReplayBuffer:
    """固定容量 FIFO 经验回放 (list-based, O(1) 索引)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self._pos = 0

    def push(self, state, action, reward, next_state, done):
        entry = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self._pos] = entry
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        indices = np.random.randint(0, n, size=min(batch_size, n))
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DDQN Agent (v2.0: + CMDP Lagrangian)
# ============================================================
class DDQNAgent:
    """Double DQN 智能体 + CMDP拉格朗日对偶."""

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ

        # 网络
        self.online_net = QNetwork(
            state_dim, action_dim,
            config.HIDDEN_DIM_1, config.HIDDEN_DIM_2,
            config.DROPOUT_RATE
        )
        self.target_net = QNetwork(
            state_dim, action_dim,
            config.HIDDEN_DIM_1, config.HIDDEN_DIM_2,
            config.DROPOUT_RATE
        )
        self.target_net.load_state_dict(self.online_net.state_dict())

        # 优化器与学习率调度
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=config.LEARNING_RATE
        )
        self.lr_init = config.LEARNING_RATE
        self.lr_final = config.LR_FINAL

        # 经验回放
        self.replay_buffer = ReplayBuffer(config.REPLAY_CAPACITY)

        # Softmax (Boltzmann) 探索: 温度衰减 (v2.0, 替代ε-greedy)
        self.temp_start = config.TEMPERATURE_START
        self.temp_min   = config.TEMPERATURE_MIN
        self.temp_k_decay = config.TEMPERATURE_K_DECAY
        self.temperature = self.temp_start
        self.total_steps = 0

        # ---- CMDP 拉格朗日对偶 (v2.0 新增) ----
        self.beta = 0.0
        self.target_blocking = config.TARGET_BLOCKING
        self.lagrangian_lr = config.LAGRANGIAN_LR
        self.beta_update_freq = config.BETA_UPDATE_FREQ
        self._blocking_history = []

    def select_action(self, state, training=True):
        """Softmax动作选择: 天然打破多UE羊群效应."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_net(state_t)
            if training:
                temp = max(self.temp_min, self.temperature)
                probs = torch.softmax(q_values / temp, dim=1).numpy()[0]
                return np.random.choice(self.action_dim, p=probs)
            else:
                return q_values.argmax(dim=1).item()

    def update_temperature(self):
        """温度指数衰减: τ = max(τ_min, τ₀·exp(-t/K_decay))."""
        self.temperature = max(
            self.temp_min,
            self.temp_start * np.exp(-self.total_steps / self.temp_k_decay)
        )

    def update_lr(self, progress):
        """学习率线性衰减."""
        lr = self.lr_init + (self.lr_final - self.lr_init) * progress
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update_beta(self, blocking_rate, episode_num=0):
        """
        CMDP对偶变量更新: β ← max(0, β + η·(阻塞率 − 目标阻塞率)).
        前2000ep为warmup期 (β=0, 专注吞吐学习); β上限20防止惩罚淹没奖励信号.
        """
        self._blocking_history.append(blocking_rate)
        if len(self._blocking_history) > self.beta_update_freq:
            self._blocking_history.pop(0)

        # Warmup: 前500ep不启用阻塞惩罚 (v2.1: 缩短以适配更短训练)
        if episode_num < 500:
            self.beta = 0.0
            return

        if len(self._blocking_history) >= 10:
            avg_blocking = np.mean(self._blocking_history)
            self.beta = min(3.0, max(0.0, self.beta
                            + self.lagrangian_lr * (avg_blocking - self.target_blocking)))

    def train_step(self):
        """从回放缓冲区采样并执行一次 DDQN 更新."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # ---- DDQN 目标值 ----
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # 当前 Q 值
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Bellman 损失
        loss = nn.MSELoss()(current_q, targets)

        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        """θ⁻ ← θ (硬更新)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update(self, state, action, reward, next_state, done):
        """存储经验并执行训练步."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.update_temperature()

        loss = self.train_step()

        if self.total_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return loss
