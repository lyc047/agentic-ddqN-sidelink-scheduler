"""
DDQN 智能体: 双网络结构 + 优先经验回放 + ε-greedy 探索
基于 Chou et al., arXiv:2509.06775, 2025.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    """固定容量 FIFO 经验回放."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
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
# DDQN Agent
# ============================================================
class DDQNAgent:
    """Double DQN 智能体."""

    def __init__(self, state_dim, action_dim, config):
        """
        config: 包含所有超参数的命名空间/对象.
        期望字段:
          hidden_dim_1, hidden_dim_2, dropout_rate,
          learning_rate, lr_final, gamma,
          replay_capacity, batch_size, target_update_freq,
          epsilon_start, epsilon_min, epsilon_decay,
        """
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

        # ε-greedy (指数衰减: ε = max(ε_min, ε₀·e^(-t/K_decay)))
        self.epsilon_start = config.EPSILON_START
        self.epsilon_min   = config.EPSILON_MIN
        self.epsilon_k_decay = config.EPSILON_K_DECAY
        self.epsilon = self.epsilon_start
        self.total_steps = 0

    def select_action(self, state, training=True):
        """ε-greedy 动作选择."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        """指数衰减 ε = max(ε_min, ε₀·exp(-t/K_decay)). 对应论文公式(5)."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * np.exp(-self.total_steps / self.epsilon_k_decay)
        )

    def update_lr(self, progress):
        """
        学习率分阶段衰减.
        progress: 0→1 训练进度.
        """
        lr = self.lr_init + (self.lr_final - self.lr_init) * progress
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train_step(self):
        """从回放缓冲区采样并执行一次 DDQN 更新."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # 不足一个batch

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # ---- DDQN 目标值计算 (公式4) ----
        # 在线网络选择最优动作
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # 目标网络评估该动作
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            # y^DDQN = r + γ * Q_target(s', argmax_a Q_online(s', a))
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # 当前 Q 值
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # ---- Bellman 损失 (公式3) ----
        loss = nn.MSELoss()(current_q, targets)

        # ---- 梯度更新 ----
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        """θ⁻ ← θ (硬更新)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def update(self, state, action, reward, next_state, done):
        """存储经验并执行训练."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.update_epsilon()

        loss = self.train_step()

        # 定期同步目标网络
        if self.total_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return loss
