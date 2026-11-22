import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

ACTIONS = [2, 4, 6, 8]
STATE_DIM = 20  # one-hot encoded state
ACTION_DIM = len(ACTIONS)

# ─────────────────────────────────────────
# Neural Network (2-layer MLP)
# ─────────────────────────────────────────
class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
# Helper: state index → one-hot tensor
# ─────────────────────────────────────────
def state_to_tensor(state_idx):
    one_hot = np.zeros(STATE_DIM, dtype=np.float32)
    one_hot[state_idx] = 1.0
    return torch.FloatTensor(one_hot)


# ─────────────────────────────────────────
# DQN Agent  (drop-in replacement for Agent)
# ─────────────────────────────────────────
class Agent:
    def __init__(self):
        # One network + replay buffer per vehicle
        self.networks = {}       # veh_id → DQNNetwork (online)
        self.targets  = {}       # veh_id → DQNNetwork (target)
        self.optimizers = {}     # veh_id → Adam optimizer
        self.buffers  = {}       # veh_id → ReplayBuffer

        self.batch_size   = 32
        self.target_update = 20  # sync target network every N calls to update()
        self.update_counts = {}  # veh_id → int

        self.gamma = 0.9         # kept here so update() is self-contained
        self.lr    = 1e-3

    # ── initialise a new vehicle's network ──────────────────────────────────
    def init(self, veh_id):
        if veh_id not in self.networks:
            net    = DQNNetwork()
            target = DQNNetwork()
            target.load_state_dict(net.state_dict())
            target.eval()

            self.networks[veh_id]     = net
            self.targets[veh_id]      = target
            self.optimizers[veh_id]   = optim.Adam(net.parameters(), lr=self.lr)
            self.buffers[veh_id]      = ReplayBuffer()
            self.update_counts[veh_id] = 0

    # ── action selection (trust-gated, mirrors original logic) ──────────────
    def choose(self, veh_id, state, epsilon, trust):
        self.init(veh_id)

        # Low trust → forced exploration
        if trust < 0.5:
            return np.random.randint(ACTION_DIM)

        # High trust → pure exploitation via DQN
        if trust > 1.5:
            with torch.no_grad():
                q_vals = self.networks[veh_id](state_to_tensor(state))
            return int(torch.argmax(q_vals).item())

        # Normal range → epsilon-greedy
        if np.random.rand() < epsilon:
            return np.random.randint(ACTION_DIM)

        with torch.no_grad():
            q_vals = self.networks[veh_id](state_to_tensor(state))
        return int(torch.argmax(q_vals).item())

    # ── store transition and train if buffer is ready ────────────────────────
    def update(self, veh_id, s, a, r, ns, alpha=None, gamma=None):
        """
        alpha / gamma kept as params for API compatibility with old code,
        but DQN uses self.lr and self.gamma internally.
        """
        self.init(veh_id)

        # Store experience
        self.buffers[veh_id].push(
            state_to_tensor(s),
            a,
            r,
            state_to_tensor(ns)
        )

        # Only train once buffer has enough samples
        if len(self.buffers[veh_id]) < self.batch_size:
            return

        self._train(veh_id)

        # Periodically sync target network
        self.update_counts[veh_id] += 1
        if self.update_counts[veh_id] % self.target_update == 0:
            self.targets[veh_id].load_state_dict(
                self.networks[veh_id].state_dict()
            )

    # ── one gradient step ────────────────────────────────────────────────────
    def _train(self, veh_id):
        batch = self.buffers[veh_id].sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states      = torch.stack(states)
        next_states = torch.stack(next_states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)

        # Current Q values
        q_current = self.networks[veh_id](states)
        q_current = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (using target network)
        with torch.no_grad():
            q_next = self.targets[veh_id](next_states).max(1)[0]
            q_target = rewards + self.gamma * q_next

        loss = nn.MSELoss()(q_current, q_target)

        self.optimizers[veh_id].zero_grad()
        loss.backward()
        self.optimizers[veh_id].step()
