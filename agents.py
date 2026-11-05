"""
Module 2 - Double DQN agent (owner: Gomana Hossam Eldine).

The agent is the "robot":
  1. QNetwork      : the robot's brain. 8 numbers in (the patient), 5 scores out (one per UF rate).
  2. ReplayBuffer  : the robot's memory box. Stores past steps so it can learn from random ones.
  3. DDQNAgent     : picks actions (act), remembers (store) and learns (learn).

`double=True`  -> Double DQN (our proposed model).
`double=False` -> standard DQN (baseline for Module 3, same code, one flag).
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import STATE_DIM, N_ACTIONS   # 8 and 5, from the shared contract


# ---------------------------------------------------------------------------
# 1. The brain: a small neural network 8 -> 64 -> 64 -> 5
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Input: normalized patient state (8). Output: Q-value (expected future points) of each UF rate (5)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. The memory box
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Stores (state, action, reward, next_state, done) and returns random batches."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)     # old memories fall out when full

    def push(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# 3. The agent
# ---------------------------------------------------------------------------
class DDQNAgent:
    def __init__(self, double=True, gamma=0.99, lr=1e-3, batch_size=64,
                 buffer_size=50_000, target_update=200, reward_scale=0.1):
        self.double = double
        self.gamma = gamma                  # how much future points matter (0..1)
        self.batch_size = batch_size
        self.target_update = target_update  # copy online -> target every N learning steps
        self.reward_scale = reward_scale    # shrink big rewards (-10 hypotension) so learning is stable

        self.online = QNetwork()            # the brain we train
        self.target = QNetwork()            # slow copy used to score (this is the "second brain")
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    # --- choose an action ---------------------------------------------------
    def act(self, state, epsilon=0.0):
        """epsilon-greedy: with probability epsilon try a random UF rate (explore), else the best one."""
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(1).item())

    # --- remember -----------------------------------------------------------
    def store(self, s, a, r, s2, done):
        self.buffer.push(s, a, r * self.reward_scale, s2, float(done))

    # --- learn ----------------------------------------------------------------
    def learn(self):
        """One learning step on a random batch of memories. Returns the loss (or None if too few memories)."""
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, s2, done = self.buffer.sample(self.batch_size)

        # What the brain currently believes: Q(s, a) for the action we actually took
        q_now = self.online(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                # DOUBLE DQN: online net CHOOSES the best next action ...
                best_next = self.online(s2).argmax(1, keepdim=True)
                # ... but the target net SCORES it (two brains -> no over-optimism)
                q_next = self.target(s2).gather(1, best_next)
            else:
                # standard DQN: the target net both chooses and scores
                q_next = self.target(s2).max(1, keepdim=True)[0]
            # target = reward now + discounted future (no future if the session ended)
            q_goal = r + self.gamma * q_next * (1.0 - done)

        # move the brain's belief toward the target (Huber loss = robust to big errors)
        loss = F.smooth_l1_loss(q_now, q_goal)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        # every N steps, refresh the slow copy
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # --- save ---------------------------------------------------------------
    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def export_scripted(self, path):
        """TorchScript export that Module 4 (Integrated Gradients) loads. Input 8 -> output 5."""
        self.online.eval()
        traced = torch.jit.trace(self.online.cpu(), torch.zeros(1, STATE_DIM))
        traced.save(str(path))
