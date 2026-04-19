"""
DQNAgent — the core RL agent that learns to select cache replacement policies.

Plain English:
  This agent uses the DQN (Deep Q-Network) algorithm — the same technique
  that DeepMind used to train Atari-playing AIs (it was published in Nature 2015).

  The learning loop:
    1. Look at the current "state" (phase + miss rate + cache stats)
    2. Pick a policy (epsilon-greedy: explore randomly OR exploit learned knowledge)
    3. Run the policy, observe the reward
    4. Store the experience in replay buffer
    5. Train on a random batch from buffer
    6. Periodically copy "policy net" to "target net" (for stable training)

  Epsilon decay: at the start, the agent explores randomly (epsilon=1.0).
  Over time, epsilon decreases toward 0.05 — it becomes more confident in
  its learned values and exploits less often.

  Two networks (policy_net + target_net):
    This is crucial for stability. target_net is updated every 500 steps
    (not every step), which prevents the learning target from "chasing itself"
    (a common DQN training instability).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from .dqn_model import DQNNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        min_buffer_size: int = 1_000,
        target_update_freq: int = 500,
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._batch_size = batch_size
        self._min_buffer_size = min_buffer_size
        self._target_update_freq = target_update_freq

        # Device: use GPU if available, else CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Two networks: policy (trained every step) and target (updated periodically)
        self._policy_net = DQNNetwork(state_dim, action_dim).to(self._device)
        self._target_net = DQNNetwork(state_dim, action_dim).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()
        self._replay_buffer = ReplayBuffer(buffer_capacity)

        self._step_count = 0
        self._train_step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        With probability epsilon: random action (exploration).
        Otherwise: greedy action from Q-network (exploitation).
        """
        if np.random.random() < self._epsilon:
            return np.random.randint(self._action_dim)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            q_values = self._policy_net(state_t)
        return int(q_values.argmax().item())

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self._replay_buffer.push(state, action, reward, next_state, done)
        self._step_count += 1

    def train_step(self) -> Optional[float]:
        """
        Sample a mini-batch and do one gradient descent step.
        Returns the loss value, or None if buffer is too small.
        """
        if len(self._replay_buffer) < self._min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = \
            self._replay_buffer.sample(self._batch_size)

        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        # Current Q values: Q(s, a)
        current_q = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self._target_net(next_states).max(1)[0]
            target_q = rewards + self._gamma * max_next_q * (1.0 - dones)

        loss = self._loss_fn(current_q, target_q)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm=10.0)
        self._optimizer.step()

        self._train_step_count += 1

        # Periodic target network update
        if self._train_step_count % self._target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        """Hard copy: copy policy_net weights → target_net."""
        self._target_net.load_state_dict(self._policy_net.state_dict())

    def decay_epsilon(self):
        """Decay epsilon toward epsilon_end."""
        self._epsilon = max(self._epsilon_end, self._epsilon * self._epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def buffer_size(self) -> int:
        return len(self._replay_buffer)

    def save(self, path: str):
        torch.save({
            "policy_net": self._policy_net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "epsilon": self._epsilon,
            "step_count": self._step_count,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self._device)
        self._policy_net.load_state_dict(checkpoint["policy_net"])
        self._target_net.load_state_dict(checkpoint["target_net"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint.get("epsilon", self._epsilon_end)
        self._step_count = checkpoint.get("step_count", 0)
