"""
ReplayBuffer — stores past experiences for DQN training.

Plain English:
  Instead of learning only from the most recent experience (which would make
  the AI forget old lessons quickly), we store the last 50,000 experiences
  in memory and train on random batches from this pool.

  This is called "Experience Replay" and it's a key technique that makes
  DQN stable. It breaks the correlation between consecutive samples
  (which would otherwise bias the learning).

  Each experience is: (state, action, reward, next_state, done)
    - state: the situation we were in
    - action: which policy we chose
    - reward: how much better/worse we did vs. LRU baseline
    - next_state: the situation after the action
    - done: whether the episode ended (always False here)
"""

import random
import numpy as np
import torch
from collections import deque
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self._buffer: deque = deque(maxlen=capacity)
        self._capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition."""
        self._buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, ...]:
        """Sample a random mini-batch. Returns tensors ready for PyTorch."""
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
