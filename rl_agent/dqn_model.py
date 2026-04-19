"""
DQN Network — the neural network at the heart of our RL agent.

Plain English:
  This is like the "brain" that, given a description of the current memory
  access situation (11 numbers), outputs 5 scores — one for each policy.
  The policy with the highest score is selected.

  It's a simple 3-layer feedforward network (like a multi-layer perceptron).
  The network is trained to output Q-values: Q(state, action) = "how good is
  it to choose this policy in this situation".

  Architecture:
    Input (11)  →  Hidden (256)  →  Hidden (256)  →  Output (5)
    with ReLU activations between layers.

  State vector (11 dimensions):
    [0:8]  phase_embedding  — 8D vector describing current workload phase
    [8]    miss_rate         — recent miss rate (0 to 1)
    [9]    cache_fullness    — fraction of cache occupied (0 to 1)
    [10]   steps_since_switch — normalized steps since last policy change

  Output (5 Q-values for each policy):
    0=LRU, 1=CLOCK, 2=LFU, 3=ARC, 4=HYBRID
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int = 11, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
