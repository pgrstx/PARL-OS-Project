"""
Reward function for the PARL DQN agent.

Plain English:
  The reward tells the AI whether its last decision was good or bad.
  We define "good" as: performing better than LRU (our baseline).
    - If PARL's hit rate > LRU's hit rate → positive reward (we improved!)
    - If PARL's hit rate < LRU's hit rate → negative reward (we did worse)
    - Small penalty for switching policies (to avoid frantic switching)

  The shadow LRU cache runs in parallel on the same trace but always uses
  LRU policy — it's our reference point. PARL needs to beat this reference
  to get a positive reward.

  Reward range: approximately [-1.01, 1.0]
  (no extra normalization needed — values are naturally bounded)
"""


def compute_reward(
    current_hit_rate: float,
    baseline_lru_hit_rate: float,
    switched_policy: bool = False,
    policy_switch_cost: float = 0.01,
) -> float:
    """
    Compute the reward for the RL agent.

    Args:
        current_hit_rate:      PARL's hit rate over the last 100 accesses.
        baseline_lru_hit_rate: Shadow LRU's hit rate over the last 100 accesses.
        switched_policy:       True if the agent changed policy this step.
        policy_switch_cost:    Penalty for unnecessary switching.

    Returns:
        Scalar reward in approximately [-1.01, 1.0].
    """
    improvement = current_hit_rate - baseline_lru_hit_rate
    switch_penalty = policy_switch_cost if switched_policy else 0.0
    reward = improvement - switch_penalty

    # Clip to prevent extreme values during early training instability
    return float(max(-1.0, min(1.0, reward)))
