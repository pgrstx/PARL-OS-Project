"""
train_agent.py — RL training loop for PARL.

Plain English:
  This script runs the AI "to school" — it feeds it thousands of page accesses,
  lets it try different policies, rewards/punishes it based on performance,
  and gradually the AI gets better at picking the right policy.

  The training process:
    1. Load a trace file (the "curriculum")
    2. Create two caches: one PARL (our AI), one shadow LRU (baseline)
    3. Feed accesses to both caches simultaneously
    4. Every 100 accesses, the AI decides which policy to use
    5. Reward = (PARL hit rate) - (LRU hit rate) — are we beating the baseline?
    6. The AI trains on batches of past experiences (experience replay)
    7. Logs are saved to results/training_log.csv for plotting

  Training runs for the full trace (or max 500,000 steps).
  Final model saved to models/dqn_parl.pt
"""

import os
import sys
import csv
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.cache import Cache
from simulator.trace_player import TracePlayer
from policies.lru import LRUPolicy
from policies.clock import CLOCKPolicy
from policies.lfu import LFUPolicy
from policies.arc import ARCPolicy
from policies.hybrid import HybridPolicy
from policies import POLICY_CLASSES, POLICY_NAMES
from phase_detection.phase_classifier import PhaseClassifier
from rl_agent.agent import DQNAgent
from rl_agent.reward import compute_reward


def build_state(phase_embedding: np.ndarray, miss_rate: float,
                cache_fullness: float, steps_since_switch: int,
                max_steps: int = 1000) -> np.ndarray:
    """Assemble the 11-dimensional state vector for the RL agent."""
    normalized_steps = min(steps_since_switch / max_steps, 1.0)
    state = np.concatenate([
        phase_embedding,                      # 8 dims
        [miss_rate],                          # 1 dim
        [cache_fullness],                     # 1 dim
        [normalized_steps],                   # 1 dim
    ]).astype(np.float32)
    return state


def train(
    trace_path: str,
    cache_size: int = 256,
    max_steps: int = 500_000,
    decision_interval: int = 100,  # how often the agent picks a policy
    output_dir: str = "results",
    model_dir: str = "models",
    log_interval: int = 1000,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading trace: {trace_path}")
    player = TracePlayer(trace_path)
    trace = player.trace
    n = min(len(trace), max_steps)
    print(f"Trace length: {len(trace)} accesses | Training on first {n}")

    # Initialize caches
    parl_policy = LRUPolicy()
    parl_cache = Cache(cache_size, parl_policy)
    shadow_policy = LRUPolicy()
    shadow_cache = Cache(cache_size, shadow_policy)

    # Initialize ARC with correct capacity
    arc_policy = ARCPolicy(capacity=cache_size)

    # Policy pool (instantiated once to avoid repeated construction)
    policy_pool = {
        0: LRUPolicy(),
        1: CLOCKPolicy(),
        2: LFUPolicy(),
        3: ARCPolicy(capacity=cache_size),
        4: HybridPolicy(),
    }

    # Phase detection
    phase_classifier = PhaseClassifier(n_phases=5, window_size=1000)

    # RL agent
    agent = DQNAgent(state_dim=11, action_dim=5)

    # Logging
    log_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "miss_rate_parl", "miss_rate_lru", "reward",
                     "loss", "epsilon", "phase_id", "policy_chosen"])

    # Training state
    current_action = 0
    steps_since_switch = 0
    prev_state = None
    prev_action = None
    total_reward = 0.0
    recent_losses = []
    reward_ema = 0.0
    ema_alpha = 0.05  # EMA smoothing factor

    print("Starting training...")
    for step in range(n):
        page_id = trace[step]

        # Both caches process the same access
        shadow_hit = shadow_cache.access(page_id)
        parl_hit = parl_cache.access(page_id)

        # Update phase classifier
        phase_classifier.update(page_id, parl_hit)

        steps_since_switch += 1

        # Every decision_interval accesses, agent picks a new policy
        if step > 0 and step % decision_interval == 0:
            phase_emb = phase_classifier.phase_embedding
            miss_rate = parl_cache.miss_rate_recent
            cache_full = parl_cache.cache_fullness
            state = build_state(phase_emb, miss_rate, cache_full, steps_since_switch)

            action = agent.select_action(state)
            switched = (action != current_action)
            current_action = action

            if switched:
                # Switch policy: instantiate fresh policy, inject current cache state
                new_policy = POLICY_CLASSES[action]()
                if action == 3:  # ARC needs capacity
                    new_policy = ARCPolicy(capacity=cache_size)
                parl_cache.set_policy(new_policy)
                steps_since_switch = 0

            # Compute reward
            reward = compute_reward(
                parl_cache.hit_rate_recent,
                shadow_cache.hit_rate_recent,
                switched_policy=switched,
            )
            reward_ema = (1 - ema_alpha) * reward_ema + ema_alpha * reward
            total_reward += reward

            # Store experience
            if prev_state is not None:
                agent.push(prev_state, prev_action, reward, state, done=False)

            # Train
            loss = agent.train_step()
            agent.decay_epsilon()
            if loss is not None:
                recent_losses.append(loss)

            prev_state = state
            prev_action = action

            # Logging
            if step % log_interval == 0:
                avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
                recent_losses.clear()
                writer.writerow([
                    step,
                    f"{parl_cache.miss_rate:.4f}",
                    f"{shadow_cache.miss_rate:.4f}",
                    f"{reward_ema:.4f}",
                    f"{avg_loss:.6f}",
                    f"{agent.epsilon:.4f}",
                    phase_classifier.phase_id,
                    POLICY_NAMES[action],
                ])
                log_file.flush()

        # Progress bar every 10000 steps
        if step % 10_000 == 0 and step > 0:
            print(f"  Step {step:6d}/{n} | "
                  f"PARL miss rate: {parl_cache.miss_rate:.3f} | "
                  f"LRU miss rate:  {shadow_cache.miss_rate:.3f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"Phase={phase_classifier.phase_id} | "
                  f"Policy={POLICY_NAMES[current_action]}")

    log_file.close()

    # Save model
    model_path = os.path.join(model_dir, "dqn_parl.pt")
    agent.save(model_path)
    print(f"\nTraining complete!")
    print(f"  Model saved → {model_path}")
    print(f"  Training log → {log_path}")
    print(f"  Final PARL miss rate:  {parl_cache.miss_rate:.4f}")
    print(f"  Final LRU  miss rate:  {shadow_cache.miss_rate:.4f}")
    print(f"  Improvement:           {shadow_cache.miss_rate - parl_cache.miss_rate:+.4f}")

    return agent, parl_cache.miss_rate, shadow_cache.miss_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PARL DQN agent")
    parser.add_argument("--trace", default="traces/synthetic/mixed_phase.trace")
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    train(
        trace_path=args.trace,
        cache_size=args.cache_size,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
