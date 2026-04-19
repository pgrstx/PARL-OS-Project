"""
train_agent.py — Training loop for PARL.

Supports two modes:
  --mode dqn_direct  (default, NEW):
    The neural network directly scores every page in the cache and evicts
    the one with the lowest score. The network trains online from its own
    eviction mistakes (was the evicted page accessed again soon?).

  --mode policy_select (original):
    The neural network picks which classical policy (LRU/LFU/ARC/...) to
    use at each step. The classical policy then decides which page to evict.

Both modes compete against a shadow LRU running on the same trace.
"""

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.cache import Cache
from simulator.trace_player import TracePlayer
from policies.lru import LRUPolicy
from policies.clock import CLOCKPolicy
from policies.lfu import LFUPolicy
from policies.arc import ARCPolicy
from policies.hybrid import HybridPolicy
from policies.dqn_policy import DQNEvictionPolicy
from policies import POLICY_CLASSES, POLICY_NAMES
from phase_detection.phase_classifier import PhaseClassifier
from rl_agent.agent import DQNAgent
from rl_agent.reward import compute_reward


def build_state(phase_embedding: np.ndarray, miss_rate: float,
                cache_fullness: float, steps_since_switch: int,
                max_steps: int = 1000) -> np.ndarray:
    normalized_steps = min(steps_since_switch / max_steps, 1.0)
    return np.concatenate([
        phase_embedding,
        [miss_rate],
        [cache_fullness],
        [normalized_steps],
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: DQN-DIRECT (neural network picks which page to evict)
# ─────────────────────────────────────────────────────────────────────────────

def train_dqn_direct(
    trace_path: str,
    cache_size: int = 256,
    max_steps: int = 500_000,
    output_dir: str = "results",
    model_dir: str = "models",
    log_interval: int = 1000,
):
    """
    Train the DQN-direct eviction policy.

    The DQNEvictionPolicy trains itself ONLINE as it runs — every eviction
    is logged, and after LOOKAHEAD steps we check if the evicted page was
    accessed again and use that as the training signal.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading trace: {trace_path}")
    player = TracePlayer(trace_path)
    trace = player.trace[:max_steps]
    n = len(trace)
    print(f"Trace length: {n} accesses")

    # PARL cache: uses DQN-direct eviction
    dqn_policy = DQNEvictionPolicy(capacity=cache_size)
    parl_cache = Cache(cache_size, dqn_policy)

    # Shadow cache: always uses LRU (our baseline)
    shadow_cache = Cache(cache_size, LRUPolicy())

    # Phase detection (feeds phase_id to DQN policy so it can use phase features)
    phase_clf = PhaseClassifier(n_phases=5, window_size=1000)

    log_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "miss_rate_parl", "miss_rate_lru",
                     "hit_rate_delta", "eviction_count"])

    print("Starting DQN-direct training...")
    for step, page_id in enumerate(trace):
        # Keep phase classifier informed
        is_hit_shadow = shadow_cache.access(page_id)
        phase_clf.update(page_id, is_hit_shadow)
        dqn_policy.set_phase(phase_clf.phase_id)

        parl_cache.access(page_id)

        if step > 0 and step % log_interval == 0:
            delta = parl_cache.hit_rate_recent - shadow_cache.hit_rate_recent
            writer.writerow([
                step,
                f"{parl_cache.miss_rate:.4f}",
                f"{shadow_cache.miss_rate:.4f}",
                f"{delta:.4f}",
                dqn_policy._eviction_count,
            ])
            log_file.flush()

        if step % 10_000 == 0 and step > 0:
            delta = parl_cache.hit_rate_recent - shadow_cache.hit_rate_recent
            print(f"  Step {step:6d}/{n} | "
                  f"DQN-direct miss: {parl_cache.miss_rate:.3f} | "
                  f"LRU miss: {shadow_cache.miss_rate:.3f} | "
                  f"Delta: {delta:+.3f} | "
                  f"Phase: {phase_clf.phase_id}")

    log_file.close()

    model_path = os.path.join(model_dir, "dqn_direct.pt")
    dqn_policy.save(model_path)
    print(f"\nDQN-direct training complete!")
    print(f"  Model → {model_path}")
    print(f"  Final DQN-direct miss rate: {parl_cache.miss_rate:.4f}")
    print(f"  Final LRU miss rate:        {shadow_cache.miss_rate:.4f}")
    improvement = shadow_cache.miss_rate - parl_cache.miss_rate
    print(f"  Improvement over LRU:       {improvement:+.4f}")
    return dqn_policy


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: POLICY-SELECT (DQN picks which classical policy to use)
# ─────────────────────────────────────────────────────────────────────────────

def train_policy_select(
    trace_path: str,
    cache_size: int = 256,
    max_steps: int = 500_000,
    decision_interval: int = 100,
    output_dir: str = "results",
    model_dir: str = "models",
    log_interval: int = 1000,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading trace: {trace_path}")
    player = TracePlayer(trace_path)
    trace = player.trace[:max_steps]
    n = len(trace)
    print(f"Trace length: {n}")

    parl_cache = Cache(cache_size, LRUPolicy())
    shadow_cache = Cache(cache_size, LRUPolicy())
    phase_clf = PhaseClassifier(n_phases=5, window_size=1000)
    agent = DQNAgent(state_dim=11, action_dim=5)

    log_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "miss_rate_parl", "miss_rate_lru", "reward",
                     "loss", "epsilon", "phase_id", "policy_chosen"])

    current_action = 0
    steps_since_switch = 0
    prev_state = None
    prev_action = None
    reward_ema = 0.0

    print("Starting policy-select training...")
    for step, page_id in enumerate(trace):
        shadow_cache.access(page_id)
        parl_hit = parl_cache.access(page_id)
        phase_clf.update(page_id, parl_hit)
        steps_since_switch += 1

        if step > 0 and step % decision_interval == 0:
            state = build_state(
                phase_clf.phase_embedding,
                parl_cache.miss_rate_recent,
                parl_cache.cache_fullness,
                steps_since_switch,
            )
            action = agent.select_action(state)
            switched = (action != current_action)
            current_action = action

            if switched:
                new_policy = POLICY_CLASSES[action]()
                if action == 3:
                    new_policy = ARCPolicy(capacity=cache_size)
                parl_cache.set_policy(new_policy)
                steps_since_switch = 0

            reward = compute_reward(
                parl_cache.hit_rate_recent,
                shadow_cache.hit_rate_recent,
                switched_policy=switched,
            )
            reward_ema = 0.95 * reward_ema + 0.05 * reward

            if prev_state is not None:
                agent.push(prev_state, prev_action, reward, state, done=False)

            loss = agent.train_step()
            agent.decay_epsilon()
            prev_state = state
            prev_action = action

            if step % log_interval == 0:
                writer.writerow([
                    step,
                    f"{parl_cache.miss_rate:.4f}",
                    f"{shadow_cache.miss_rate:.4f}",
                    f"{reward_ema:.4f}",
                    f"{loss:.6f}" if loss else "0",
                    f"{agent.epsilon:.4f}",
                    phase_clf.phase_id,
                    POLICY_NAMES[action],
                ])
                log_file.flush()

        if step % 10_000 == 0 and step > 0:
            print(f"  Step {step:6d}/{n} | PARL: {parl_cache.miss_rate:.3f} | "
                  f"LRU: {shadow_cache.miss_rate:.3f} | ε={agent.epsilon:.3f} | "
                  f"Phase={phase_clf.phase_id} | Policy={POLICY_NAMES[current_action]}")

    log_file.close()
    model_path = os.path.join(model_dir, "dqn_parl.pt")
    agent.save(model_path)
    print(f"\nPolicy-select training complete! Model → {model_path}")
    print(f"  PARL miss: {parl_cache.miss_rate:.4f} | LRU miss: {shadow_cache.miss_rate:.4f}")
    return agent


def train(trace_path, cache_size=256, max_steps=500_000,
          output_dir="results", model_dir="models", mode="dqn_direct"):
    if mode == "dqn_direct":
        return train_dqn_direct(trace_path, cache_size, max_steps, output_dir, model_dir)
    else:
        return train_policy_select(trace_path, cache_size, max_steps,
                                   output_dir=output_dir, model_dir=model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="traces/synthetic/mixed_phase.trace")
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--mode", default="dqn_direct",
                        choices=["dqn_direct", "policy_select"])
    args = parser.parse_args()

    train(args.trace, args.cache_size, args.max_steps,
          args.output_dir, args.model_dir, args.mode)
