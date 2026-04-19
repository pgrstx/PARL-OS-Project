"""
ablation_study.py — compare PARL with and without phase detection.

Tests two configurations:
  1. PARL_full:    DQN agent with phase_embedding in state (full system)
  2. PARL_no_phase: DQN agent WITHOUT phase_embedding (state only has miss_rate etc.)

This demonstrates the contribution of the phase detection module.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cache import Cache
from simulator.trace_player import TracePlayer
from policies.lru import LRUPolicy
from policies.arc import ARCPolicy
from policies import POLICY_CLASSES
from phase_detection.phase_classifier import PhaseClassifier
from rl_agent.agent import DQNAgent
from rl_agent.reward import compute_reward


def run_ablation(trace_path: str, cache_size: int = 256, max_steps: int = 100_000):
    player = TracePlayer(trace_path)
    trace = player.trace[:max_steps]
    n = len(trace)

    results = {}
    for use_phase in [True, False]:
        label = "PARL_full" if use_phase else "PARL_no_phase"
        state_dim = 11 if use_phase else 3  # phase_emb(8) + miss + full + switch vs just 3

        parl_cache = Cache(cache_size, LRUPolicy())
        shadow_cache = Cache(cache_size, LRUPolicy())
        phase_clf = PhaseClassifier(n_phases=5)
        agent = DQNAgent(state_dim=state_dim, action_dim=5)

        current_action = 0
        steps_since_switch = 0

        for i, page_id in enumerate(trace):
            shadow_cache.access(page_id)
            is_hit = parl_cache.access(page_id)
            phase_clf.update(page_id, is_hit)
            steps_since_switch += 1

            if i > 0 and i % 100 == 0:
                if use_phase:
                    state = np.concatenate([
                        phase_clf.phase_embedding,
                        [parl_cache.miss_rate_recent],
                        [parl_cache.cache_fullness],
                        [min(steps_since_switch / 1000, 1.0)],
                    ]).astype(np.float32)
                else:
                    state = np.array([
                        parl_cache.miss_rate_recent,
                        parl_cache.cache_fullness,
                        min(steps_since_switch / 1000, 1.0),
                    ], dtype=np.float32)

                action = agent.select_action(state)
                if action != current_action:
                    new_policy = POLICY_CLASSES[action]()
                    if action == 3:
                        new_policy = ARCPolicy(capacity=cache_size)
                    parl_cache.set_policy(new_policy)
                    current_action = action
                    steps_since_switch = 0

                reward = compute_reward(
                    parl_cache.hit_rate_recent,
                    shadow_cache.hit_rate_recent,
                    switched_policy=(action != current_action),
                )
                agent.push(state, action, reward, state, False)
                agent.train_step()
                agent.decay_epsilon()

        results[label] = parl_cache.miss_rate
        print(f"  {label:<20} miss_rate={parl_cache.miss_rate:.4f}")

    return results


if __name__ == "__main__":
    print("Running ablation study...")
    results = run_ablation("traces/synthetic/mixed_phase.trace")
    improvement = results["PARL_no_phase"] - results["PARL_full"]
    print(f"\nPhase detection contribution: {improvement:+.4f} miss rate improvement")
