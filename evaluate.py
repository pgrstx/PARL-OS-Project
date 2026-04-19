"""
evaluate.py — benchmarking all policies head-to-head.

Plain English:
  This script runs every algorithm (LRU, CLOCK, LFU, ARC, LIRS, OPT, and PARL)
  on every trace at every cache size, records their miss rates, and saves results
  to a CSV file. Think of it as a "tournament" between all the competitors.

  The output is used to generate all 5 publication-quality figures.
"""

import os
import sys
import csv
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator.cache import Cache
from simulator.trace_player import TracePlayer
from simulator.metrics import HitRateTracker
from policies.lru import LRUPolicy
from policies.clock import CLOCKPolicy
from policies.lfu import LFUPolicy
from policies.arc import ARCPolicy
from policies.lirs import LIRSPolicy
from policies.opt import OPTPolicy
from policies import POLICY_CLASSES, POLICY_NAMES
from phase_detection.phase_classifier import PhaseClassifier
from rl_agent.agent import DQNAgent
from rl_agent.reward import compute_reward
from train_agent import build_state


def run_policy(policy, trace: list[int], cache_size: int,
               labels: list[int] = None) -> dict:
    """Run a single policy on a trace, return detailed statistics."""
    cache = Cache(cache_size, policy)
    tracker = HitRateTracker(window_size=500)

    rolling_miss_rates = []
    per_phase_stats = {i: {"hits": 0, "total": 0} for i in range(5)}

    start_time = time.time()
    for i, page_id in enumerate(trace):
        is_hit = cache.access(page_id)
        tracker.record(is_hit)
        rolling_miss_rates.append(tracker.rolling_miss_rate)

        if labels and i < len(labels):
            phase = labels[i]
            per_phase_stats[phase]["total"] += 1
            per_phase_stats[phase]["hits"] += int(is_hit)

    elapsed = time.time() - start_time
    overall_miss_rate = cache.miss_rate

    # Per-phase miss rates
    phase_miss_rates = {}
    for phase_id, stats in per_phase_stats.items():
        if stats["total"] > 0:
            phase_miss_rates[phase_id] = 1.0 - stats["hits"] / stats["total"]
        else:
            phase_miss_rates[phase_id] = None

    return {
        "miss_rate": overall_miss_rate,
        "hit_rate": cache.hit_rate,
        "phase_miss_rates": phase_miss_rates,
        "rolling_miss_rates": rolling_miss_rates,
        "runtime_sec": elapsed,
        "stats": cache.get_stats(),
    }


def run_parl(model_path: str, trace: list[int], cache_size: int,
             labels: list[int] = None) -> dict:
    """Run the trained PARL agent on a trace."""
    agent = DQNAgent(state_dim=11, action_dim=5)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent._epsilon = 0.0  # pure exploitation during evaluation
    else:
        print(f"  [WARN] No model at {model_path} — using untrained agent")

    parl_policy = LRUPolicy()
    parl_cache = Cache(cache_size, parl_policy)
    shadow_cache = Cache(cache_size, LRUPolicy())
    phase_classifier = PhaseClassifier(n_phases=5, window_size=1000)
    tracker = HitRateTracker(window_size=500)

    rolling_miss_rates = []
    per_phase_stats = {i: {"hits": 0, "total": 0} for i in range(5)}
    current_action = 0
    steps_since_switch = 0

    start_time = time.time()
    for i, page_id in enumerate(trace):
        shadow_cache.access(page_id)
        is_hit = parl_cache.access(page_id)
        phase_classifier.update(page_id, is_hit)
        tracker.record(is_hit)
        rolling_miss_rates.append(tracker.rolling_miss_rate)
        steps_since_switch += 1

        if i > 0 and i % 100 == 0:
            state = build_state(
                phase_classifier.phase_embedding,
                parl_cache.miss_rate_recent,
                parl_cache.cache_fullness,
                steps_since_switch,
            )
            action = agent.select_action(state)
            if action != current_action:
                new_policy = POLICY_CLASSES[action]()
                if action == 3:
                    new_policy = ARCPolicy(capacity=cache_size)
                parl_cache.set_policy(new_policy)
                current_action = action
                steps_since_switch = 0

        if labels and i < len(labels):
            phase = labels[i]
            per_phase_stats[phase]["total"] += 1
            per_phase_stats[phase]["hits"] += int(is_hit)

    elapsed = time.time() - start_time
    phase_miss_rates = {}
    for phase_id, stats in per_phase_stats.items():
        if stats["total"] > 0:
            phase_miss_rates[phase_id] = 1.0 - stats["hits"] / stats["total"]
        else:
            phase_miss_rates[phase_id] = None

    return {
        "miss_rate": parl_cache.miss_rate,
        "hit_rate": parl_cache.hit_rate,
        "phase_miss_rates": phase_miss_rates,
        "rolling_miss_rates": rolling_miss_rates,
        "runtime_sec": elapsed,
        "stats": parl_cache.get_stats(),
    }


def evaluate_all(
    traces: list[str],
    cache_sizes: list[int] = (64, 128, 256, 512),
    model_path: str = "models/dqn_parl.pt",
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    rolling_path = os.path.join(output_dir, "rolling_miss_rates.npz")

    all_rows = []
    rolling_data = {}   # policy → trace → cache_size → array

    for trace_path in traces:
        if not os.path.exists(trace_path):
            print(f"Skipping missing trace: {trace_path}")
            continue

        player = TracePlayer(trace_path)
        trace = player.trace
        labels = player.labels
        trace_name = os.path.basename(trace_path).replace(".trace", "")
        print(f"\n=== Trace: {trace_name} ({len(trace)} accesses) ===")

        # Pre-scan OPT (needs full trace)
        opt_policy = OPTPolicy()
        opt_policy.prescan(trace)

        for cache_size in cache_sizes:
            print(f"  Cache size: {cache_size}")

            policies = {
                "LRU":   LRUPolicy(),
                "CLOCK": CLOCKPolicy(),
                "LFU":   LFUPolicy(),
                "ARC":   ARCPolicy(capacity=cache_size),
                "LIRS":  LIRSPolicy(capacity=cache_size),
                "OPT":   OPTPolicy(),
            }
            # Re-prescan OPT fresh for each cache_size run
            for name, pol in policies.items():
                if name == "OPT":
                    pol.prescan(trace)

            for policy_name, policy in policies.items():
                print(f"    Running {policy_name}...", end="", flush=True)
                result = run_policy(policy, trace, cache_size, labels)
                print(f" miss_rate={result['miss_rate']:.4f}  ({result['runtime_sec']:.2f}s)")

                key = (policy_name, trace_name, cache_size)
                rolling_data[key] = result["rolling_miss_rates"]

                for phase_id, pmr in result["phase_miss_rates"].items():
                    all_rows.append({
                        "policy": policy_name,
                        "trace": trace_name,
                        "cache_size": cache_size,
                        "miss_rate": result["miss_rate"],
                        "phase": phase_id,
                        "phase_miss_rate": pmr if pmr is not None else "",
                        "runtime_sec": result["runtime_sec"],
                    })

            # Run PARL
            print(f"    Running PARL...", end="", flush=True)
            parl_result = run_parl(model_path, trace, cache_size, labels)
            print(f" miss_rate={parl_result['miss_rate']:.4f}  ({parl_result['runtime_sec']:.2f}s)")
            key = ("PARL", trace_name, cache_size)
            rolling_data[key] = parl_result["rolling_miss_rates"]

            for phase_id, pmr in parl_result["phase_miss_rates"].items():
                all_rows.append({
                    "policy": "PARL",
                    "trace": trace_name,
                    "cache_size": cache_size,
                    "miss_rate": parl_result["miss_rate"],
                    "phase": phase_id,
                    "phase_miss_rate": pmr if pmr is not None else "",
                    "runtime_sec": parl_result["runtime_sec"],
                })

    # Write CSV
    with open(results_path, "w", newline="") as f:
        fields = ["policy", "trace", "cache_size", "miss_rate", "phase",
                  "phase_miss_rate", "runtime_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    # Save rolling miss rate arrays for plotting
    save_dict = {}
    for (policy, trace_name, cs), arr in rolling_data.items():
        save_dict[f"{policy}__{trace_name}__{cs}"] = np.array(arr)
    np.savez(rolling_path, **save_dict)

    print(f"\nResults saved → {results_path}")
    print(f"Rolling data  → {rolling_path}")
    return all_rows, rolling_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all policies")
    parser.add_argument("--traces", nargs="+",
                        default=["traces/synthetic/mixed_phase.trace",
                                 "traces/synthetic/phase_shift_abrupt.trace"])
    parser.add_argument("--cache_sizes", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--model_path", default="models/dqn_parl.pt")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    evaluate_all(
        traces=args.traces,
        cache_sizes=args.cache_sizes,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
