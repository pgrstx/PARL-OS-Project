"""
phase_transition_test.py — test adaptation speed around phase boundaries.

Produces data for Figure 5 (adaptation latency box plot).
Specifically tests on phase_shift_abrupt.trace where transitions are sudden.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import evaluate_all

if __name__ == "__main__":
    evaluate_all(
        traces=["traces/synthetic/phase_shift_abrupt.trace"],
        cache_sizes=[256],
        model_path="models/dqn_parl.pt",
        output_dir="results",
    )
    from results.plots import figure5_adaptation_latency
    import numpy as np
    figure5_adaptation_latency(
        rolling_npz="results/rolling_miss_rates.npz",
        labels_file="traces/synthetic/phase_shift_abrupt.labels",
        trace_name="phase_shift_abrupt",
        cache_sizes=[256],
        output_path="results/fig5_abrupt.png",
    )
    print("Done — see results/fig5_abrupt.png")
