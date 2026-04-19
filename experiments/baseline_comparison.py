"""
baseline_comparison.py — compare all policies on every trace and cache size.
Produces the data needed for Figure 1 and Figure 3.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import evaluate_all

if __name__ == "__main__":
    evaluate_all(
        traces=[
            "traces/synthetic/mixed_phase.trace",
            "traces/synthetic/phase_shift_abrupt.trace",
        ],
        cache_sizes=[64, 128, 256, 512],
        model_path="models/dqn_parl.pt",
        output_dir="results",
    )
