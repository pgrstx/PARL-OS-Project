"""
run_experiment.py — main CLI entry point for the PARL project.

Usage:
  python run_experiment.py --mode generate_traces
  python run_experiment.py --mode train  [--train_mode dqn_direct|policy_select]
  python run_experiment.py --mode eval
  python run_experiment.py --mode compare
  python run_experiment.py --mode plots
  python run_experiment.py --mode monitor    ← LIVE macOS/Linux dashboard
  python run_experiment.py --mode all        (generate → train → eval → plots)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mode_generate_traces():
    print("=" * 60)
    print("STEP 1: Generating synthetic traces")
    print("=" * 60)
    import subprocess
    subprocess.run([sys.executable,
                    os.path.join("traces", "synthetic", "generate_traces.py")],
                   check=True)


def mode_train(args):
    print("=" * 60)
    train_mode = getattr(args, "train_mode", "dqn_direct")
    print(f"STEP 2: Training PARL — mode={train_mode}")
    print("=" * 60)
    from train_agent import train
    train(
        trace_path=args.trace,
        cache_size=args.cache_size,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        mode=train_mode,
    )


def mode_eval(args):
    print("=" * 60)
    print("STEP 3: Evaluating all policies")
    print("=" * 60)
    from evaluate import evaluate_all
    traces = [args.trace] if args.trace else [
        "traces/synthetic/mixed_phase.trace",
        "traces/synthetic/phase_shift_abrupt.trace",
    ]
    evaluate_all(
        traces=traces,
        cache_sizes=args.cache_sizes,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


def mode_compare(args):
    """Quick comparison table printed to stdout."""
    print("=" * 60)
    print("Policy Comparison (quick run, cache=256)")
    print("=" * 60)
    from evaluate import evaluate_all
    traces = [
        "traces/synthetic/mixed_phase.trace",
    ]
    rows, _ = evaluate_all(
        traces=traces,
        cache_sizes=[256],
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    from collections import defaultdict
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r["miss_rate"])

    print(f"\n{'Policy':<10} {'Miss Rate (%)':>15}")
    print("-" * 28)
    for policy, rates in sorted(by_policy.items(), key=lambda x: x[1][0]):
        avg = sum(rates) / len(rates) * 100
        print(f"{policy:<10} {avg:>15.2f}")


def mode_plots(args):
    print("=" * 60)
    print("STEP 4: Generating figures")
    print("=" * 60)
    from results.plots import generate_all_figures
    generate_all_figures(
        results_dir=args.output_dir,
        traces_dir="traces/synthetic",
        trace_name="mixed_phase",
        cache_sizes=args.cache_sizes,
    )


def main():
    parser = argparse.ArgumentParser(
        description="PARL — Phase-Aware RL Cache Replacement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  generate_traces  Generate synthetic .trace files
  train            Train the DQN agent
  eval             Evaluate all policies head-to-head
  compare          Quick comparison table (cache=256)
  plots            Generate publication figures
  all              Run everything end-to-end

Examples:
  python run_experiment.py --mode all
  python run_experiment.py --mode train --trace traces/synthetic/mixed_phase.trace
  python run_experiment.py --mode compare
        """,
    )
    parser.add_argument("--mode", required=True,
                        choices=["generate_traces", "train", "eval",
                                 "compare", "plots", "monitor", "all"])
    parser.add_argument("--trace", default="traces/synthetic/mixed_phase.trace",
                        help="Path to .trace file")
    parser.add_argument("--train_mode", default="dqn_direct",
                        choices=["dqn_direct", "policy_select"],
                        help="Training mode: dqn_direct (per-page NN) or policy_select (choose strategy)")
    parser.add_argument("--cache_size", type=int, default=256)
    parser.add_argument("--cache_sizes", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--model_path", default="models/dqn_direct.pt")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--monitor_interval", type=float, default=1.0,
                        help="Real-time monitor sampling interval (seconds)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.mode == "generate_traces":
        mode_generate_traces()
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "eval":
        mode_eval(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "plots":
        mode_plots(args)
    elif args.mode == "monitor":
        print("=" * 60)
        print("PARL Real-Time Web Dashboard")
        print(f"  Open: http://localhost:5050")
        print("=" * 60)
        from realtime.server import run_server
        try:
            run_server(
                host="0.0.0.0",
                port=5050,
                model_path=args.model_path,
                cache_size=args.cache_size,
                interval=args.monitor_interval,
            )
        except KeyboardInterrupt:
            pass
    elif args.mode == "all":
        mode_generate_traces()
        mode_train(args)
        mode_eval(args)
        mode_plots(args)
        print("\nAll done! Check results/ for figures and evaluation_results.csv")


if __name__ == "__main__":
    main()
