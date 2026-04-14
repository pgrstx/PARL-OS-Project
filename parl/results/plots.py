"""
plots.py — generates all 5 publication-quality figures for the PARL report.

Figure 1: Overall miss rate comparison (bar chart, grouped by cache size)
Figure 2: Miss rate over time with phase annotations (line chart)
Figure 3: Per-phase miss rate heatmap (seaborn)
Figure 4: Training curve — smoothed reward over time
Figure 5: Adaptation latency box plot

All figures saved as results/fig1.png ... fig5.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHASE_COLORS = {
    0: "#AED6F1",   # light blue — initialization
    1: "#A9DFBF",   # light green — steady state
    2: "#FAD7A0",   # light orange — sequential scan
    3: "#F1948A",   # light red — random access
    4: "#D7BDE2",   # light purple — hotspot
}
PHASE_NAMES = {
    0: "Init", 1: "Steady", 2: "Sequential", 3: "Random", 4: "Hotspot"
}

POLICY_COLORS = {
    "LRU":   "#2196F3",
    "CLOCK": "#4CAF50",
    "LFU":   "#FF9800",
    "ARC":   "#9C27B0",
    "LIRS":  "#00BCD4",
    "OPT":   "#F44336",
    "PARL":  "#E91E63",
}


def figure1_miss_rate_comparison(results_csv: str, output_path: str):
    """Bar chart: overall miss rate by policy, grouped by cache size."""
    df = pd.read_csv(results_csv)
    # Take unique (policy, trace, cache_size) combos
    agg = df.groupby(["policy", "cache_size"])["miss_rate"].mean().reset_index()

    cache_sizes = sorted(agg["cache_size"].unique())
    policies = sorted(agg["policy"].unique(), key=lambda p: agg[agg["policy"] == p]["miss_rate"].mean())

    n_policies = len(policies)
    n_sizes = len(cache_sizes)
    x = np.arange(n_sizes)
    width = 0.8 / n_policies

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, policy in enumerate(policies):
        vals = []
        for cs in cache_sizes:
            row = agg[(agg["policy"] == policy) & (agg["cache_size"] == cs)]
            vals.append(row["miss_rate"].values[0] * 100 if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=policy,
                      color=POLICY_COLORS.get(policy, "#999"), alpha=0.85, edgecolor="white")

    ax.set_xlabel("Cache Size (pages)", fontsize=13)
    ax.set_ylabel("Miss Rate (%)", fontsize=13)
    ax.set_title("Figure 1: Overall Miss Rate Comparison — All Policies", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_policies - 1) / 2)
    ax.set_xticklabels([str(cs) for cs in cache_sizes])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def figure2_miss_rate_over_time(rolling_npz: str, labels_file: str,
                                trace_name: str, cache_size: int,
                                output_path: str):
    """Line chart: rolling miss rate over time with phase shading."""
    data = np.load(rolling_npz, allow_pickle=True)

    policies_to_plot = ["LRU", "ARC", "PARL"]
    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade background by phase if labels available
    if labels_file and os.path.exists(labels_file):
        labels = np.array([int(l) for l in open(labels_file).readlines()])
        n = len(labels)
        # Find phase boundaries
        boundaries = [0]
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                boundaries.append(i)
        boundaries.append(n)
        for j in range(len(boundaries) - 1):
            start, end = boundaries[j], boundaries[j + 1]
            phase = labels[start]
            ax.axvspan(start, end, alpha=0.15, color=PHASE_COLORS.get(phase, "#eee"),
                       label=f"Phase {phase}: {PHASE_NAMES.get(phase, '')}" if j < 5 else "")

    for policy in policies_to_plot:
        key = f"{policy}__{trace_name}__{cache_size}"
        if key in data:
            arr = data[key] * 100
            # Smooth with exponential moving average
            smoothed = pd.Series(arr).ewm(span=50, adjust=False).mean().values
            ax.plot(smoothed, label=policy, color=POLICY_COLORS.get(policy, "#999"),
                    linewidth=1.8, alpha=0.9)

    ax.set_xlabel("Access Number", fontsize=13)
    ax.set_ylabel("Rolling Miss Rate (%)", fontsize=13)
    ax.set_title(f"Figure 2: Miss Rate Over Time — {trace_name} (cache={cache_size})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def figure3_phase_heatmap(results_csv: str, output_path: str):
    """Seaborn heatmap: per-phase miss rate for each policy."""
    df = pd.read_csv(results_csv)
    df = df[df["phase_miss_rate"] != ""]
    df["phase_miss_rate"] = df["phase_miss_rate"].astype(float)

    # Average across traces and cache sizes
    pivot = df.groupby(["policy", "phase"])["phase_miss_rate"].mean().unstack()
    pivot = pivot * 100  # to percentage

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
                ax=ax, linewidths=0.5, cbar_kws={"label": "Miss Rate (%)"},
                xticklabels=[f"Phase {p}: {PHASE_NAMES[p]}" for p in sorted(pivot.columns)])
    ax.set_title("Figure 3: Per-Phase Miss Rate Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Workload Phase", fontsize=12)
    ax.set_ylabel("Replacement Policy", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def figure4_training_curve(training_log_csv: str, output_path: str):
    """Line chart: smoothed reward over training steps."""
    df = pd.read_csv(training_log_csv)
    steps = df["step"].values
    rewards = df["reward"].values.astype(float)

    # EMA smoothing
    smoothed = pd.Series(rewards).ewm(alpha=0.05, adjust=False).mean().values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, rewards, color="#BBDEFB", linewidth=0.5, alpha=0.4, label="Raw reward")
    ax.plot(steps, smoothed, color=POLICY_COLORS["PARL"], linewidth=2.0,
            label="EMA smoothed (α=0.05)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="LRU baseline")

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Reward (PARL improvement over LRU)", fontsize=13)
    ax.set_title("Figure 4: PARL Training Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def figure5_adaptation_latency(rolling_npz: str, labels_file: str,
                                trace_name: str, cache_sizes: list,
                                output_path: str):
    """Box plot: how many steps until miss rate recovers after phase transition."""
    data = np.load(rolling_npz, allow_pickle=True)

    if not labels_file or not os.path.exists(labels_file):
        print("  Skipping Fig 5: no labels file")
        return

    labels = np.array([int(l) for l in open(labels_file).readlines()])
    transitions = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]

    policies_to_compare = ["LRU", "ARC", "LIRS", "PARL"]
    latency_data = {p: [] for p in policies_to_compare}

    for policy in policies_to_compare:
        cache_size = cache_sizes[2] if len(cache_sizes) > 2 else cache_sizes[-1]
        key = f"{policy}__{trace_name}__{cache_size}"
        if key not in data:
            continue
        rolling = np.array(data[key]) * 100

        for t in transitions:
            if t + 500 >= len(rolling):
                continue
            pre_rate = float(np.mean(rolling[max(0, t - 100):t]))
            spike = rolling[t]
            target = pre_rate * 1.05  # recover to within 5% of pre-transition

            recovery_steps = 500  # default if never recovers
            for steps_after in range(1, 500):
                if rolling[t + steps_after] <= target:
                    recovery_steps = steps_after
                    break
            latency_data[policy].append(recovery_steps)

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_policies = [p for p in policies_to_compare if latency_data[p]]
    bp = ax.boxplot([latency_data[p] for p in valid_policies],
                    labels=valid_policies, patch_artist=True, notch=False)

    for patch, policy in zip(bp["boxes"], valid_policies):
        patch.set_facecolor(POLICY_COLORS.get(policy, "#999"))
        patch.set_alpha(0.7)

    ax.set_xlabel("Replacement Policy", fontsize=13)
    ax.set_ylabel("Adaptation Latency (steps to recover)", fontsize=13)
    ax.set_title("Figure 5: Adaptation Latency After Phase Transition",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(
    results_dir: str = "results",
    traces_dir: str = "traces/synthetic",
    trace_name: str = "mixed_phase",
    cache_sizes: list = (64, 128, 256, 512),
):
    results_csv = os.path.join(results_dir, "evaluation_results.csv")
    rolling_npz = os.path.join(results_dir, "rolling_miss_rates.npz")
    training_log = os.path.join(results_dir, "training_log.csv")
    labels_file = os.path.join(traces_dir, f"{trace_name}.labels")

    print("Generating Figure 1...")
    if os.path.exists(results_csv):
        figure1_miss_rate_comparison(results_csv, os.path.join(results_dir, "fig1.png"))

    print("Generating Figure 2...")
    if os.path.exists(rolling_npz):
        figure2_miss_rate_over_time(rolling_npz, labels_file, trace_name,
                                     cache_sizes[2], os.path.join(results_dir, "fig2.png"))

    print("Generating Figure 3...")
    if os.path.exists(results_csv):
        figure3_phase_heatmap(results_csv, os.path.join(results_dir, "fig3.png"))

    print("Generating Figure 4...")
    if os.path.exists(training_log):
        figure4_training_curve(training_log, os.path.join(results_dir, "fig4.png"))

    print("Generating Figure 5...")
    if os.path.exists(rolling_npz):
        figure5_adaptation_latency(rolling_npz, labels_file, trace_name,
                                    list(cache_sizes), os.path.join(results_dir, "fig5.png"))

    print("\nAll figures generated in:", results_dir)


if __name__ == "__main__":
    generate_all_figures()
