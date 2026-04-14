"""
Synthetic trace generator for PARL project.

Generates multi-phase page reference traces that simulate real workload behavior:
  Phase 0 - Initialization:  Zipf(0.8)  — cold start, compulsory misses
  Phase 1 - Steady state:    Zipf(1.2)  — high temporal locality
  Phase 2 - Sequential scan: sequential — no locality, LRU thrashes
  Phase 3 - Random access:   uniform    — frequency matters more
  Phase 4 - Hotspot:         pareto     — 80/20 hot-page skew

Each phase's accesses are written one page_id per line.
"""

import random
import numpy as np
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def zipf_sample(alpha: float, n_pages: int, n_samples: int) -> list[int]:
    """Sample from a Zipf distribution over n_pages."""
    ranks = np.arange(1, n_pages + 1, dtype=float)
    probs = 1.0 / np.power(ranks, alpha)
    probs /= probs.sum()
    return list(np.random.choice(n_pages, size=n_samples, p=probs))


def sequential_scan(n_pages: int, n_accesses: int) -> list[int]:
    """Sequentially scan pages 0..n_pages-1, looping as needed."""
    trace = []
    for i in range(n_accesses):
        trace.append(i % n_pages)
    return trace


def uniform_random(n_pages: int, n_accesses: int) -> list[int]:
    """Uniform random access over n_pages."""
    return list(np.random.randint(0, n_pages, size=n_accesses))


def hotspot(hot_pages: int, total_pages: int, hot_fraction: float,
            n_accesses: int) -> list[int]:
    """80% accesses to hot_pages, 20% to remaining pages."""
    trace = []
    hot_set = list(range(hot_pages))
    cold_set = list(range(hot_pages, total_pages))
    for _ in range(n_accesses):
        if random.random() < hot_fraction:
            trace.append(random.choice(hot_set))
        else:
            trace.append(random.choice(cold_set) if cold_set else random.choice(hot_set))
    return trace


def generate_mixed_phase_trace() -> tuple[list[int], list[int]]:
    """
    Returns (trace, phase_labels) where phase_labels[i] is the phase of access i.
    """
    phases = [
        (zipf_sample(0.8, 500, 2000), 0),      # Phase 0: Init
        (zipf_sample(1.2, 200, 5000), 1),       # Phase 1: Steady
        (sequential_scan(300, 3000), 2),         # Phase 2: Sequential
        (uniform_random(1000, 3000), 3),         # Phase 3: Random
        (hotspot(20, 500, 0.80, 4000), 4),       # Phase 4: Hotspot
    ]
    trace, labels = [], []
    for accesses, phase_id in phases:
        trace.extend(accesses)
        labels.extend([phase_id] * len(accesses))
    return trace, labels


def generate_abrupt_phase_shift_trace() -> tuple[list[int], list[int]]:
    """
    Trace with abrupt transitions (no gradual shift) — tests adaptation speed.
    Alternates between high-locality and sequential scan.
    """
    phases = [
        (zipf_sample(1.5, 100, 3000), 1),
        (sequential_scan(500, 3000), 2),
        (zipf_sample(1.5, 100, 3000), 1),
        (sequential_scan(500, 3000), 2),
        (hotspot(10, 300, 0.90, 3000), 4),
    ]
    trace, labels = [], []
    for accesses, phase_id in phases:
        trace.extend(accesses)
        labels.extend([phase_id] * len(accesses))
    return trace, labels


def save_trace(trace: list[int], labels: list[int], trace_path: str, labels_path: str):
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    with open(trace_path, "w") as f:
        for page_id in trace:
            f.write(f"{page_id}\n")
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved {len(trace)} accesses → {trace_path}")
    print(f"Saved phase labels       → {labels_path}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    trace, labels = generate_mixed_phase_trace()
    save_trace(trace, labels,
               os.path.join(base, "mixed_phase.trace"),
               os.path.join(base, "mixed_phase.labels"))

    trace2, labels2 = generate_abrupt_phase_shift_trace()
    save_trace(trace2, labels2,
               os.path.join(base, "phase_shift_abrupt.trace"),
               os.path.join(base, "phase_shift_abrupt.labels"))
