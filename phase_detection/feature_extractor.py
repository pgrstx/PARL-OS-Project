"""
FeatureExtractor — converts a window of page accesses into a numeric feature vector.

Plain English:
  Just like a doctor takes measurements (temperature, blood pressure, etc.) to
  diagnose a patient, we extract 6 "measurements" from the recent access pattern
  to characterize what "phase" the workload is currently in.

  The 6 features are:
    f1: miss_rate           — how often we miss (high = bad locality)
    f2: ird_mean            — average distance between re-accesses to same page
    f3: ird_std             — how variable that distance is
    f4: entropy             — how random the access pattern is (high = random)
    f5: reuse_dist_p50      — median distance between two accesses to same page
    f6: unique_pages_ratio  — fraction of pages that appear exactly once (streaming?)

  These 6 numbers tell the phase classifier enough to distinguish:
    - Cold start (lots of unique pages, high miss rate, high entropy)
    - Steady state (low ird, low entropy, low miss rate)
    - Sequential scan (very high unique ratio, sequential IRD ~1)
    - Random access (very high entropy, uniform IRD)
    - Hotspot (very low entropy, few unique pages, low reuse distance)
"""

import math
from collections import Counter
import numpy as np
from .phase_buffer import PhaseBuffer


class FeatureExtractor:
    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._buffer = PhaseBuffer(window_size)

    def push(self, page_id: int, is_hit: bool):
        self._buffer.push(page_id, is_hit)

    @property
    def buffer(self) -> PhaseBuffer:
        return self._buffer

    def extract(self) -> np.ndarray:
        """
        Extract the 6-dimensional feature vector from the current window.
        Returns a numpy array of shape (6,).
        """
        window = self._buffer.window
        hits = self._buffer.hits
        n = len(window)

        if n == 0:
            return np.zeros(6, dtype=np.float32)

        # f1: miss_rate
        miss_rate = 1.0 - (sum(hits) / n) if n > 0 else 0.5

        # Compute Inter-Reference Distances (IRD) and reuse distances
        last_seen: dict[int, int] = {}
        ird_values: list[int] = []
        reuse_distances: list[int] = []

        for i, page in enumerate(window):
            if page in last_seen:
                dist = i - last_seen[page]
                ird_values.append(dist)
                reuse_distances.append(dist)
            last_seen[page] = i

        # f2: ird_mean
        ird_mean = float(np.mean(ird_values)) if ird_values else float(n)

        # f3: ird_std
        ird_std = float(np.std(ird_values)) if ird_values else 0.0

        # f4: entropy of page access distribution
        counts = Counter(window)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # f5: reuse_distance_p50 (median reuse distance)
        reuse_dist_p50 = float(np.median(reuse_distances)) if reuse_distances else float(n)

        # f6: unique_pages_ratio
        unique_pages_ratio = len(counts) / n

        features = np.array([
            miss_rate,
            ird_mean / n,            # normalize by window size
            ird_std / n,             # normalize by window size
            entropy / math.log2(max(len(counts), 2)),  # normalize by max entropy
            reuse_dist_p50 / n,      # normalize by window size
            unique_pages_ratio,
        ], dtype=np.float32)

        # Clip to [0, 1] for stability
        features = np.clip(features, 0.0, 1.0)
        return features

    def reset(self):
        self._buffer.reset()
