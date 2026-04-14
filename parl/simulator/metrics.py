"""
Metrics collection for cache simulation.

HitRateTracker: maintains a rolling window of hit/miss events and can report
  statistics at any point during a simulation run.
"""

from collections import deque
import numpy as np


class HitRateTracker:
    """
    Tracks hit rate over a sliding window.
    Used to compute per-phase statistics and RL reward signals.
    """

    def __init__(self, window_size: int = 500):
        self._window = deque(maxlen=window_size)
        self._window_size = window_size
        self._total_hits = 0
        self._total_accesses = 0
        self._history: list[float] = []   # snapshot of rolling hit rate at each step

    def record(self, is_hit: bool):
        self._window.append(1 if is_hit else 0)
        self._total_hits += int(is_hit)
        self._total_accesses += 1
        self._history.append(self.rolling_hit_rate)

    @property
    def rolling_hit_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def rolling_miss_rate(self) -> float:
        return 1.0 - self.rolling_hit_rate

    @property
    def overall_hit_rate(self) -> float:
        if self._total_accesses == 0:
            return 0.0
        return self._total_hits / self._total_accesses

    @property
    def overall_miss_rate(self) -> float:
        return 1.0 - self.overall_hit_rate

    def history_array(self) -> np.ndarray:
        return np.array(self._history, dtype=float)

    def reset(self):
        self._window.clear()
        self._total_hits = 0
        self._total_accesses = 0
        self._history.clear()
