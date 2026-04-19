"""
TracePlayer — loads and replays .trace files.

Plain English:
  A trace file is just a log of "which page was accessed at each step".
  TracePlayer reads this file and feeds it to a Cache one access at a time.
  This lets us simulate exactly what a real program's memory access pattern
  would look like, and measure how different cache policies perform.
"""

import os
from typing import Iterator, Optional


class TracePlayer:
    def __init__(self, trace_path: str):
        self._path = trace_path
        self._trace: list[int] = []
        self._labels: Optional[list[int]] = None   # ground truth phase labels if available
        self._load()

    def _load(self):
        with open(self._path, "r") as f:
            self._trace = [int(line.strip()) for line in f if line.strip()]

        # Try to load companion .labels file
        labels_path = self._path.replace(".trace", ".labels")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self._labels = [int(line.strip()) for line in f if line.strip()]

    @property
    def trace(self) -> list[int]:
        return self._trace

    @property
    def labels(self) -> Optional[list[int]]:
        return self._labels

    def __len__(self) -> int:
        return len(self._trace)

    def __iter__(self) -> Iterator[int]:
        return iter(self._trace)

    def iterate_with_labels(self) -> Iterator[tuple[int, Optional[int]]]:
        """Yield (page_id, phase_label) pairs. Phase label is None if no labels file."""
        labels = self._labels or [None] * len(self._trace)
        for page, label in zip(self._trace, labels):
            yield page, label

    @staticmethod
    def from_list(pages: list[int]) -> "TracePlayer":
        """Create a TracePlayer from an in-memory list (for testing)."""
        tp = TracePlayer.__new__(TracePlayer)
        tp._path = "<memory>"
        tp._trace = list(pages)
        tp._labels = None
        return tp
