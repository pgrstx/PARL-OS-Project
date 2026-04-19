"""
OPT (Belady's Optimal) page replacement policy — Belady (1966).

How it works (plain English):
  This is the "cheating" algorithm — it looks into the FUTURE to decide which
  page to evict. It always evicts the page that will be needed furthest in the
  future (or never again). This is provably optimal — no algorithm can do better.

  We use OPT as a theoretical upper bound for comparison: it tells us "this is
  the best possible miss rate". Our goal is to get PARL as close to OPT as possible.

  Since OPT needs the full trace in advance, it's only usable for offline
  benchmarking (not real-time use in an OS). The pre-scan builds a map:
    next_use[i] = the next index where trace[i]'s page is accessed again
                  (infinity if never accessed again)
"""

from typing import Optional
from .base import Policy


class OPTPolicy(Policy):
    def __init__(self):
        self._trace: list[int] = []
        self._next_use: list[int] = []   # populated by prescan()
        self._current_idx: int = 0
        self._in_cache: set[int] = set()

    def prescan(self, trace: list[int]):
        """
        Pre-scan the trace to build per-page next_occurrence lists.

        For each page, build a sorted list of all indices where it appears.
        Then during eviction, binary search to find the next occurrence after
        the current position — O(log n) instead of O(n) per query.
        """
        import bisect
        self._trace = trace
        n = len(trace)

        # Build: page_id → sorted list of occurrence indices
        occurrences: dict[int, list[int]] = {}
        for i, page in enumerate(trace):
            if page not in occurrences:
                occurrences[page] = []
            occurrences[page].append(i)
        self._occurrences = occurrences
        self._n = n
        self._bisect = bisect
        self._current_idx = 0

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        idx = self._current_idx
        self._current_idx += 1

        if is_hit:
            return None

        if not cache_full:
            self._in_cache.add(page_id)
            return None

        # Evict the page whose next use is furthest (or never) in the future
        evict_candidate = None
        max_next_use = -1

        for p in self._in_cache:
            nu = self._next_occurrence(p, idx + 1)
            if nu > max_next_use:
                max_next_use = nu
                evict_candidate = p

        if evict_candidate is not None:
            self._in_cache.discard(evict_candidate)
        self._in_cache.add(page_id)
        return evict_candidate

    def _next_occurrence(self, page_id: int, from_idx: int) -> int:
        """Binary search for next occurrence of page_id at or after from_idx."""
        occ = self._occurrences.get(page_id)
        if occ is None:
            return self._n   # never accessed again
        pos = self._bisect.bisect_left(occ, from_idx)
        if pos < len(occ):
            return occ[pos]
        return self._n  # never accessed again

    def reset(self):
        self._current_idx = 0
        self._in_cache.clear()
        # Note: _trace and _next_use are NOT cleared — prescan data persists

    def clone_for_shadow(self) -> "OPTPolicy":
        new = OPTPolicy()
        new._trace = self._trace
        new._next_use = self._next_use
        return new
