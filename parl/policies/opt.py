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
        Pre-scan the trace to build the next_use table.
        Call this ONCE before running the simulation.

        next_use[i] = index of next access to trace[i] after position i.
        If page is never accessed again, next_use[i] = infinity (len(trace)).
        """
        self._trace = trace
        n = len(trace)
        self._next_use = [n] * n          # default: never used again = n (infinity)

        # Build next_use by scanning backwards
        last_seen: dict[int, int] = {}
        for i in range(n - 1, -1, -1):
            page = trace[i]
            if page in last_seen:
                self._next_use[i] = last_seen[page]
            last_seen[page] = i

        self._current_idx = 0

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        idx = self._current_idx
        self._current_idx += 1

        if is_hit:
            return None

        # Miss — need to insert page_id
        if not cache_full:
            self._in_cache.add(page_id)
            return None

        # Evict the page whose next use is furthest in the future
        # We need next_use for each page currently in cache
        # Build a quick lookup: for pages in cache, find their next access from idx
        evict_candidate = None
        max_next_use = -1

        for p in self._in_cache:
            # Find the next use of p from current position idx onward
            # We use the precomputed next_use of the most recent occurrence
            # Efficient: scan forward from idx for each page (approximation)
            # For correctness, we track the current position's next_use mapping
            nu = self._get_next_use(p, idx)
            if nu > max_next_use:
                max_next_use = nu
                evict_candidate = p

        if evict_candidate is not None:
            self._in_cache.discard(evict_candidate)
        self._in_cache.add(page_id)
        return evict_candidate

    def _get_next_use(self, page_id: int, from_idx: int) -> int:
        """Find next use of page_id starting from from_idx."""
        n = len(self._trace)
        for i in range(from_idx, n):
            if self._trace[i] == page_id:
                return i
        return n  # never used again

    def reset(self):
        self._current_idx = 0
        self._in_cache.clear()
        # Note: _trace and _next_use are NOT cleared — prescan data persists

    def clone_for_shadow(self) -> "OPTPolicy":
        new = OPTPolicy()
        new._trace = self._trace
        new._next_use = self._next_use
        return new
