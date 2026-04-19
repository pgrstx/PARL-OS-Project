"""
HybridPolicy — weighted combination of LRU and LFU.

How it works (plain English):
  This is a "blend" policy that the RL agent can choose when the workload
  is in transition between phases. It maintains both an LRU and an LFU
  structure internally. When evicting, it picks based on a weighted score:
    score(page) = alpha * lru_rank + (1 - alpha) * lfu_rank
  where alpha=0.5 means equal weight, alpha=1.0 is pure LRU, alpha=0.0 is pure LFU.

  The agent selects HYBRID when it detects the workload is mid-transition
  and neither LRU nor LFU alone is the right call.
"""

from collections import OrderedDict, defaultdict
from typing import Optional
from .base import Policy


class HybridPolicy(Policy):
    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha   # weight for recency vs. frequency

        # LRU component
        self._lru_order: OrderedDict[int, None] = OrderedDict()

        # LFU component
        self._freq: dict[int, int] = {}

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool, current_pages=None) -> Optional[int]:
        if is_hit:
            self._lru_order.move_to_end(page_id)
            self._freq[page_id] = self._freq.get(page_id, 0) + 1
            return None

        # Miss
        evict = None
        if cache_full:
            evict = self._evict()

        # Insert
        self._lru_order[page_id] = None
        self._lru_order.move_to_end(page_id)
        self._freq[page_id] = 1
        return evict

    def _evict(self) -> int:
        """Score all cached pages, evict the one with the lowest score."""
        pages = list(self._lru_order.keys())
        n = len(pages)
        if n == 0:
            return None

        # LRU rank: index in ordered dict (0=oldest/worst, n-1=newest/best)
        lru_ranks = {page: i / max(n - 1, 1) for i, page in enumerate(pages)}

        # LFU rank: normalized frequency (higher freq = better → higher rank)
        max_freq = max(self._freq.get(p, 1) for p in pages)
        lfu_ranks = {p: self._freq.get(p, 1) / max_freq for p in pages}

        # Combined score (higher = keep, lower = evict)
        scores = {
            p: self._alpha * lru_ranks[p] + (1 - self._alpha) * lfu_ranks[p]
            for p in pages
        }
        evict = min(scores, key=scores.get)

        del self._lru_order[evict]
        del self._freq[evict]
        return evict

    def reset(self):
        self._lru_order.clear()
        self._freq.clear()

    def clone_for_shadow(self) -> "HybridPolicy":
        return HybridPolicy(self._alpha)
