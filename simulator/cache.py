"""
Cache — the core memory simulator.

Plain English:
  This is the virtual "RAM" we're simulating. It has a fixed number of slots
  (capacity). Every time a program requests a page:
    - If it's already in the cache → HIT (fast, no disk access)
    - If it's not → MISS (slow, need to load from disk, maybe evict something)

  The Cache class doesn't decide WHICH page to evict — that's the policy's job.
  Cache just keeps track of what's in memory and calls the policy when needed.

  We also maintain a rolling hit rate over the last N accesses for the RL reward.
"""

from collections import deque
from typing import Optional
from policies.base import Policy


class Cache:
    def __init__(self, capacity: int, policy: Policy):
        self._capacity = capacity
        self._policy = policy
        self._pages: set[int] = set()     # set of page_ids currently in cache

        # Statistics
        self._hits = 0
        self._misses = 0
        self._total = 0

        # Rolling window for recent hit rate (used by RL reward)
        self._window_size = 100
        self._recent: deque[int] = deque(maxlen=self._window_size)  # 1=hit, 0=miss

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def policy(self) -> Policy:
        return self._policy

    def set_policy(self, new_policy: Policy):
        """
        Switch to a new replacement policy mid-run.
        The new policy's internal state starts fresh, but the cache contents remain.
        We populate the new policy by replaying current pages as cold inserts
        (is_hit=False, cache_full=False) so each policy builds its internal state.
        """
        self._policy = new_policy
        # Re-inject current pages as cold insertions (new policy has no state yet)
        for page in list(self._pages):
            new_policy.on_access(page, is_hit=False, cache_full=False)

    def access(self, page_id: int) -> bool:
        """
        Access a page. Returns True if hit, False if miss.
        """
        self._total += 1
        is_hit = page_id in self._pages
        cache_full = len(self._pages) >= self._capacity

        evict = self._policy.on_access(page_id, is_hit, cache_full,
                                        current_pages=frozenset(self._pages))

        if not is_hit:
            self._misses += 1
            self._recent.append(0)
            if evict is not None:
                self._pages.discard(evict)
            self._pages.add(page_id)
        else:
            self._hits += 1
            self._recent.append(1)

        return is_hit

    @property
    def hit_rate(self) -> float:
        """Overall hit rate since last reset."""
        if self._total == 0:
            return 0.0
        return self._hits / self._total

    @property
    def miss_rate(self) -> float:
        """Overall miss rate since last reset."""
        return 1.0 - self.hit_rate

    @property
    def hit_rate_recent(self) -> float:
        """Hit rate over the last window_size accesses."""
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)

    @property
    def miss_rate_recent(self) -> float:
        return 1.0 - self.hit_rate_recent

    @property
    def cache_fullness(self) -> float:
        """Fraction of cache currently filled (0.0 to 1.0)."""
        return len(self._pages) / self._capacity

    def get_stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": self._total,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "hit_rate_recent": self.hit_rate_recent,
            "cache_size": len(self._pages),
            "capacity": self._capacity,
        }

    def reset(self):
        """Clear all cache contents and statistics. Policy state also reset."""
        self._pages.clear()
        self._policy.reset()
        self._hits = 0
        self._misses = 0
        self._total = 0
        self._recent.clear()
