"""
LFU — Least Frequently Used page replacement policy.

How it works (plain English):
  Count how many times each page has been used. When you need to evict,
  remove the page that has been used the least. Ties broken by LRU order
  (the oldest among pages with the same frequency gets evicted first).

Why it's useful: good for workloads where some pages are "permanently hot"
(databases, hot code paths). Performs well on stable workloads, but suffers
on cold starts because every new page starts with frequency=1.

Implementation uses frequency buckets (OrderedDicts) for O(1) operations.
"""

from collections import OrderedDict, defaultdict
from typing import Optional
from .base import Policy


class LFUPolicy(Policy):
    def __init__(self):
        self._freq_map: dict[int, int] = {}                          # page_id → freq
        self._freq_to_pages: defaultdict[int, OrderedDict] = \
            defaultdict(OrderedDict)                                  # freq → {page_id: None}
        self._min_freq: int = 0

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        if is_hit:
            self._increment_freq(page_id)
            return None

        # Miss
        evict = None
        if cache_full:
            evict = self._evict_lfu()

        # Insert new page with freq=1
        self._freq_map[page_id] = 1
        self._freq_to_pages[1][page_id] = None
        self._min_freq = 1       # new page always has freq 1 = new minimum
        return evict

    def _increment_freq(self, page_id: int):
        old_freq = self._freq_map[page_id]
        new_freq = old_freq + 1
        self._freq_map[page_id] = new_freq

        # Move from old bucket to new bucket
        del self._freq_to_pages[old_freq][page_id]
        if not self._freq_to_pages[old_freq] and old_freq == self._min_freq:
            self._min_freq = new_freq
        self._freq_to_pages[new_freq][page_id] = None

    def _evict_lfu(self) -> int:
        # Evict the oldest page in the minimum-frequency bucket
        bucket = self._freq_to_pages[self._min_freq]
        evict, _ = bucket.popitem(last=False)  # FIFO within same freq
        del self._freq_map[evict]
        return evict

    def reset(self):
        self._freq_map.clear()
        self._freq_to_pages.clear()
        self._min_freq = 0

    def clone_for_shadow(self) -> "LFUPolicy":
        return LFUPolicy()
