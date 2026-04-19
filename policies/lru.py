"""
LRU — Least Recently Used page replacement policy.

How it works (plain English):
  Imagine books on a desk arranged in a line. When you use a book, move it to
  the far right (most recent). When the desk is full and you need a new book,
  remove the leftmost one (least recently used). O(1) with OrderedDict.

Implementation:
  - Uses Python's OrderedDict which remembers insertion order.
  - on HIT:  move_to_end(page_id)  — makes it "most recent"
  - on MISS: popitem(last=False)   — removes the "oldest" (leftmost) entry
"""

from collections import OrderedDict
from typing import Optional
from .base import Policy


class LRUPolicy(Policy):
    def __init__(self):
        self._cache: OrderedDict[int, None] = OrderedDict()

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        if is_hit:
            self._cache.move_to_end(page_id)
            return None
        # Miss
        evict = None
        if cache_full:
            evict, _ = self._cache.popitem(last=False)
        self._cache[page_id] = None
        self._cache.move_to_end(page_id)
        return evict

    def reset(self):
        self._cache.clear()

    def clone_for_shadow(self) -> "LRUPolicy":
        return LRUPolicy()
