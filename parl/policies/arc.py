"""
ARC — Adaptive Replacement Cache (Megiddo & Modha, 2003, USENIX FAST).

How it works (plain English):
  ARC maintains FOUR lists:
    T1: recently accessed pages (recency — seen once)
    T2: frequently accessed pages (frequency — seen 2+ times)
    B1: "ghost" entries for recently evicted T1 pages (no actual data)
    B2: "ghost" entries for recently evicted T2 pages (no actual data)

  The magic: the balance between T1 and T2 is adaptive (parameter p).
    - If a page in B1 is re-accessed → it should have stayed! Grow T1.
    - If a page in B2 is re-accessed → T2 should be bigger. Shrink T1.

  This means ARC *automatically learns* whether the workload is recency-based
  or frequency-based — and adapts the balance accordingly.
  ARC is one of the best practical algorithms (within 2x of OPT on most traces).
"""

from collections import OrderedDict
from typing import Optional
from .base import Policy


class ARCPolicy(Policy):
    def __init__(self, capacity: int = 256):
        self._capacity = capacity
        self._p = 0          # target size for T1 (adaptive parameter)
        self._t1: OrderedDict[int, None] = OrderedDict()   # recent, once-accessed
        self._t2: OrderedDict[int, None] = OrderedDict()   # frequent, 2+ accessed
        self._b1: OrderedDict[int, None] = OrderedDict()   # ghost of T1
        self._b2: OrderedDict[int, None] = OrderedDict()   # ghost of T2

    def set_capacity(self, capacity: int):
        self._capacity = capacity

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        evict = None

        # Case 1: page is in T1 (recently seen once) → promote to T2
        if page_id in self._t1:
            del self._t1[page_id]
            self._t2[page_id] = None
            self._t2.move_to_end(page_id)
            return None

        # Case 2: page is in T2 (frequently seen) → refresh in T2
        if page_id in self._t2:
            self._t2.move_to_end(page_id)
            return None

        # Page is NOT in cache (miss)
        # Case 3: page in B1 ghost → was recently evicted from T1
        if page_id in self._b1:
            # Adapt: B1 hit means we should have kept T1 entries longer → grow p
            delta = max(1, len(self._b2) // max(len(self._b1), 1))
            self._p = min(self._p + delta, self._capacity)
            evict = self._replace(page_id)
            # Remove from ghost list, add to T2 (second access)
            del self._b1[page_id]
            self._t2[page_id] = None
            return evict

        # Case 4: page in B2 ghost → was recently evicted from T2
        if page_id in self._b2:
            # Adapt: B2 hit means T2 should be bigger → shrink p
            delta = max(1, len(self._b1) // max(len(self._b2), 1))
            self._p = max(self._p - delta, 0)
            evict = self._replace(page_id)
            del self._b2[page_id]
            self._t2[page_id] = None
            return evict

        # Case 5: complete miss — not in any list
        total_live = len(self._t1) + len(self._t2)
        total_ghost = len(self._b1) + len(self._b2)

        if total_live == self._capacity:
            # Cache is full — need to evict
            if len(self._t1) < self._capacity:
                # Also trim B1 if needed
                if len(self._t1) + len(self._b1) >= self._capacity:
                    if self._b1:
                        self._b1.popitem(last=False)
                evict = self._replace(page_id)
            else:
                # T1 alone fills cache — evict from T1, no ghost
                if self._t1:
                    evict, _ = self._t1.popitem(last=False)
        elif total_live < self._capacity and total_live + total_ghost >= self._capacity:
            # Cache not full but directory is full
            if total_live + total_ghost >= 2 * self._capacity:
                if self._b2:
                    self._b2.popitem(last=False)

        # Insert new page into T1 (first access)
        self._t1[page_id] = None
        self._t1.move_to_end(page_id)
        return evict

    def _replace(self, page_id: int) -> Optional[int]:
        """Evict a page from T1 or T2 according to p, add to ghost list."""
        if self._t1 and (len(self._t1) > self._p or
                         (page_id in self._b2 and len(self._t1) == self._p)):
            # Evict from T1
            evict, _ = self._t1.popitem(last=False)
            self._b1[evict] = None
        elif self._t2:
            # Evict from T2
            evict, _ = self._t2.popitem(last=False)
            self._b2[evict] = None
        elif self._t1:
            evict, _ = self._t1.popitem(last=False)
            self._b1[evict] = None
        else:
            return None
        return evict

    def reset(self):
        self._p = 0
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()

    def clone_for_shadow(self) -> "ARCPolicy":
        return ARCPolicy(self._capacity)
