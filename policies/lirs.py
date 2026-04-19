"""
LIRS — Low Inter-reference Recency Set (Jiang & Zhang, 2002, SIGMETRICS).

How it works (plain English):
  LIRS classifies pages as "hot" (LIR) or "cold" (HIR) based on their
  Inter-Reference Recency (IRR) — how long between two consecutive accesses.

  - LIR pages: small IRR (frequently reused) → stay in cache
  - HIR pages: large IRR (rarely reused) → candidates for eviction

  The key insight: instead of counting accesses (like LFU) or just using order
  (like LRU), LIRS uses the *distance between accesses* to predict future use.
  This makes it much better at handling sequential scans (loop-through patterns)
  while keeping hot pages cached.

  Hot LIR set: up to 99% of cache capacity.
  Cold HIR set: up to 1% of cache capacity (resident HIR pages).

  Implementation is a simplified version that maintains the core invariant.
"""

from collections import OrderedDict
from typing import Optional
from .base import Policy


class LIRSPolicy(Policy):
    def __init__(self, capacity: int = 256, lir_ratio: float = 0.99):
        self._capacity = capacity
        self._lir_limit = max(1, int(capacity * lir_ratio))
        self._hir_limit = max(1, capacity - self._lir_limit)

        self._lir_set: OrderedDict[int, None] = OrderedDict()   # hot pages
        self._hir_res: OrderedDict[int, None] = OrderedDict()   # cold resident pages
        self._hir_nonres: OrderedDict[int, None] = OrderedDict() # cold non-resident (ghost)
        self._stack: OrderedDict[int, str] = OrderedDict()       # LRU stack (page → type)

    def set_capacity(self, capacity: int):
        self._capacity = capacity
        self._lir_limit = max(1, int(capacity * 0.99))
        self._hir_limit = max(1, capacity - self._lir_limit)

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        evict = None

        if page_id in self._lir_set:
            # LIR hit: move to top of stack
            self._stack.move_to_end(page_id)
            self._lir_set.move_to_end(page_id)
            self._prune_stack()
            return None

        if page_id in self._hir_res:
            # HIR resident hit
            if page_id in self._stack:
                # Page is in stack → promote to LIR
                del self._hir_res[page_id]
                self._lir_set[page_id] = None
                self._stack[page_id] = "LIR"
                self._stack.move_to_end(page_id)
                # Demote LIR bottom if LIR set overflows
                if len(self._lir_set) > self._lir_limit:
                    evict = self._demote_lir_bottom()
            else:
                # Not in stack → stays HIR, move to top of stack
                self._stack[page_id] = "HIR"
                self._stack.move_to_end(page_id)
            self._prune_stack()
            return evict

        # Miss (not in LIR or resident HIR)
        if page_id in self._hir_nonres:
            del self._hir_nonres[page_id]

        # Evict from HIR resident if needed
        if len(self._lir_set) + len(self._hir_res) >= self._capacity:
            if self._hir_res:
                evict, _ = self._hir_res.popitem(last=False)
            elif self._lir_set:
                # Extreme case: demote from LIR
                evict = self._demote_lir_bottom()

        # Insert as HIR resident if LIR is full, else LIR
        if len(self._lir_set) < self._lir_limit:
            self._lir_set[page_id] = None
            self._stack[page_id] = "LIR"
        else:
            self._hir_res[page_id] = None
            self._stack[page_id] = "HIR"
        self._stack.move_to_end(page_id)
        self._prune_stack()
        return evict

    def _demote_lir_bottom(self) -> Optional[int]:
        """Demote the bottom LIR page to HIR and evict it."""
        # Find oldest LIR entry in stack
        for page_id, ptype in self._stack.items():
            if ptype == "LIR" and page_id in self._lir_set:
                del self._lir_set[page_id]
                self._stack[page_id] = "HIR"
                # Immediately evict (move to non-resident)
                del self._stack[page_id]
                self._hir_nonres[page_id] = None
                return page_id
        return None

    def _prune_stack(self):
        """Remove bottom HIR entries from stack (keep only up to top LIR)."""
        while self._stack:
            bottom_page = next(iter(self._stack))
            if self._stack[bottom_page] == "HIR" and bottom_page not in self._lir_set:
                del self._stack[bottom_page]
            else:
                break

    def reset(self):
        self._lir_set.clear()
        self._hir_res.clear()
        self._hir_nonres.clear()
        self._stack.clear()

    def clone_for_shadow(self) -> "LIRSPolicy":
        return LIRSPolicy(self._capacity)
