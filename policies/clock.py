"""
CLOCK page replacement policy (also called "Second Chance").

How it works (plain English):
  Imagine pages arranged in a circle with a clock hand pointing at one of them.
  Each page has a "reference bit" (0 or 1). When you access a page, set its bit to 1.
  When you need to evict: advance the clock hand. If the page it lands on has bit=1,
  clear the bit (give it a second chance) and move on. Evict the first page with bit=0.

This is used in real OS kernels because it's O(1) and hardware-friendly.
"""

from typing import Optional
from .base import Policy


class CLOCKPolicy(Policy):
    def __init__(self):
        self._pages: list[int] = []        # circular buffer of page_ids
        self._ref_bits: list[int] = []     # reference bit per slot
        self._page_to_slot: dict[int, int] = {}  # page_id → slot index
        self._hand: int = 0                # clock hand position

    def on_access(self, page_id: int, is_hit: bool, cache_full: bool, current_pages=None) -> Optional[int]:
        if is_hit:
            slot = self._page_to_slot[page_id]
            self._ref_bits[slot] = 1
            return None

        # Miss — need to insert page_id
        evict = None

        if cache_full:
            # Advance hand until we find a page with ref_bit=0
            while True:
                if self._ref_bits[self._hand] == 0:
                    break
                self._ref_bits[self._hand] = 0  # second chance: clear bit
                self._hand = (self._hand + 1) % len(self._pages)

            # Evict the page at current hand position
            evict = self._pages[self._hand]
            del self._page_to_slot[evict]

            # Insert new page into this slot
            self._pages[self._hand] = page_id
            self._ref_bits[self._hand] = 1
            self._page_to_slot[page_id] = self._hand
            self._hand = (self._hand + 1) % len(self._pages)
        else:
            # Cache not full yet — append
            slot = len(self._pages)
            self._pages.append(page_id)
            self._ref_bits.append(1)
            self._page_to_slot[page_id] = slot

        return evict

    def reset(self):
        self._pages.clear()
        self._ref_bits.clear()
        self._page_to_slot.clear()
        self._hand = 0

    def clone_for_shadow(self) -> "CLOCKPolicy":
        return CLOCKPolicy()
