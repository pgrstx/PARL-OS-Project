"""
Abstract base class for all page replacement policies.

Every policy implements two methods:
  on_access(page_id, is_hit) → evict_page or None
  reset()

The Cache calls on_access after determining hit/miss.
If it's a miss AND cache is full, the policy must return the page_id to evict.
If it's a hit OR cache is not yet full, return None.
"""

from abc import ABC, abstractmethod
from typing import Optional


class Policy(ABC):
    """Base class for all page replacement policies."""

    @abstractmethod
    def on_access(self, page_id: int, is_hit: bool, cache_full: bool) -> Optional[int]:
        """
        Called on every page access.

        Args:
            page_id:    The page being accessed.
            is_hit:     True if page is already in cache.
            cache_full: True if cache is at capacity (eviction needed on miss).

        Returns:
            page_id to evict if (not is_hit and cache_full), else None.
        """

    @abstractmethod
    def reset(self):
        """Reset all internal state (called when cache is cleared)."""

    def clone_for_shadow(self) -> "Policy":
        """Return a fresh instance of the same policy (for shadow caches)."""
        return self.__class__()
