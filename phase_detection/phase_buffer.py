"""
PhaseBuffer — ring buffer that stores the recent page access sequence.

Plain English:
  The phase detection engine only looks at the LAST W=1000 accesses to determine
  the current phase. This buffer acts like a sliding window — older accesses
  fall off the back as new ones come in from the front.

  Think of it like a moving average in finance — we don't care about what happened
  10000 accesses ago, only the recent pattern matters for detecting current behavior.
"""

from collections import deque


class PhaseBuffer:
    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._buffer: deque[int] = deque(maxlen=window_size)
        self._hit_buffer: deque[bool] = deque(maxlen=window_size)
        self._total_count = 0   # total accesses ever recorded

    def push(self, page_id: int, is_hit: bool):
        self._buffer.append(page_id)
        self._hit_buffer.append(is_hit)
        self._total_count += 1

    @property
    def window(self) -> list[int]:
        return list(self._buffer)

    @property
    def hits(self) -> list[bool]:
        return list(self._hit_buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._window_size

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def total_count(self) -> int:
        return self._total_count

    def reset(self):
        self._buffer.clear()
        self._hit_buffer.clear()
        self._total_count = 0
