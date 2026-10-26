"""A very small rate limiter so a public demo cannot be hammered.

It remembers when each visitor made their recent requests and refuses new ones
once they pass the limit. It lives in memory, which is fine for one server.
"""

import time
from collections import defaultdict, deque

from . import config


class RateLimiter:
    def __init__(
        self,
        limit: int = config.RATE_LIMIT_REQUESTS,
        window_seconds: float = config.RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, visitor: str) -> bool:
        now = time.monotonic()
        calls = self._calls[visitor]
        while calls and now - calls[0] > self.window_seconds:
            calls.popleft()  # forget calls that are outside the window
        if len(calls) >= self.limit:
            return False
        calls.append(now)
        return True
