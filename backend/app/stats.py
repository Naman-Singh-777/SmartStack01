"""Running totals shown on the dashboard page.

The numbers live in memory, so they reset when the server restarts. For a real
product you would save them in a database.
"""

import time
from dataclasses import dataclass, field

from .schemas import QueryResponse


@dataclass
class Stats:
    requests: int = 0
    cache_hits: int = 0
    escalations: int = 0
    actual_usd: float = 0.0
    baseline_usd: float = 0.0
    by_tier: dict[str, int] = field(default_factory=lambda: {"small": 0, "medium": 0, "large": 0})
    started_at: float = field(default_factory=time.time)

    def record(self, response: QueryResponse) -> None:
        self.requests += 1
        self.actual_usd += response.cost.actual_usd
        self.baseline_usd += response.cost.baseline_usd
        if response.cache.hit:
            self.cache_hits += 1
        else:
            self.by_tier[response.routing.tier] += 1
        if response.routing.escalated:
            self.escalations += 1

    def snapshot(self) -> dict:
        saved = self.baseline_usd - self.actual_usd
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "escalations": self.escalations,
            "by_tier": dict(self.by_tier),
            "actual_usd": round(self.actual_usd, 8),
            "baseline_usd": round(self.baseline_usd, 8),
            "saved_usd": round(saved, 8),
            "saved_percent": round(saved / self.baseline_usd * 100, 1) if self.baseline_usd else 0.0,
            "uptime_seconds": int(time.time() - self.started_at),
        }
