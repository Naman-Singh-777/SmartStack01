"""Suggests where to run a model so the compute bill stays low.

This is a planner. It does not start any machines. It works from the price table
below, which holds example numbers, so edit the table to match your own quotes.

The rules are simple:
* The tier decides the GPU class (a small model does not need an expensive GPU).
* "interactive" work has a person waiting, so it must stay under a latency
  budget and uses reliable on-demand machines.
* "batch" work can wait and can be retried, so it ignores latency and uses the
  cheaper spot machines in the cheapest region.
"""

from dataclasses import dataclass

from .schemas import ComputePlan

INTERACTIVE_LATENCY_BUDGET_MS = 80
BASELINE_GPU = "A100-class"  # what a team that never optimises would rent

GPU_FOR_TIER = {"small": "T4-class", "medium": "A10-class", "large": "A100-class"}


@dataclass(frozen=True)
class Offer:
    provider: str
    region: str
    gpu: str
    on_demand: float  # example USD per hour
    spot: float  # example USD per hour, can be interrupted
    latency_ms: int  # example network delay to the API server


OFFERS = [
    Offer("AWS", "us-east-1", "T4-class", 0.53, 0.21, 35),
    Offer("GCP", "us-central1", "T4-class", 0.35, 0.14, 45),
    Offer("Azure", "eastus", "T4-class", 0.53, 0.19, 40),
    Offer("GCP", "asia-south1", "T4-class", 0.33, 0.12, 120),
    Offer("AWS", "us-east-1", "A10-class", 1.01, 0.40, 35),
    Offer("GCP", "us-central1", "A10-class", 0.95, 0.38, 45),
    Offer("Azure", "eastus", "A10-class", 1.10, 0.42, 40),
    Offer("GCP", "asia-south1", "A10-class", 0.90, 0.35, 120),
    Offer("AWS", "us-east-1", "A100-class", 4.10, 1.60, 35),
    Offer("GCP", "us-central1", "A100-class", 3.67, 1.10, 45),
    Offer("Azure", "eastus", "A100-class", 3.40, 1.30, 40),
    Offer("GCP", "asia-south1", "A100-class", 3.30, 1.05, 120),
]

# The "do nothing clever" choice: biggest GPU, first provider, on demand.
BASELINE = next(o for o in OFFERS if o.gpu == BASELINE_GPU)


def plan_compute(tier: str, priority: str) -> ComputePlan:
    gpu = GPU_FOR_TIER[tier]
    candidates = [offer for offer in OFFERS if offer.gpu == gpu]
    use_spot = priority == "batch"

    if not use_spot:
        candidates = [o for o in candidates if o.latency_ms <= INTERACTIVE_LATENCY_BUDGET_MS]

    def price(offer: Offer) -> float:
        return offer.spot if use_spot else offer.on_demand

    chosen = min(candidates, key=price)
    saving = (1 - price(chosen) / BASELINE.on_demand) * 100
    if use_spot:
        note = "Batch work can wait and be retried, so it runs on spot machines in the cheapest region."
    else:
        note = f"A person is waiting, so only regions under {INTERACTIVE_LATENCY_BUDGET_MS} ms were considered."
    return ComputePlan(
        provider=chosen.provider,
        region=chosen.region,
        gpu=chosen.gpu,
        pricing="spot" if use_spot else "on-demand",
        price_per_hour=price(chosen),
        baseline_price_per_hour=BASELINE.on_demand,
        saving_percent=round(saving, 1),
        latency_ms=chosen.latency_ms,
        note=note,
    )
