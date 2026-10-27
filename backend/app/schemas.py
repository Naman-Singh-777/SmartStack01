"""The shapes of the data going in and out of the API.

FastAPI reads these classes to validate requests and to document the API at /docs.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import MAX_QUERY_CHARS

Priority = Literal["interactive", "batch"]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    # "interactive" means a person is waiting. "batch" means it can run cheaply later.
    priority: Priority = "interactive"

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class RoutingInfo(BaseModel):
    label: str  # simple, moderate or complex
    confidence: float  # 0 to 1
    probabilities: dict[str, float]
    tier: str  # small, medium or large
    model_id: str
    escalated: bool  # True when low confidence pushed us up a tier
    reason: str


class CacheInfo(BaseModel):
    hit: bool
    similarity: float | None = None
    matched_query: str | None = None


class SourceInfo(BaseModel):
    title: str
    snippet: str
    score: float
    url: str | None = None  # link to the source when it is a web page


class RetrievalInfo(BaseModel):
    used: bool  # True when at least one chunk was good enough to send to the model
    origin: Literal["documents", "wikipedia", "none"] = "none"
    sources: list[SourceInfo]


class ComputePlan(BaseModel):
    provider: str
    region: str
    gpu: str
    pricing: str  # "on-demand" or "spot"
    price_per_hour: float
    baseline_price_per_hour: float
    saving_percent: float
    latency_ms: int
    note: str


class CostInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    actual_usd: float
    baseline_usd: float
    saved_usd: float
    saved_percent: float
    baseline_model: str


class Timings(BaseModel):
    route_ms: float
    cache_ms: float
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    answer_mode: Literal["live", "offline", "cache"]
    warning: str | None = None
    routing: RoutingInfo
    cache: CacheInfo
    retrieval: RetrievalInfo
    compute: ComputePlan | None  # None when the cache answered and nothing had to run
    cost: CostInfo
    timings: Timings
    timestamp: str
