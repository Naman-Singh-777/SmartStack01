"""The web API. This file only handles HTTP; the real work happens in engine.py.

Run it locally with:   uvicorn app.main:app --reload
Interactive docs at:   http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .cache import SemanticCache
from .classifier import ComplexityClassifier
from .engine import SmartStack
from .llm import LLMClient
from .ratelimit import RateLimiter
from .retriever import Retriever
from .schemas import QueryRequest, QueryResponse
from .stats import Stats
from .wikipedia import WikipediaSearch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load documents and train the classifier."""
    classifier = ComplexityClassifier.from_file(config.TRAINING_FILE)
    app.state.engine = SmartStack(
        classifier=classifier,
        cache=SemanticCache(),
        retriever=Retriever(),
        llm=LLMClient(),
        stats=Stats(),
        web=WikipediaSearch() if config.WEB_LOOKUP else None,
    )
    app.state.accuracy = classifier.cross_validated_accuracy()
    app.state.limiter = RateLimiter()
    yield


app = FastAPI(
    title="SmartStack API",
    version="1.0.0",
    description="Routes each question to the cheapest model that can handle it.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-LLM-Key"],
)


def visitor_id(request: Request) -> str:
    """Behind a proxy (like Render) the real address is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    if not request.app.state.limiter.allow(visitor_id(request)):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")


@app.get("/")
def root():
    return {"name": "SmartStack API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health(request: Request):
    engine: SmartStack = request.app.state.engine
    return {
        "status": "ok",
        "llm_mode": engine.llm.mode,
        "provider": engine.llm.server_provider,
        "documents": len(engine.retriever.chunks),
        "wikipedia": engine.web is not None,
    }


@app.get("/api/models")
def models(request: Request):
    """The model ladder, example prices and how good the classifier is."""
    engine: SmartStack = request.app.state.engine
    ladder = config.PROVIDERS[engine.llm.server_provider or config.DEFAULT_PROVIDER]
    return {
        "tiers": [
            {
                "key": key,
                "label": ladder[key].label,
                "model_id": ladder[key].model_id,
                "input_per_million": ladder[key].input_per_million,
                "output_per_million": ladder[key].output_per_million,
            }
            for key in config.TIER_ORDER
        ],
        "baseline_tier": config.BASELINE_TIER,
        "min_confidence": config.MIN_CONFIDENCE,
        "classifier_accuracy": round(request.app.state.accuracy, 3),
        "training_examples": len(engine.classifier.texts),
        "llm_mode": engine.llm.mode,
    }


@app.post("/api/query", response_model=QueryResponse, dependencies=[Depends(check_rate_limit)])
async def query(
    body: QueryRequest,
    request: Request,
    x_llm_key: str | None = Header(default=None, max_length=300),
):
    """x_llm_key is the X-LLM-Key header: an optional key the visitor brings.
    It is used for this request only. We never store it or write it to a log."""
    return await request.app.state.engine.handle(body.query, body.priority, x_llm_key)


@app.get("/api/stats")
def stats(request: Request):
    return request.app.state.engine.stats.snapshot()
