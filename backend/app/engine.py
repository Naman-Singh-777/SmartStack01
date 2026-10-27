"""The heart of SmartStack: the steps every question goes through, in order.

    question
       |
       v
    1. ROUTE      classifier decides small, medium or large model
    2. CACHE      a similar question answered recently? return it for free
    3. RETRIEVE   find the few document passages that help
    4. GENERATE   the chosen model writes the answer
    5. PLAN       suggest where to run it
    6. COST       compare with sending everything to the large model

Each step lives in its own file. This file only connects them.
"""

import datetime
import time

from . import config
from .cache import SemanticCache
from .classifier import ComplexityClassifier
from .llm import LLMClient
from .retriever import Hit, Retriever
from .scheduler import plan_compute
from .schemas import (
    CacheInfo, CostInfo, QueryResponse, RetrievalInfo, RoutingInfo, SourceInfo, Timings,
)
from .stats import Stats
from .text_utils import stem, tokenize
from .wikipedia import WikipediaSearch, important_stems


def elapsed_ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def cost_usd(ladder: dict, tier_key: str, input_tokens: int, output_tokens: int) -> float:
    tier = ladder[tier_key]
    return (
        input_tokens * tier.input_per_million + output_tokens * tier.output_per_million
    ) / 1_000_000


class SmartStack:
    def __init__(
        self,
        classifier: ComplexityClassifier,
        cache: SemanticCache,
        retriever: Retriever,
        llm: LLMClient,
        stats: Stats,
        web: WikipediaSearch | None = None,
    ):
        self.classifier = classifier
        self.cache = cache
        self.retriever = retriever
        self.llm = llm
        self.stats = stats
        self.web = web  # None switches the Wikipedia lookup off

    async def handle(
        self, query: str, priority: str, api_key: str | None = None
    ) -> QueryResponse:
        started = time.perf_counter()
        # Which provider answers decides which three models (and prices) the router uses.
        provider, key, problem = self.llm.choose(api_key)
        ladder = config.PROVIDERS[provider or config.DEFAULT_PROVIDER]

        step = time.perf_counter()
        routing = self.route(query, ladder)
        route_ms = elapsed_ms(step)

        step = time.perf_counter()
        hit = self.cache.lookup(query)
        cache_ms = elapsed_ms(step)

        if hit:
            response = self._answer_from_cache(query, routing, ladder, hit, route_ms, cache_ms, started)
        else:
            response = await self._answer_fresh(
                query, priority, routing, ladder, route_ms, cache_ms, started, provider, key, problem
            )

        self.stats.record(response)
        return response

    # -- step 1 -------------------------------------------------------------

    def route(self, query: str, ladder: dict) -> RoutingInfo:
        prediction = self.classifier.predict(query)
        tier_key = config.LABEL_TO_TIER[prediction.label]
        escalated = False
        reason = f"The classifier is {prediction.confidence:.0%} sure this is {prediction.label}."

        # When unsure, pay a little more instead of risking a weak answer.
        last_index = len(config.TIER_ORDER) - 1
        index = config.TIER_ORDER.index(tier_key)
        if prediction.confidence < config.MIN_CONFIDENCE and index < last_index:
            tier_key = config.TIER_ORDER[index + 1]
            escalated = True
            reason += (
                f" That is below {config.MIN_CONFIDENCE:.0%}, so the question moves up one tier to be safe."
            )

        tier = ladder[tier_key]
        return RoutingInfo(
            label=prediction.label,
            confidence=round(prediction.confidence, 4),
            probabilities={k: round(v, 4) for k, v in prediction.probabilities.items()},
            tier=tier_key,
            model_id=tier.model_id,
            escalated=escalated,
            reason=reason,
        )

    # -- steps 2 to 6 -------------------------------------------------------

    def _answer_from_cache(self, query, routing, ladder, hit, route_ms, cache_ms, started) -> QueryResponse:
        saved = hit.entry.payload
        baseline = cost_usd(ladder, config.BASELINE_TIER, saved["input_tokens"], saved["output_tokens"])
        return QueryResponse(
            query=query,
            answer=saved["answer"],
            answer_mode="cache",
            routing=routing,
            cache=CacheInfo(hit=True, similarity=round(hit.similarity, 4), matched_query=hit.entry.query),
            retrieval=RetrievalInfo(
                used=bool(saved["sources"]), origin=saved["origin"], sources=saved["sources"]
            ),
            compute=None,  # nothing had to run
            cost=CostInfo(
                input_tokens=saved["input_tokens"],
                output_tokens=saved["output_tokens"],
                actual_usd=0.0,
                baseline_usd=round(baseline, 8),
                saved_usd=round(baseline, 8),
                saved_percent=100.0,
                baseline_model=ladder[config.BASELINE_TIER].model_id,
            ),
            timings=Timings(
                route_ms=route_ms, cache_ms=cache_ms, retrieve_ms=0, generate_ms=0,
                total_ms=elapsed_ms(started),
            ),
            timestamp=now_iso(),
        )

    async def _answer_fresh(
        self, query, priority, routing, ladder, route_ms, cache_ms, started,
        provider=None, key="", problem=None,
    ) -> QueryResponse:
        step = time.perf_counter()
        # Our own documents come first. If they have nothing useful, ask Wikipedia.
        passages = self.good_passages(query, self.retriever.search(query))
        origin = "documents" if passages else "none"
        if not passages and self.web is not None:
            passages = await self.web.lookup(query)
            origin = "wikipedia" if passages else "none"
        retrieve_ms = elapsed_ms(step)

        step = time.perf_counter()
        # If this model is slow or busy, the next smaller one can step in.
        order = config.TIER_ORDER
        lower = order[order.index(routing.tier) - 1] if routing.tier != order[0] else None
        result = await self.llm.answer(
            query, routing.model_id, passages, provider, key, problem,
            fallback_model_id=ladder[lower].model_id if lower else None,
        )
        generate_ms = elapsed_ms(step)

        sources = [
            SourceInfo(title=p.title, snippet=shorten(p.text), score=round(p.score, 4), url=p.url)
            for p in passages
        ]

        actual = cost_usd(ladder, routing.tier, result.input_tokens, result.output_tokens)
        # The comparison assumes the large model would write an answer of the same length.
        baseline = cost_usd(ladder, config.BASELINE_TIER, result.input_tokens, result.output_tokens)
        saved = baseline - actual

        # Only remember answers that came out cleanly and actually say something.
        # An answer with no source and no model is just an apology, so skip it.
        if not result.warning and (passages or result.mode == "live"):
            self.cache.store(query, {
                "answer": result.text,
                "sources": sources,
                "origin": origin,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            })

        return QueryResponse(
            query=query,
            answer=result.text,
            answer_mode=result.mode,
            warning=result.warning,
            routing=routing,
            cache=CacheInfo(hit=False),
            retrieval=RetrievalInfo(used=bool(passages), origin=origin, sources=sources),
            compute=plan_compute(routing.tier, priority),
            cost=CostInfo(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                actual_usd=round(actual, 8),
                baseline_usd=round(baseline, 8),
                saved_usd=round(saved, 8),
                saved_percent=round(saved / baseline * 100, 1) if baseline else 0.0,
                baseline_model=ladder[config.BASELINE_TIER].model_id,
            ),
            timings=Timings(
                route_ms=route_ms, cache_ms=cache_ms, retrieve_ms=retrieve_ms,
                generate_ms=generate_ms, total_ms=elapsed_ms(started),
            ),
            timestamp=now_iso(),
        )

    @staticmethod
    def good_passages(query: str, hits: list[Hit]) -> list[Hit]:
        """Keep only passages that match well. Weak ones cost tokens and can mislead.

        A passage must also mention most of the question's key words. Otherwise
        "What does CPU stand for?" would match a document that never says "CPU".
        """
        wanted = important_stems(query)
        kept = []
        for hit in hits:
            if hit.score < config.RETRIEVAL_MIN_SCORE:
                continue
            found = {stem(word) for word in tokenize(hit.text + " " + hit.title)}
            if wanted and len(wanted & found) / len(wanted) <= 0.5:
                continue
            kept.append(hit)
        return kept


def shorten(text: str, limit: int = 180) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
