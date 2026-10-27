"""Measures how much SmartStack saves on a fixed set of questions.

Run it from the backend folder:   python -m benchmarks.run_benchmark

It asks every question once, in offline mode, and adds up the estimated cost with
SmartStack and the estimated cost of sending everything to the large model.
The saving depends on the mix of questions, so treat it as an example, not a promise.
"""

import asyncio
import json
from pathlib import Path

from app import config
from app.cache import SemanticCache
from app.classifier import ComplexityClassifier
from app.engine import SmartStack
from app.llm import LLMClient
from app.retriever import Retriever
from app.stats import Stats


async def main() -> None:
    queries = json.loads((Path(__file__).parent / "queries.json").read_text())["queries"]
    engine = SmartStack(
        classifier=ComplexityClassifier.from_file(config.TRAINING_FILE),
        cache=SemanticCache(),
        retriever=Retriever(),
        llm=LLMClient(server_keys={}, ollama_url=""),  # offline, so the benchmark needs no API key or money
        stats=Stats(),
    )
    for query in queries:
        response = await engine.handle(query, "interactive")
        print(f"{response.routing.tier:<7} {response.cost.saved_percent:>5.1f}% saved   {query[:70]}")

    summary = engine.stats.snapshot()
    print()
    print(f"questions            {summary['requests']}")
    print(f"tier usage           {summary['by_tier']}")
    print(f"with SmartStack      ${summary['actual_usd']:.6f}")
    print(f"always large model   ${summary['baseline_usd']:.6f}")
    print(f"saved                {summary['saved_percent']}%")


if __name__ == "__main__":
    asyncio.run(main())
