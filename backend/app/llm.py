"""Writes the final answer.

Two modes:

* live: an API key is set, so we call the real model that the router picked.
* offline: no key. We cannot write new text, so we quote the sentences from the
  retrieved documents that best match the question. The app still works and the
  response says clearly that it ran offline.

We call the API with plain HTTP (httpx) instead of an SDK to keep the code short
and easy to follow.
"""

from dataclasses import dataclass

import httpx

from . import config
from .retriever import Hit
from .text_utils import content_words, estimate_tokens, split_sentences

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question clearly and briefly. "
    "If context passages are provided and they help, use them. Do not invent facts."
)
NO_ANSWER = (
    "I could not find anything about that in the knowledge base. This server has no language "
    "model key, so it can only answer from the documents in backend/app/data/knowledge."
)


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    mode: str  # "live" or "offline"
    warning: str | None = None


def build_prompt(query: str, passages: list[Hit]) -> str:
    if not passages:
        return query
    context = "\n\n".join(f"[{i}] {p.title}: {p.text}" for i, p in enumerate(passages, start=1))
    return f"Context:\n{context}\n\nQuestion: {query}"


class LLMClient:
    def __init__(self, api_key: str = config.ANTHROPIC_API_KEY):
        self.api_key = api_key

    @property
    def mode(self) -> str:
        return "live" if self.api_key else "offline"

    async def answer(self, query: str, model_id: str, passages: list[Hit]) -> LLMResult:
        prompt = build_prompt(query, passages)
        if not self.api_key:
            return self._offline_answer(query, prompt, passages)
        try:
            return await self._live_answer(prompt, model_id)
        except (httpx.HTTPError, KeyError, IndexError) as error:
            # Never let a provider outage break the demo. Fall back and say so.
            result = self._offline_answer(query, prompt, passages)
            result.warning = f"The live model call failed ({type(error).__name__}), so this is an offline answer."
            return result

    async def _live_answer(self, prompt: str, model_id: str) -> LLMResult:
        body = {
            "model": model_id,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(API_URL, json=body, headers=headers)
            response.raise_for_status()
        data = response.json()
        return LLMResult(
            text=data["content"][0]["text"],
            input_tokens=data["usage"]["input_tokens"],  # real counts from the provider
            output_tokens=data["usage"]["output_tokens"],
            mode="live",
        )

    def _offline_answer(self, query: str, prompt: str, passages: list[Hit]) -> LLMResult:
        text = extract_best_sentences(query, passages) or NO_ANSWER
        return LLMResult(
            text=text,
            input_tokens=estimate_tokens(SYSTEM_PROMPT + prompt),  # estimated, not measured
            output_tokens=estimate_tokens(text),
            mode="offline",
        )


def extract_best_sentences(query: str, passages: list[Hit], how_many: int = 2) -> str:
    """Pick the sentences that share the most important words with the question."""
    wanted = content_words(query)
    scored = []
    position = 0
    for passage in passages:
        for sentence in split_sentences(passage.text):
            overlap = len(wanted & content_words(sentence))
            if overlap > 0:
                scored.append((overlap, position, sentence))
            position += 1
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:how_many]
    best.sort(key=lambda item: item[1])  # read them in their original order
    return " ".join(sentence for _, _, sentence in best)
