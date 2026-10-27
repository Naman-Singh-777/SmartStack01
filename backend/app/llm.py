"""Writes the final answer.

Two modes:

* live: a key is available, so we call the real model that the router picked.
  The key can come from the server (ANTHROPIC_API_KEY) or from the visitor, who
  can paste their own key on the Try it page. A visitor's key is used for that one
  request only and is never stored or logged.
* offline: no key. We cannot write new text, so we quote the sentences from the
  retrieved sources that best match the question. The response says clearly that
  it ran offline.

We call the API with plain HTTP (httpx) instead of an SDK to keep the code short
and easy to follow.
"""

from dataclasses import dataclass

import httpx

from . import config
from .retriever import Hit
from .text_utils import content_words, estimate_tokens, split_sentences, stem

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question clearly and briefly. "
    "Context passages may be provided. Use them when they help and mention the passage "
    "number like [1]. If they do not help, answer from your own knowledge. "
    "Do not invent facts, and say so when you are not sure."
)
NO_ANSWER = (
    "I could not find a reliable source for that. No language model key is set, so I can only "
    "quote sources such as Wikipedia and the project documents. Paste your own key under "
    "\"Use your own language model key\" above, or set ANTHROPIC_API_KEY on the server, "
    "for full answers."
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


def explain_failure(error: Exception) -> str:
    """A short, honest reason a live call failed."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in (401, 403):
            return "The model provider rejected the API key, so this is an offline answer."
        if status == 429:
            return "The model provider is rate limiting this key, so this is an offline answer."
        return f"The model provider returned an error ({status}), so this is an offline answer."
    return "The model provider could not be reached, so this is an offline answer."


class LLMClient:
    def __init__(self, api_key: str = config.ANTHROPIC_API_KEY):
        self.api_key = api_key

    @property
    def mode(self) -> str:
        return "live" if self.api_key else "offline"

    async def answer(
        self, query: str, model_id: str, passages: list[Hit], api_key: str | None = None
    ) -> LLMResult:
        key = api_key or self.api_key
        prompt = build_prompt(query, passages)
        if not key:
            return self._offline_answer(query, prompt, passages)
        try:
            return await self._live_answer(prompt, model_id, key)
        except (httpx.HTTPError, KeyError, IndexError) as error:
            # Never let a provider problem break the demo. Fall back and say why.
            result = self._offline_answer(query, prompt, passages)
            result.warning = explain_failure(error)
            return result

    async def _live_answer(self, prompt: str, model_id: str, key: str) -> LLMResult:
        body = {
            "model": model_id,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": key,
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


def stems(text: str) -> set[str]:
    return {stem(word) for word in content_words(text)}


def extract_best_sentences(query: str, passages: list[Hit], how_many: int = 2) -> str:
    """Pick the sentences that share the most important words with the question.

    If no sentence shares a word, fall back to the opening of the best source,
    which is usually its definition.
    """
    if not passages:
        return ""
    wanted = stems(query)
    scored = []
    position = 0
    for passage in passages:
        for sentence in split_sentences(passage.text):
            overlap = len(wanted & stems(sentence))
            if overlap > 0:
                scored.append((overlap, position, sentence))
            position += 1
    if not scored:
        return " ".join(split_sentences(passages[0].text)[:2])
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:how_many]
    best.sort(key=lambda item: item[1])  # read them in their original order
    return " ".join(sentence for _, _, sentence in best)
