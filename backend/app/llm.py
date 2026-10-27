"""Writes the final answer.

Two modes:

* live: a key is available, so we call the real model that the router picked.
  Three providers are supported:
    - Gemini (a free key from Google AI Studio works),
    - Anthropic,
    - Ollama, which runs models on your own computer (set OLLAMA_URL).
  The key can come from the server (GEMINI_API_KEY or ANTHROPIC_API_KEY) or from
  the visitor, who can paste their own key on the Try it page. We tell the
  providers apart by the key's shape ("AIza..." is Gemini, "sk-ant-..." is
  Anthropic). A visitor's key is used for that one request only and is never
  stored or logged.
* offline: no key. We cannot write new text, so we quote the sentences from the
  retrieved sources that best match the question. The response says clearly that
  it ran offline.

We call every API with plain HTTP (httpx) instead of an SDK to keep the code short
and easy to follow.
"""

from dataclasses import dataclass

import httpx

from . import config
from .retriever import Hit
from .text_utils import content_words, estimate_tokens, split_sentences, stem, tokenize

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question clearly and briefly. "
    "Context passages may be provided. They come from an automatic search, so they can be "
    "about the wrong thing. Use them only when they match what the question most likely "
    "means, and mention the passage number like [1]. Otherwise ignore them and answer from "
    "your own knowledge. Do not invent facts, and say so when you are not sure."
)
NO_ANSWER = (
    "I could not find a reliable source for that. No language model key is set, so I can only "
    "quote sources such as Wikipedia and the project documents. Paste a free Gemini key under "
    "\"Use your own language model key\" above, or set GEMINI_API_KEY on the server, "
    "for full answers."
)


NO_ANSWER_AFTER_FAILURE = (
    "I could not find a source for that, and the language model could not answer right now. "
    "Please try again in a moment."
)
# Failures worth one retry on a smaller, faster model: slow, busy or briefly broken.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def worth_retrying(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS
    return isinstance(error, httpx.TransportError)  # includes timeouts


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    mode: str  # "live" or "offline"
    warning: str | None = None


def provider_for_key(key: str) -> str | None:
    """Guess the provider from the look of a key. None means we do not recognise it."""
    if key.startswith("AIza"):
        return "gemini"
    if key.startswith("sk-ant-"):
        return "anthropic"
    return None


def build_prompt(query: str, passages: list[Hit], earlier: tuple[str, str] | None = None) -> str:
    """The question, plus any context. `earlier` is a related (question, answer) from the cache."""
    parts = []
    if earlier:
        parts.append(
            "Earlier related question and answer. It may be about something different, "
            "so use it only for background:\n"
            f"Q: {earlier[0]}\nA: {earlier[1]}"
        )
    if passages:
        parts.append(
            "Context:\n" + "\n\n".join(f"[{i}] {p.title}: {p.text}" for i, p in enumerate(passages, start=1))
        )
    if not parts:
        return query
    return "\n\n".join(parts) + f"\n\nQuestion: {query}"


def explain_failure(error: Exception) -> str:
    """A short, honest reason a live call failed."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        body = error.response.text.lower()
        # Gemini reports a bad key as a 400 that mentions the key.
        if status in (401, 403) or (status == 400 and "api key" in body):
            return "The model provider rejected the API key, so this is an offline answer."
        if status == 429:
            return "The model provider is rate limiting this key, so this is an offline answer."
        return f"The model provider returned an error ({status}), so this is an offline answer."
    if isinstance(error, (KeyError, IndexError, ValueError)):
        return "The model provider sent a reply we could not read, so this is an offline answer."
    if isinstance(error, httpx.TimeoutException):
        return "The model provider took too long to answer, so this is an offline answer."
    return "The model provider could not be reached, so this is an offline answer."


class LLMClient:
    def __init__(self, server_keys: dict[str, str] | None = None, ollama_url: str | None = None):
        # Keys set on the server, by provider. Empty values are ignored.
        self.server_keys = {
            name: key
            for name, key in (
                server_keys
                if server_keys is not None
                else {"gemini": config.GEMINI_API_KEY, "anthropic": config.ANTHROPIC_API_KEY}
            ).items()
            if key
        }
        self.ollama_url = config.OLLAMA_URL if ollama_url is None else ollama_url

    @property
    def mode(self) -> str:
        return "live" if self.server_keys or self.ollama_url else "offline"

    @property
    def server_provider(self) -> str | None:
        """The provider the server itself will use when a visitor brings no key."""
        for name in ("gemini", "anthropic"):
            if name in self.server_keys:
                return name
        return "ollama" if self.ollama_url else None

    def choose(self, visitor_key: str | None = None) -> tuple[str | None, str, str | None]:
        """Decide who answers this request: (provider, key, problem).

        provider is None when the answer must be offline. problem explains why when
        a visitor's key was the reason.
        """
        if visitor_key:
            provider = provider_for_key(visitor_key.strip())
            if provider:
                return provider, visitor_key.strip(), None
            problem = "That key does not look like a Gemini (AIza...) or Anthropic (sk-ant-...) key, so this is an offline answer."
            fallback = self.server_provider
            if fallback is None:
                return None, "", problem
            return fallback, self.server_keys.get(fallback, ""), None
        provider = self.server_provider
        return provider, self.server_keys.get(provider or "", ""), None

    async def answer(
        self, query: str, model_id: str, passages: list[Hit],
        provider: str | None = None, key: str = "", problem: str | None = None,
        fallback_model_id: str | None = None, earlier: tuple[str, str] | None = None,
    ) -> LLMResult:
        prompt = build_prompt(query, passages, earlier)
        if provider is None:
            result = self._offline_answer(query, prompt, passages)
            result.warning = problem
            return result
        note = None
        try:
            try:
                return await self._live_answer(provider, prompt, model_id, key)
            except (httpx.HTTPError, ValueError) as error:
                # The big model can be slow or busy. Try the next smaller one once.
                if fallback_model_id and worth_retrying(error):
                    note = f"{model_id} did not answer in time, so {fallback_model_id} answered instead."
                    result = await self._live_answer(provider, prompt, fallback_model_id, key)
                    result.warning = note
                    return result
                raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            # Never let a provider problem break the demo. Fall back and say why.
            result = self._offline_answer(query, prompt, passages, live_failed=True)
            result.warning = explain_failure(error)
            return result

    async def _live_answer(self, provider: str, prompt: str, model_id: str, key: str) -> LLMResult:
        calls = {
            "gemini": self._call_gemini,
            "anthropic": self._call_anthropic,
            "ollama": self._call_ollama,
        }
        return await calls[provider](prompt, model_id, key)

    @staticmethod
    async def _post(url: str, body: dict, headers: dict) -> dict:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
        return response.json()

    async def _call_gemini(self, prompt: str, model_id: str, key: str) -> LLMResult:
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": config.GEMINI_MAX_OUTPUT_TOKENS},
        }
        data = await self._post(
            GEMINI_URL.format(model=model_id), body, {"x-goog-api-key": key}
        )
        parts = data["candidates"][0]["content"]["parts"]
        # Skip the model's "thinking" parts and keep only the answer text.
        text = "".join(p["text"] for p in parts if p.get("text") and not p.get("thought")).strip()
        if not text:
            raise ValueError("empty reply")
        usage = data.get("usageMetadata", {})
        return LLMResult(
            text=text,
            input_tokens=usage.get("promptTokenCount", 0),  # real counts from the provider
            output_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            mode="live",
        )

    async def _call_anthropic(self, prompt: str, model_id: str, key: str) -> LLMResult:
        body = {
            "model": model_id,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        data = await self._post(ANTHROPIC_URL, body, headers)
        return LLMResult(
            text=data["content"][0]["text"],
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
            mode="live",
        )

    async def _call_ollama(self, prompt: str, model_id: str, key: str) -> LLMResult:
        """Ollama speaks the common "chat completions" format on /v1/chat/completions."""
        body = {
            "model": model_id,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        data = await self._post(f"{self.ollama_url}/v1/chat/completions", body, {})
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"].strip(),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            mode="live",
        )

    def _offline_answer(
        self, query: str, prompt: str, passages: list[Hit], live_failed: bool = False
    ) -> LLMResult:
        text = extract_best_sentences(query, passages) or (
            NO_ANSWER_AFTER_FAILURE if live_failed else NO_ANSWER
        )
        return LLMResult(
            text=text,
            input_tokens=estimate_tokens(SYSTEM_PROMPT + prompt),  # estimated, not measured
            output_tokens=estimate_tokens(text),
            mode="offline",
        )


def stems(text: str) -> set[str]:
    return {stem(word) for word in content_words(text)}


MEASURE_WORDS = frozenset("tall high long big large old far deep heavy wide many much fast".split())


def extract_best_sentences(query: str, passages: list[Hit], how_many: int = 2) -> str:
    """Pick the sentences that share the most important words with the question.

    Only sources that score close to the best one are used. Words from the
    source's own title are ignored, because they appear in nearly every
    sentence ("France" tells us nothing about which sentence names the capital).
    For "how tall..." questions, sentences that contain a number get a boost.
    If no sentence shares a word, fall back to the opening of the best source,
    which is usually its definition.
    """
    if not passages:
        return ""
    best_score = passages[0].score
    wanted_all = stems(query)
    asks_for_number = bool(content_words(query) & MEASURE_WORDS) and "how" in tokenize(query)
    scored = []
    position = 0
    for passage in passages:
        if passage.score < best_score * 0.9:
            continue
        title_stems = stems(passage.title.removeprefix("Wikipedia: "))
        wanted = (wanted_all - title_stems) or wanted_all
        for sentence in split_sentences(passage.text):
            overlap = len(wanted & stems(sentence))
            if asks_for_number and any(ch.isdigit() for ch in sentence):
                overlap += 1
            if overlap > 0:
                scored.append((overlap, position, sentence))
            position += 1
    if not scored:
        return " ".join(split_sentences(passages[0].text)[:2])
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:how_many]
    best.sort(key=lambda item: item[1])  # read them in their original order
    return " ".join(sentence for _, _, sentence in best)
