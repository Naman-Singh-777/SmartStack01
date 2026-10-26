"""A cache that recognises questions that mean the same thing, not only identical ones.

"What's the capital of France?" and "what is the capital of france" should share
one answer. But "capital of France" and "capital of Spain" must not, even though
they look almost the same letter by letter. So a cached answer is reused only
when BOTH checks pass:

1. Letter check: the questions share most of their 3 to 4 letter chunks
   (cosine similarity of character n-grams). This forgives typos and word order.
2. Word check: most of the important words match (Jaccard overlap).
   This catches the "France versus Spain" problem.

A wrong cached answer is worse than a missed cache hit, so both checks must pass.
"""

import time
from dataclasses import dataclass

from sklearn.feature_extraction.text import HashingVectorizer

from . import config
from .text_utils import content_words


@dataclass
class CacheEntry:
    query: str
    vector: object  # sparse row vector of character n-grams
    words: set[str]
    payload: dict  # whatever the caller wants to remember about the answer
    created_at: float


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float


def word_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard overlap: shared words divided by all words. 1.0 means identical sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SemanticCache:
    def __init__(
        self,
        similarity: float = config.CACHE_SIMILARITY,
        word_overlap_needed: float = config.CACHE_WORD_OVERLAP,
        max_entries: int = config.CACHE_MAX_ENTRIES,
        ttl_seconds: float = config.CACHE_TTL_SECONDS,
    ):
        self.similarity = similarity
        self.word_overlap_needed = word_overlap_needed
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: list[CacheEntry] = []
        # HashingVectorizer needs no training, so it works from the very first query.
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), n_features=2**14, alternate_sign=False
        )

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, query: str) -> CacheHit | None:
        self._drop_expired()
        vector = self._vectorizer.transform([query.lower()])
        words = content_words(query)
        best: CacheHit | None = None
        for entry in self._entries:
            letters = float(vector.multiply(entry.vector).sum())  # cosine, vectors are normalised
            if letters < self.similarity:
                continue
            if word_overlap(words, entry.words) < self.word_overlap_needed:
                continue
            if best is None or letters > best.similarity:
                best = CacheHit(entry, letters)
        return best

    def store(self, query: str, payload: dict) -> None:
        # Replace an older answer to the very same question instead of keeping both.
        self._entries = [e for e in self._entries if e.query.lower() != query.lower()]
        self._entries.append(
            CacheEntry(
                query=query,
                vector=self._vectorizer.transform([query.lower()]),
                words=content_words(query),
                payload=payload,
                created_at=time.time(),
            )
        )
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)  # forget the oldest

    def _drop_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        self._entries = [e for e in self._entries if e.created_at >= cutoff]
