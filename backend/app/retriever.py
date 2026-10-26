"""Finds the passages in our documents that best match a question (the "R" in RAG).

Steps:
1. At startup, read every .md file in data/knowledge and cut it into chunks.
2. Turn each chunk into two sets of numbers (TF-IDF vectors):
   - word vectors reward chunks that use the same words as the question
   - character vectors reward chunks with similar letter patterns, which
     forgives typos and word forms like "cache" versus "caching"
3. For a question, score every chunk both ways and mix the scores. This
   is called hybrid search. The best few chunks win.

To teach SmartStack something new, drop another .md file into data/knowledge.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .text_utils import GENERIC_WORDS

MIN_CHUNK_CHARS = 250  # short paragraphs are glued together until they reach this size


@dataclass
class Chunk:
    title: str  # the document the chunk came from
    text: str


@dataclass
class Hit:
    title: str
    text: str
    score: float  # hybrid score, 0 to 1
    word_score: float
    char_score: float


def load_chunks(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        paragraphs = [p.strip() for p in "\n".join(lines[1:]).split("\n\n") if p.strip()]
        current = ""
        for paragraph in paragraphs:
            current = f"{current} {paragraph}".strip()
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(title, current))
                current = ""
        if current:  # leftover text at the end of the file
            chunks.append(Chunk(title, current))
    return chunks


class Retriever:
    def __init__(self, folder: Path = config.KNOWLEDGE_DIR):
        self.chunks = load_chunks(folder)
        if not self.chunks:
            raise ValueError(f"No .md documents found in {folder}")
        # The title is searchable too: "Semantic caching" should help a question about caching.
        texts = [f"{chunk.title}. {chunk.text}" for chunk in self.chunks]
        self._word_vectorizer = TfidfVectorizer(
            stop_words=sorted(ENGLISH_STOP_WORDS | GENERIC_WORDS), sublinear_tf=True
        )
        self._char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._word_matrix = self._word_vectorizer.fit_transform(texts)
        self._char_matrix = self._char_vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[Hit]:
        word_scores = cosine_similarity(self._word_vectorizer.transform([query]), self._word_matrix)[0]
        char_scores = cosine_similarity(self._char_vectorizer.transform([query]), self._char_matrix)[0]
        mixed = config.WORD_WEIGHT * word_scores + config.CHAR_WEIGHT * char_scores
        best_first = np.argsort(mixed)[::-1][:top_k]
        return [
            Hit(
                title=self.chunks[i].title,
                text=self.chunks[i].text,
                score=float(mixed[i]),
                word_score=float(word_scores[i]),
                char_score=float(char_scores[i]),
            )
            for i in best_first
        ]
