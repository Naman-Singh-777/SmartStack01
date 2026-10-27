"""Looks things up on Wikipedia when our own documents have nothing useful.

Wikipedia's search is built for keywords, not questions. Searching for
"capital of France" returns "Capital punishment in France" first. So we do not
trust its order. Instead:

1. Search with a couple of cleaned-up versions of the question.
2. Collect the candidate pages and their opening paragraphs.
3. Re-rank the candidates ourselves: a good page has a title made of words from
   the question (like "France") and an opening that mentions most of the
   question's important words.
4. Drop pages that do not cover the question well. No source beats a wrong one.

No API key is needed. Wikipedia asks API users to send a descriptive User-Agent.
Wikipedia text is available under the CC BY-SA licence, which is why every
answer links back to the page it came from.
"""

import asyncio

import httpx

from .retriever import Hit
from .text_utils import (
    GENERIC_WORDS, STOPWORDS, content_words, split_sentences, stem, tokenize,
)

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SmartStackDemo/1.0 (https://github.com/Naman-Singh-777/SmartStack01)"
TIMEOUT_SECONDS = 6
MIN_COVERAGE = 0.5  # a page must mention at least half of the question's key words
PASSAGE_CHARS = 1400
KEEP_RATIO = 0.9  # drop a runner-up that scores far below the best page
# Pages about a band, film or song share names with ordinary topics ("The Vaccines").
MEDIA_WORDS = frozenset("band album song single film novel series musician rapper singer".split())
MEDIA_PENALTY = 0.3


def important_stems(question: str) -> set[str]:
    """The key words of a question, lightly normalised ("vaccines" becomes "vaccine")."""
    return {stem(word) for word in content_words(question) if word not in GENERIC_WORDS}


def search_variants(question: str) -> list[str]:
    """Two ways to phrase the search: key words only, and the question as written."""
    keywords = [w for w in tokenize(question) if w not in STOPWORDS and w not in GENERIC_WORDS]
    variants = []
    if keywords:
        variants.append(" ".join(keywords))
    plain = question.strip().rstrip("?.! ")
    if plain and plain.lower() not in variants:
        variants.append(plain)
    return variants


def parse_pages(data: dict) -> list[dict]:
    """Turn Wikipedia's JSON into a plain list of {title, extract, url, position}."""
    pages = data.get("query", {}).get("pages", {})
    parsed = []
    for page in pages.values():
        parsed.append({
            "title": page.get("title", ""),
            "extract": " ".join(page.get("extract", "").split()),
            "url": page.get("fullurl"),
            "position": page.get("index", 99),
        })
    return sorted(parsed, key=lambda page: page["position"])


def is_usable(page: dict) -> bool:
    text = page["extract"]
    if not text or "may refer to" in text or "disambiguation" in page["title"].lower():
        return False
    return True


def shorten_passage(text: str) -> str:
    """Keep whole sentences up to about PASSAGE_CHARS characters."""
    kept = ""
    for sentence in split_sentences(text):
        if kept and len(kept) + len(sentence) > PASSAGE_CHARS:
            break
        kept = f"{kept} {sentence}".strip()
    return kept or text[:PASSAGE_CHARS]


def rank_pages(question: str, pages: list[dict], variant_count: int) -> list[Hit]:
    """Score candidate pages ourselves. Each page dict needs a "votes" count."""
    wanted = important_stems(question)
    if not wanted:
        return []
    hits = []
    for page in pages:
        if not is_usable(page):
            continue
        title_words = {stem(w) for w in tokenize(page["title"]) if w not in STOPWORDS}
        text_words = {stem(w) for w in tokenize(page["extract"])} | title_words
        coverage = len(wanted & text_words) / len(wanted)
        if coverage < MIN_COVERAGE:
            continue
        # A title made only of question words ("France") beats one with extras
        # ("Capital punishment in France").
        purity = len(title_words & wanted) / len(title_words) if title_words else 0.0
        if page.get("exact"):
            purity = 1.0  # Wikipedia itself says this page is what the term means
        popularity = min(1.0, page["votes"] / variant_count)
        score = 0.5 * coverage + 0.35 * purity + 0.15 * popularity
        opening = {w for w in tokenize(page["extract"][:200])}
        if opening & MEDIA_WORDS and not (MEDIA_WORDS & {w for w in tokenize(question)}):
            score -= MEDIA_PENALTY
        hits.append(Hit(
            title=f"Wikipedia: {page['title']}",
            text=shorten_passage(page["extract"]),
            score=round(score, 4),
            word_score=round(coverage, 4),
            char_score=round(purity, 4),
            url=page["url"],
        ))
    hits.sort(key=lambda hit: -hit.score)
    return [hit for hit in hits if hit.score >= hits[0].score * KEEP_RATIO]


class WikipediaSearch:
    async def lookup(self, question: str, how_many: int = 2) -> list[Hit]:
        """Best Wikipedia passages for a question, or an empty list on any problem."""
        variants = search_variants(question)
        if not variants:
            return []
        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
            ) as client:
                searches = [self._search(client, v) for v in variants]
                # A single key word such as "CPU" is often a redirect to the real page
                # ("Central processing unit"), which plain search ranks too low.
                if " " not in variants[0]:
                    searches.append(self._by_title(client, variants[0]))
                results = await asyncio.gather(*searches)
        except httpx.HTTPError:
            return []  # Wikipedia being unreachable must never break an answer

        merged: dict[str, dict] = {}
        for pages in results:
            for page in pages:
                known = merged.setdefault(page["title"], {**page, "votes": 0})
                known["votes"] += 1
                known["exact"] = known.get("exact") or page.get("exact", False)
        return rank_pages(question, list(merged.values()), len(variants))[:how_many]

    @staticmethod
    async def _by_title(client: httpx.AsyncClient, term: str) -> list[dict]:
        response = await client.get(API_URL, params={
            "action": "query", "format": "json", "titles": term, "redirects": 1,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 10,
            "inprop": "url",
        })
        response.raise_for_status()
        pages = parse_pages(response.json())
        for page in pages:
            page["exact"] = True
        return pages

    @staticmethod
    async def _search(client: httpx.AsyncClient, query: str) -> list[dict]:
        response = await client.get(API_URL, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrlimit": 5,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exsentences": 10,
            "inprop": "url", "redirects": 1,
        })
        response.raise_for_status()
        return parse_pages(response.json())
