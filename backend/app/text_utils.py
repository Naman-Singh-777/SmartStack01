"""Small text helpers that several files share."""

import re

# Words that carry little meaning. We ignore them when comparing two questions.
STOPWORDS = frozenset(
    """a an the is are was were be been am do does did of to in on at for with and or but if so as
    by from it its this that these those what whats how why when where who which can could should
    would will you your i me my we our they them their there here about into than then just also
    not no yes please tell give s t""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into letters-and-digits words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def content_words(text: str) -> set[str]:
    """The meaningful words in a text, without stopwords."""
    return {word for word in tokenize(text) if word not in STOPWORDS}


def estimate_tokens(text: str) -> int:
    """Rough token count. English averages about four characters per token."""
    return max(1, round(len(text) / 4))


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by a space. Good enough for plain prose."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


# Everyday words that appear in many documents but say nothing about the topic.
# Without this list, "How do vaccines work?" would match any chunk containing "work".
GENERIC_WORDS = frozenset(
    """good bad better best work works working use used using make makes made need needs
    example examples way many much one two three new like get gets time part often usually
    different same simple common sometimes also even still really data percent""".split()
)
