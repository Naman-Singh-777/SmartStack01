"""Every setting lives in this file. Change a number here and the whole app follows.

The prices below are examples, not an official price list. Check your provider's
pricing page and edit them before you trust any dollar figure the app shows.
"""

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
TRAINING_FILE = DATA_DIR / "training_queries.json"


@dataclass(frozen=True)
class ModelTier:
    """One rung on the ladder of models, from cheap to expensive."""

    key: str  # "small", "medium" or "large"
    label: str  # name shown in the UI
    model_id: str  # id sent to the provider in live mode
    input_per_million: float  # USD per 1,000,000 input tokens
    output_per_million: float  # USD per 1,000,000 output tokens


# Model ids can be changed with environment variables, prices by editing here.
TIERS: dict[str, ModelTier] = {
    "small": ModelTier("small", "Small", os.getenv("MODEL_SMALL", "claude-haiku-5-5"), 1.0, 5.0),
    "medium": ModelTier("medium", "Medium", os.getenv("MODEL_MEDIUM", "claude-sonnet-5-5"), 3.0, 15.0),
    "large": ModelTier("large", "Large", os.getenv("MODEL_LARGE", "claude-opus-5-5"), 15.0, 75.0),
}
TIER_ORDER = ["small", "medium", "large"]

# The classifier says "simple", "moderate" or "complex". This maps that to a tier.
LABEL_TO_TIER = {"simple": "small", "moderate": "medium", "complex": "large"}

# "Without SmartStack" means sending every question to this tier.
BASELINE_TIER = "large"

# If the classifier is less sure than this, we pick the next tier up to be safe.
MIN_CONFIDENCE = 0.55

# Semantic cache: how alike two questions must be to share an answer.
CACHE_SIMILARITY = 0.80  # character n-gram cosine similarity, 0 to 1
CACHE_WORD_OVERLAP = 0.60  # share of important words the two questions have in common
CACHE_MAX_ENTRIES = 500
CACHE_TTL_SECONDS = 60 * 60

# Retrieval: how many chunks to look at and how good a match must be to be used.
RETRIEVAL_TOP_K = 3
RETRIEVAL_MIN_SCORE = 0.13
WORD_WEIGHT = 0.6  # weight of exact word matching in the hybrid score
CHAR_WEIGHT = 0.4  # weight of fuzzy (spelling tolerant) matching

# Safety limits for a public API.
MAX_QUERY_CHARS = 1000
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60

# Look things up on Wikipedia when our own documents have nothing useful.
WEB_LOOKUP = os.getenv("WEB_LOOKUP", "on").lower() != "off"

# Live mode turns on when this key exists. Without it the app answers from documents only.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAX_OUTPUT_TOKENS = 500

# Origins allowed to call the API from a browser (comma separated).
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://naman-singh-777.github.io,http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
