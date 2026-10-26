"""Judges how hard a question is, using a small machine learning model.

How it works, in plain words:

1. data/training_queries.json holds example questions, each labelled
   "simple", "moderate" or "complex".
2. TF-IDF turns each question into numbers based on the words it uses.
3. A few extra numbers describe the shape of the question: how long it is,
   how many steps it asks for, whether it contains code.
4. Logistic regression learns which numbers go with which label.

Training takes a fraction of a second, so the app trains at startup and we do
not need to save a model file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

# Words that usually show up when a question asks for several things at once.
MULTI_STEP_HINTS = (
    "step by step", "trade-off", "tradeoff", "including", "covering", "justify",
    "recommend", "compare", "analyze", "design", "plan", "evaluate", "then",
)


class ShapeFeatures(BaseEstimator, TransformerMixin):
    """Turns a text into four numbers between 0 and 1 that describe its shape."""

    def fit(self, texts, labels=None):
        return self  # nothing to learn, the numbers are computed with fixed rules

    def transform(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            hint_count = sum(lowered.count(hint) for hint in MULTI_STEP_HINTS)
            looks_like_code = "```" in text or "def " in text or "{" in text
            rows.append([
                min(len(text.split()) / 50, 1.0),  # long questions tend to be harder
                min(hint_count / 3, 1.0),  # asking for several things at once
                min(text.count(",") / 4, 1.0),  # many commas usually mean many requirements
                1.0 if looks_like_code else 0.0,
            ])
        return np.array(rows)


@dataclass
class Prediction:
    label: str  # simple, moderate or complex
    confidence: float  # probability of the chosen label
    probabilities: dict[str, float]  # probability of every label


def load_examples(path: Path) -> tuple[list[str], list[str]]:
    """Read the training file and return (texts, labels)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    texts, labels = [], []
    for label, questions in data.items():
        texts.extend(questions)
        labels.extend([label] * len(questions))
    return texts, labels


def build_model() -> Pipeline:
    """The untrained model: word features + shape features, then logistic regression."""
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("shape", ShapeFeatures()),
    ])
    return Pipeline([
        ("features", features),
        ("classify", LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
    ])


class ComplexityClassifier:
    def __init__(self, texts: list[str], labels: list[str]):
        self.texts = texts
        self.labels = labels
        self.model = build_model()
        self.model.fit(texts, labels)

    @classmethod
    def from_file(cls, path: Path) -> "ComplexityClassifier":
        texts, labels = load_examples(path)
        return cls(texts, labels)

    def predict(self, text: str) -> Prediction:
        probabilities = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        best = int(np.argmax(probabilities))
        return Prediction(
            label=classes[best],
            confidence=float(probabilities[best]),
            probabilities={name: float(p) for name, p in zip(classes, probabilities)},
        )

    def cross_validated_accuracy(self, folds: int = 5) -> float:
        """Train on part of the examples, test on the rest, repeat, and average.

        This is an honest estimate of accuracy on questions the model has not seen.
        """
        scores = cross_val_score(build_model(), self.texts, self.labels, cv=folds)
        return float(scores.mean())
