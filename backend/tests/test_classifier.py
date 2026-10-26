import pytest

from app.classifier import ComplexityClassifier
from app.config import TRAINING_FILE


@pytest.fixture(scope="module")
def classifier():
    return ComplexityClassifier.from_file(TRAINING_FILE)


def test_probabilities_add_up_to_one(classifier):
    prediction = classifier.predict("What is the capital of Italy?")
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert prediction.label in {"simple", "moderate", "complex"}


def test_short_fact_is_simple(classifier):
    assert classifier.predict("What is the capital of Spain?").label == "simple"


def test_design_request_is_complex(classifier):
    question = (
        "Design a fault tolerant job queue, compare two storage options, "
        "and describe how it recovers after a data center outage."
    )
    assert classifier.predict(question).label == "complex"


def test_accuracy_on_unseen_questions_is_reasonable(classifier):
    # Five fold cross validation on hand written data. A coin flip between three
    # labels scores about 0.33, so 0.7 means the model learned something real.
    assert classifier.cross_validated_accuracy() >= 0.7
