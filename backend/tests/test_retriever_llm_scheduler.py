import asyncio

from app.llm import LLMClient, extract_best_sentences
from app.retriever import Hit, Retriever
from app.scheduler import plan_compute


def test_retriever_finds_the_right_document():
    top = Retriever().search("How do spot instances lower my GPU bill?")[0]
    assert top.title == "Reducing cloud computing cost"


def test_retriever_scores_nonsense_low():
    good = Retriever().search("How does semantic caching avoid false hits?")[0].score
    bad = Retriever().search("Who won the 1998 football world cup final?")[0].score
    assert good > bad


def test_offline_answer_quotes_the_documents():
    passages = Retriever().search("Why is a small model cheaper than a large model?")
    text = extract_best_sentences("Why is a small model cheaper than a large model?", passages)
    assert text  # found at least one matching sentence


def test_offline_client_reports_offline_mode_and_token_counts():
    client = LLMClient(api_key="")
    passages = Retriever().search("What is a semantic cache?")
    result = asyncio.run(client.answer("What is a semantic cache?", "any-model", passages))
    assert result.mode == "offline"
    assert result.input_tokens > 0 and result.output_tokens > 0


def test_offline_client_admits_when_nothing_matches():
    client = LLMClient(api_key="")
    result = asyncio.run(client.answer("Who painted the Mona Lisa?", "any-model", []))
    assert "could not find" in result.text


def test_batch_work_is_cheaper_than_interactive_work():
    interactive = plan_compute("large", "interactive")
    batch = plan_compute("large", "batch")
    assert batch.price_per_hour < interactive.price_per_hour
    assert batch.pricing == "spot"


def test_interactive_work_respects_the_latency_budget():
    assert plan_compute("small", "interactive").latency_ms <= 80


def test_small_tier_uses_cheaper_hardware_than_large():
    small = plan_compute("small", "interactive")
    large = plan_compute("large", "interactive")
    assert small.price_per_hour < large.price_per_hour


def test_title_words_do_not_decide_which_sentence_is_quoted():
    passage = Hit(
        title="Wikipedia: France", score=0.9, word_score=1, char_score=1,
        text="France is a country in Europe. It has many regions. Its capital is Paris.",
    )
    assert extract_best_sentences("What is the capital of France?", [passage], how_many=1) == "Its capital is Paris."
