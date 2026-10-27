import pytest
from fastapi.testclient import TestClient

from app.llm import LLMClient, LLMResult
from app.main import app
from app.retriever import Hit
from app.ratelimit import RateLimiter


@pytest.fixture()
def client(monkeypatch):
    with TestClient(app) as test_client:
        engine = test_client.app.state.engine
        engine.llm.server_keys = {}  # force offline mode in tests
        engine.llm.ollama_url = ""
        engine.web = None  # never touch the network in tests
        yield test_client


def ask(client, text, priority="interactive"):
    return client.post("/api/query", json={"query": text, "priority": priority})


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_mode"] == "offline"


def test_models_lists_three_tiers(client):
    body = client.get("/api/models").json()
    assert [t["key"] for t in body["tiers"]] == ["small", "medium", "large"]
    assert body["classifier_accuracy"] >= 0.7


def test_query_returns_every_section(client):
    data = ask(client, "How do spot instances lower my GPU bill?").json()
    for key in ("answer", "routing", "cache", "retrieval", "compute", "cost", "timings"):
        assert key in data
    assert data["retrieval"]["used"] is True
    assert data["cost"]["actual_usd"] <= data["cost"]["baseline_usd"]


def test_asking_twice_hits_the_cache_and_costs_nothing(client):
    ask(client, "What does a semantic cache do?")
    second = ask(client, "what does a semantic cache do").json()
    assert second["answer_mode"] == "cache"
    assert second["cost"]["actual_usd"] == 0
    assert second["compute"] is None


def test_hard_question_goes_to_a_bigger_model_than_an_easy_one(client):
    easy = ask(client, "What does CPU stand for?").json()["routing"]["tier"]
    hard = ask(
        client,
        "Design a multi region database failover plan, compare three replication "
        "strategies, and explain the risks and rollback steps for each.",
    ).json()["routing"]["tier"]
    order = ["small", "medium", "large"]
    assert order.index(hard) > order.index(easy)


def test_batch_priority_plans_spot_machines(client):
    data = ask(client, "Explain how model routing picks a tier", priority="batch").json()
    assert data["compute"]["pricing"] == "spot"


def test_blank_query_is_rejected(client):
    assert ask(client, "   ").status_code == 422


def test_overlong_query_is_rejected(client):
    assert ask(client, "x" * 1001).status_code == 422


def test_stats_count_requests(client):
    before = client.get("/api/stats").json()["requests"]
    ask(client, "What is retrieval augmented generation?")
    assert client.get("/api/stats").json()["requests"] == before + 1


def test_rate_limit_blocks_the_extra_request(client):
    client.app.state.limiter = RateLimiter(limit=2, window_seconds=60)
    assert ask(client, "first question about tokens").status_code == 200
    assert ask(client, "second question about caching").status_code == 200
    assert ask(client, "third question about routing").status_code == 429


class FakeWeb:
    """Stands in for Wikipedia so the test needs no internet."""

    async def lookup(self, question):
        return [Hit(
            title="Wikipedia: Paris", text="Paris is the capital and largest city of France.",
            score=0.9, word_score=1.0, char_score=1.0, url="https://en.wikipedia.org/wiki/Paris",
        )]


def test_wikipedia_is_used_when_our_documents_have_nothing(client):
    client.app.state.engine.web = FakeWeb()
    data = ask(client, "What is the capital of France?").json()
    assert data["retrieval"]["origin"] == "wikipedia"
    assert data["retrieval"]["sources"][0]["url"].startswith("https://en.wikipedia.org/")
    assert "capital" in data["answer"]


def test_our_own_documents_win_over_wikipedia(client):
    client.app.state.engine.web = FakeWeb()
    data = ask(client, "How do spot instances lower my GPU bill?").json()
    assert data["retrieval"]["origin"] == "documents"


def test_an_answer_with_no_source_is_not_cached(client):
    first = ask(client, "Who painted the Mona Lisa?").json()
    second = ask(client, "Who painted the Mona Lisa?").json()
    assert first["retrieval"]["origin"] == "none"
    assert second["answer_mode"] == "offline"  # not "cache"


def test_visitor_key_switches_on_live_mode_for_that_request(client, monkeypatch):
    async def fake_live(self, provider, prompt, model_id, key):
        assert (provider, key) == ("anthropic", "sk-ant-visitor")
        return LLMResult(text="live text", input_tokens=10, output_tokens=5, mode="live")

    monkeypatch.setattr(LLMClient, "_live_answer", fake_live)
    response = client.post(
        "/api/query", json={"query": "Who painted the Mona Lisa?"}, headers={"X-LLM-Key": "sk-ant-visitor"}
    )
    assert response.json()["answer_mode"] == "live"
    # Another question without the header goes back to offline mode.
    assert ask(client, "Who wrote Hamlet?").json()["answer_mode"] == "offline"
    # The live answer was cached, so asking the same thing again is free for everyone.
    assert ask(client, "Who painted the Mona Lisa?").json()["answer_mode"] == "cache"


def test_a_gemini_key_makes_the_router_use_gemini_models(client, monkeypatch):
    seen = {}

    async def fake_live(self, provider, prompt, model_id, key):
        seen.update(provider=provider, model=model_id)
        return LLMResult(text="live text", input_tokens=10, output_tokens=5, mode="live")

    monkeypatch.setattr(LLMClient, "_live_answer", fake_live)
    data = client.post(
        "/api/query", json={"query": "Who wrote Hamlet?"}, headers={"X-LLM-Key": "AIzaFAKE"}
    ).json()
    assert seen["provider"] == "gemini"
    assert seen["model"].startswith("gemini-")
    assert data["routing"]["model_id"] == seen["model"]


def test_an_unrecognised_key_is_ignored_with_a_clear_warning(client):
    data = client.post(
        "/api/query", json={"query": "Who wrote Hamlet?"}, headers={"X-LLM-Key": "not-a-real-key"}
    ).json()
    assert data["answer_mode"] == "offline"
    assert "does not look like" in data["warning"]


def test_a_related_question_gets_a_fresh_model_answer_with_the_cache_as_background(client, monkeypatch):
    prompts = []

    async def fake_live(self, provider, prompt, model_id, key):
        prompts.append(prompt)
        return LLMResult(text=f"answer {len(prompts)}", input_tokens=10, output_tokens=5, mode="live")

    monkeypatch.setattr(LLMClient, "_live_answer", fake_live)
    headers = {"X-LLM-Key": "AIzaFAKE"}
    first = client.post("/api/query", json={"query": "indian names example"}, headers=headers).json()
    second = client.post(
        "/api/query", json={"query": "indian names example for kid naming"}, headers=headers
    ).json()
    assert first["answer"] == "answer 1"
    assert second["answer_mode"] == "live" and second["answer"] == "answer 2"  # not the cached answer
    assert second["cache"]["hit"] is False
    assert second["cache"]["context_from"] == "indian names example"
    assert "indian names example" in prompts[1] and "answer 1" in prompts[1]
    # The very same question is still answered from the cache for free.
    third = client.post("/api/query", json={"query": "Indian names example"}, headers=headers).json()
    assert third["answer_mode"] == "cache"
