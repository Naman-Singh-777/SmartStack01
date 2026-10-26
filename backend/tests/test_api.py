import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ratelimit import RateLimiter


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr("app.llm.config.ANTHROPIC_API_KEY", "")
    with TestClient(app) as test_client:
        test_client.app.state.engine.llm.api_key = ""  # force offline mode in tests
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
