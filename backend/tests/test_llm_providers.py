import asyncio

import httpx

from app.llm import LLMClient, explain_failure, provider_for_key


def test_keys_are_matched_to_providers_by_their_shape():
    assert provider_for_key("AIzaSyExample") == "gemini"
    assert provider_for_key("sk-ant-example") == "anthropic"
    assert provider_for_key("something-else") is None


def test_the_server_prefers_gemini_then_anthropic_then_ollama():
    both = LLMClient(server_keys={"gemini": "AIza1", "anthropic": "sk-ant-1"}, ollama_url="http://x")
    assert both.choose() [0] == "gemini"
    assert LLMClient(server_keys={"anthropic": "sk-ant-1"}, ollama_url="http://x").choose()[0] == "anthropic"
    assert LLMClient(server_keys={}, ollama_url="http://x").choose()[0] == "ollama"
    assert LLMClient(server_keys={}, ollama_url="").choose()[0] is None


def test_a_visitor_key_beats_the_server_key():
    client = LLMClient(server_keys={"anthropic": "sk-ant-server"}, ollama_url="")
    assert client.choose("AIzaVisitor") == ("gemini", "AIzaVisitor", None)


def test_gemini_reply_is_read_and_thinking_parts_are_skipped(monkeypatch):
    sent = {}

    async def fake_post(url, body, headers):
        sent.update(url=url, headers=headers)
        return {
            "candidates": [{"content": {"parts": [
                {"text": "hidden reasoning", "thought": True},
                {"text": "Paris is the capital."},
            ]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "thoughtsTokenCount": 30},
        }

    monkeypatch.setattr(LLMClient, "_post", staticmethod(fake_post))
    client = LLMClient(server_keys={}, ollama_url="")
    result = asyncio.run(client.answer("capital?", "gemini-x", [], "gemini", "AIzaKey"))
    assert result.text == "Paris is the capital."
    assert (result.input_tokens, result.output_tokens) == (12, 35)
    assert "gemini-x:generateContent" in sent["url"]
    assert sent["headers"] == {"x-goog-api-key": "AIzaKey"}


def test_ollama_reply_is_read(monkeypatch):
    async def fake_post(url, body, headers):
        assert url == "http://localhost:11434/v1/chat/completions"
        return {"choices": [{"message": {"content": " Hello "}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    monkeypatch.setattr(LLMClient, "_post", staticmethod(fake_post))
    client = LLMClient(server_keys={}, ollama_url="http://localhost:11434")
    result = asyncio.run(client.answer("hi", "llama3.2:3b", [], "ollama", ""))
    assert result.text == "Hello"


def test_an_unreadable_reply_falls_back_to_offline_with_a_reason(monkeypatch):
    async def fake_post(url, body, headers):
        return {"candidates": []}

    monkeypatch.setattr(LLMClient, "_post", staticmethod(fake_post))
    client = LLMClient(server_keys={}, ollama_url="")
    result = asyncio.run(client.answer("hi", "m", [], "gemini", "AIza"))
    assert result.mode == "offline"
    assert "could not read" in result.warning


def test_gemini_bad_key_error_is_explained_as_a_rejected_key():
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(400, request=request, text='{"error": {"message": "API key not valid"}}')
    error = httpx.HTTPStatusError("bad", request=request, response=response)
    assert "rejected the API key" in explain_failure(error)


def test_a_slow_big_model_hands_over_to_the_next_smaller_one(monkeypatch):
    calls = []

    async def fake_live(self, provider, prompt, model_id, key):
        calls.append(model_id)
        if model_id == "big":
            raise httpx.ReadTimeout("too slow")
        from app.llm import LLMResult
        return LLMResult(text="ok", input_tokens=1, output_tokens=1, mode="live")

    monkeypatch.setattr(LLMClient, "_live_answer", fake_live)
    client = LLMClient(server_keys={}, ollama_url="")
    result = asyncio.run(client.answer("q", "big", [], "gemini", "AIza", fallback_model_id="small"))
    assert calls == ["big", "small"]
    assert result.mode == "live" and "small answered instead" in result.warning


def test_failure_message_does_not_claim_the_key_is_missing(monkeypatch):
    async def fake_live(self, provider, prompt, model_id, key):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(LLMClient, "_live_answer", fake_live)
    client = LLMClient(server_keys={}, ollama_url="")
    result = asyncio.run(client.answer("q", "m", [], "gemini", "AIza"))
    assert "No language model key" not in result.text
    assert "took too long" in result.warning
