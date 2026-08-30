"""LLM client adapters: message mapping, JSON handling, refusals.

Uses injected stub clients — no network, no real SDK objects.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_oo.clients import AnthropicLLMClient, OpenAILLMClient


def make_response(text="hello", stop_reason="end_turn", category=None):
    blocks = [SimpleNamespace(type="text", text=text)]
    stop_details = SimpleNamespace(category=category) if category else None
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, stop_details=stop_details)


class StubAnthropicClient:
    """Mimics AsyncAnthropic's .messages / .beta.messages surface."""

    def __init__(self, response):
        self.calls: list[dict] = []
        self._response = response

        async def create(**kwargs):
            self.calls.append(kwargs)
            return self._response

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def make_client(response=None, **kwargs) -> tuple[AnthropicLLMClient, StubAnthropicClient]:
    stub = StubAnthropicClient(response or make_response())
    return AnthropicLLMClient(client=stub, **kwargs), stub


async def test_system_message_hoisted_and_temperature_dropped():
    client, stub = make_client()
    out = await client.chat(
        [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "plan this"},
        ],
        temperature=0.7,
    )
    assert out == "hello"
    call = stub.calls[0]
    assert call["system"] == "You are a planner."
    assert call["messages"] == [{"role": "user", "content": "plan this"}]
    assert call["model"] == "claude-opus-5"
    assert "temperature" not in call  # removed on Claude Opus 5 — must not be sent


async def test_fallbacks_enabled_by_default():
    client, stub = make_client()
    await client.chat([{"role": "user", "content": "hi"}])
    call = stub.calls[0]
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["extra_body"] == {"fallbacks": "default"}


async def test_fallbacks_can_be_disabled():
    client, stub = make_client(enable_fallbacks=False)
    await client.chat([{"role": "user", "content": "hi"}])
    assert "betas" not in stub.calls[0]


async def test_json_object_strips_markdown_fences():
    client, _ = make_client(make_response('```json\n{"query": "x"}\n```'))
    out = await client.chat(
        [{"role": "user", "content": "q"}],
        response_format={"type": "json_object"},
    )
    assert out == '{"query": "x"}'


async def test_plain_text_not_stripped():
    client, _ = make_client(make_response("```code sample``` explained"))
    out = await client.chat([{"role": "user", "content": "q"}])
    assert out == "```code sample``` explained"


async def test_refusal_raises():
    client, _ = make_client(make_response(stop_reason="refusal", category="cyber"))
    with pytest.raises(RuntimeError, match="category=cyber"):
        await client.chat([{"role": "user", "content": "q"}])


async def test_max_tokens_default_and_override():
    client, stub = make_client()
    await client.chat([{"role": "user", "content": "a"}])
    await client.chat([{"role": "user", "content": "b"}], max_tokens=512)
    assert stub.calls[0]["max_tokens"] == 16000
    assert stub.calls[1]["max_tokens"] == 512


async def test_vision_builds_image_block():
    client, stub = make_client(make_response("a chart"))
    out = await client.vision(b"\x89PNG...", "what is this?", mime_type="image/png")
    assert out == "a chart"
    content = stub.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "what is this?"}


# ==================== OpenAILLMClient ====================


def make_openai_response(text="hello", refusal=None):
    message = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class StubOpenAIClient:
    """Mimics AsyncOpenAI's .chat.completions surface."""

    def __init__(self, response):
        self.calls: list[dict] = []
        self._response = response

        async def create(**kwargs):
            self.calls.append(kwargs)
            return self._response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def make_openai(response=None, **kwargs) -> tuple[OpenAILLMClient, StubOpenAIClient]:
    stub = StubOpenAIClient(response or make_openai_response())
    return OpenAILLMClient(client=stub, **kwargs), stub


async def test_openai_reasoning_model_drops_temperature_uses_completion_cap():
    client, stub = make_openai()  # default gpt-5.1 -> reasoning family
    out = await client.chat(
        [
            {"role": "system", "content": "You are a planner."},
            {"role": "user", "content": "plan this"},
        ],
        temperature=0.7,
    )
    assert out == "hello"
    call = stub.calls[0]
    assert call["model"] == "gpt-5.1"
    assert "temperature" not in call
    assert call["max_completion_tokens"] == 16000
    assert "max_tokens" not in call
    # system message passes through unchanged (native role on this API)
    assert call["messages"][0] == {"role": "system", "content": "You are a planner."}


async def test_openai_classic_model_keeps_temperature_and_max_tokens():
    client, stub = make_openai(model="gpt-4o")
    await client.chat([{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=512)
    call = stub.calls[0]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 512
    assert "max_completion_tokens" not in call


async def test_openai_response_format_passes_through():
    client, stub = make_openai()
    await client.chat(
        [{"role": "user", "content": "q"}],
        response_format={"type": "json_object"},
    )
    assert stub.calls[0]["response_format"] == {"type": "json_object"}


async def test_openai_refusal_raises():
    client, _ = make_openai(make_openai_response(text=None, refusal="cannot help with that"))
    with pytest.raises(RuntimeError, match="cannot help with that"):
        await client.chat([{"role": "user", "content": "q"}])


async def test_openai_vision_builds_data_uri_block():
    client, stub = make_openai(make_openai_response("a chart"))
    out = await client.vision(b"\x89PNG...", "what is this?", mime_type="image/png")
    assert out == "a chart"
    content = stub.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "what is this?"}


async def test_openai_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-nano")
    client, stub = make_openai()
    assert client.model == "gpt-5-nano"
    # nano is in the gpt-5 reasoning family: no temperature, completion-token cap
    await client.chat([{"role": "user", "content": "hi"}], temperature=0.5)
    call = stub.calls[0]
    assert call["model"] == "gpt-5-nano"
    assert "temperature" not in call
    assert call["max_completion_tokens"] == 16000


async def test_explicit_model_beats_env(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-nano")
    client, _ = make_openai(model="gpt-4o")
    assert client.model == "gpt-4o"


async def test_anthropic_model_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    client, stub = make_client()
    await client.chat([{"role": "user", "content": "hi"}])
    assert stub.calls[0]["model"] == "claude-sonnet-5"
