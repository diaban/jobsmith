"""Shared test fixtures: fake clients + in-memory LangGraph persistence."""
from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field


class FakeLLM:
    """Scripted LLM: responses keyed by a substring of the system prompt.

    `script` maps a substring (matched against the system message) to either
    a string response or a list of responses consumed in order.
    Falls back to `default` when nothing matches.
    """

    def __init__(self, script: dict[str, Any] | None = None, *, default: str = "fake answer"):
        self.script = dict(script or {})
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def _system_of(self, messages: list[dict[str, Any]]) -> str:
        for m in messages:
            if m.get("role") == "system":
                return m.get("content", "")
        return ""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "response_format": response_format})
        system = self._system_of(messages)
        for key, resp in self.script.items():
            if key in system:
                if isinstance(resp, list):
                    return resp.pop(0) if len(resp) > 1 else resp[0]
                return resp
        return self.default

    async def vision(self, image_bytes: bytes, prompt: str, *, mime_type: str = "image/png") -> str:
        self.calls.append({"vision_prompt": prompt})
        return "a chart showing quarterly revenue"


class ScriptedChatModel(BaseChatModel):
    """Scripted LangChain chat model for the chat-agent tests.

    Pops `responses` in order (last one repeats); records every model input in
    `calls` so tests can assert on injected messages. `bind_tools` is a no-op —
    scripted responses carry their own `tool_calls`.
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        msg = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=msg)])


class FakeSearch:
    """Search engine returning canned docs; can be scripted to fail."""

    def __init__(self, docs: list[dict[str, Any]] | None = None, *, fail: bool = False):
        self.docs = docs if docs is not None else [{"id": "doc_1", "text": "canned document"}]
        self.fail = fail
        self.calls: list[str] = []

    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("search down")
        return self.docs

    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        self.calls.append(f"cached:{query}")
        if self.fail:
            raise RuntimeError("cache down")
        return self.docs


class FakeS3:
    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = objects or {}

    async def get_object(self, key: str) -> bytes:
        return self.objects[key]


def plan_json(*caps: str, deps: dict[str, list[str]] | None = None) -> str:
    """Build a planner JSON response for the given capability names."""
    deps = deps or {}
    return json.dumps({
        "steps": [{"capability": c, "depends_on": deps.get(c, [])} for c in caps],
        "rationale": "test plan",
    })


@pytest.fixture
def checkpointer():
    return MemorySaver()


@pytest.fixture
def store():
    return InMemoryStore()
