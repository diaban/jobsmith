"""Shared test fixtures: fake clients + in-memory LangGraph persistence."""
from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
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

    **It streams, and that is not a detail.** A model implementing `_generate`
    only makes LangChain fall back to an `astream` that yields the whole
    answer as ONE chunk — so every streaming test would pass while proving
    nothing about streaming. `_astream` cuts each scripted response into
    `chunk_size` pieces, which is what makes "the answer arrives in several
    Tokens" a claim a test can falsify. LangGraph reaches this path on its
    own: `_should_stream` is true as soon as a streaming callback handler is
    attached, which is exactly what `stream_mode="messages"` installs, so
    `ainvoke` still goes through `_generate` and the non-streaming tests are
    untouched.
    """

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    chunk_size: int = 5

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        text = msg.text if isinstance(msg.text, str) else ""
        pieces = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        for piece in pieces or [""]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        # Tool calls travel as `tool_call_chunks`, the only form the chunk
        # merger reassembles into a real `tool_calls` list — a chunk carrying
        # `tool_calls=` directly is dropped on aggregation and the agent then
        # never calls the tool.
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=[
                {"name": call["name"], "args": json.dumps(call["args"]),
                 "id": call["id"], "index": index, "type": "tool_call_chunk"}
                for index, call in enumerate(msg.tool_calls)
            ]))


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


@pytest.fixture(scope="session", autouse=True)
def sandboxed_data_dir(tmp_path_factory):
    """No test may write into the user's real jobsmith data directory (#63).

    An unconfigured `build_app` now keeps its jobs in a SQLite file and its
    reports under the per-user data dir, so a test that forgot to say
    `db="memory"` / `reports_dir=...` would quietly add jobs to the list of
    whoever ran the suite. Two things make that impossible AND visible:

    - the data dir is relocated for the whole session (`XDG_DATA_HOME`, which
      `data_dir()` honours on every platform), and the variables that would
      redirect a default somewhere else entirely are cleared — a developer's
      own `$JOBSMITH_DB` must not become where a careless test writes;
    - at the end of the session the relocated directory must still not
      exist. A site relying on the default fails the run here, by name,
      instead of passing while writing to a disk nobody is watching.

    A test that exercises the default on purpose points `XDG_DATA_HOME` at
    its own `tmp_path` (monkeypatch), so it never touches this one.
    """
    sandbox = tmp_path_factory.mktemp("xdg-data")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_DATA_HOME", str(sandbox))
        for name in ("JOBSMITH_DB", "JOBSMITH_REPORTS_DIR"):
            mp.delenv(name, raising=False)
        yield sandbox
    written = sorted(str(p.relative_to(sandbox)) for p in sandbox.rglob("*"))
    assert not written, (
        "the suite wrote into the (sandboxed) user data dir — some build_app "
        f"relies on the persistent default instead of saying db='memory': {written}")


@pytest.fixture
def checkpointer():
    return MemorySaver()


@pytest.fixture
def store():
    return InMemoryStore()


def registered_capabilities(app: Any, *, gated: bool = False) -> list[str]:
    """What the composed app will actually chain, in registry order.

    The default pack's registry is configuration-dependent — `documents` needs
    a directory, `web_search` a key, `slide_deck` the `.[pptx]` extra — so a
    test that pins what `KeywordLLM` plans (it chains every registered
    capability) asks the app what it composed instead of hardcoding the
    environment it happened to be written in.

    Registered is not the same as planned, though: a capability gated on
    `requires_inputs` (`read_files` needs a file the request named) IS in the
    registry and IS in the fake's chain, and then the planner drops it because
    the request carries no such input. What a plain query runs is therefore the
    ungated set, which is what this returns; `gated=True` gives the whole
    registry.
    """
    prefix = "cap_"
    names = [n[len(prefix):] for n in app.manager.graph.nodes if n.startswith(prefix)]
    registry = getattr(app, "registry", None) or getattr(
        getattr(app, "app", None), "registry", None)
    if gated or registry is None:
        return names
    return [n for n in names if not registry.get(n).spec.requires_inputs]
