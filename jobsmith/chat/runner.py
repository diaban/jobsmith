"""Driving one conversational turn, and translating it into domain events.

This is the chat twin of `jobs/runner.py`, and it exists for the same reason:
it is the **only** module that knows the shape of what LangGraph's `astream`
emits for a chat agent — the `("messages", (chunk, metadata))` pairs, the
`("updates", {node: update})` ones, the `model`/`tools` node names, the
`__interrupt__` key. Everything above it reacts to the small typed events
below, so the service, the HTTP adapter and every UI are written against a
vocabulary that does not move when LangChain reshapes its agent.

Five events, and no more, because a turn only ever shows five things:

    Token          the answer being written
    ToolStarted    the model asked for a tool, by its real name
    ToolFinished   that tool answered
    Message        the turn is over, here is the reply
    Proposal       the turn is over, it is waiting for an approval

The two terminals are not an invention of the stream: they are the duality
`LocalAgentService._reply` already carried as a return value. Streaming moves
it to the end of a flow, which is why `send()` can be *defined* as draining
this stream — one implementation of the turn, not two.

`ToolStarted` carries the tool's real name (`launch_job`), never a phrase for
a human. Turning it into readable prose is the presentation layer's business,
for the same reason `REPORT_MEDIA_TYPES` lives in `api/app.py` and not in
`jobs/report.py`: the CLI and a future TUI word it differently, and neither
wording belongs in the thing that reports the fact.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

# `create_agent`'s node names. The updates stream is keyed by them, so this is
# where that coupling is admitted rather than spread over the service.
_MODEL_NODE = "model"
_TOOLS_NODE = "tools"
_INTERRUPT = "__interrupt__"


@dataclass(frozen=True)
class Token:
    """A piece of the answer, as the model writes it."""
    text: str


@dataclass(frozen=True)
class ToolStarted:
    """The model asked for a tool — its real name, not a phrase."""
    name: str


@dataclass(frozen=True)
class ToolFinished:
    """That tool answered."""
    name: str


@dataclass(frozen=True)
class Message:
    """Terminal: the turn ended with a reply."""
    content: str


@dataclass(frozen=True)
class Proposal:
    """Terminal: the turn ended on a job awaiting human approval."""
    query: str | None
    rationale: str | None


ChatEvent = Token | ToolStarted | ToolFinished | Message | Proposal


def _text_of(message: Any) -> str:
    """The message's text, unstripped.

    Unstripped on purpose: a token is a fragment, and trimming it would glue
    words together. `.text` flattens provider content blocks, so a reply that
    arrives as blocks reads the same as one that arrives as a string.
    """
    text = getattr(message, "text", "")
    return text if isinstance(text, str) else ""


class ChatRunner:
    """Runs one turn of a conversation and yields what happened, in domain terms.

    Two ways in — `stream()` sends a message, `resume()` answers a pending
    approval — and both are translated by the same code, because a
    post-approval reply is a turn like any other.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        # Same convention as the job engine: the session id IS the thread id,
        # which is what makes a conversation rebuildable from the checkpointer.
        return {"configurable": {"thread_id": session_id}}

    def stream(self, session_id: str, text: str) -> AsyncIterator[ChatEvent]:
        from langchain_core.messages import HumanMessage

        return self._translate(self.agent.astream(
            {"messages": [HumanMessage(text)]},
            config=self._config(session_id),
            stream_mode=["messages", "updates"],
        ))

    def resume(self, session_id: str, approved: bool) -> AsyncIterator[ChatEvent]:
        """Answer the approval the last turn stopped on, and stream what follows."""
        from langgraph.types import Command

        return self._translate(self.agent.astream(
            Command(resume={"approved": approved}),
            config=self._config(session_id),
            stream_mode=["messages", "updates"],
        ))

    async def _translate(self, stream: AsyncIterator[Any]) -> AsyncIterator[ChatEvent]:
        """LangGraph's two stream modes → the events above, terminal last.

        Both modes are needed and neither is redundant: `messages` carries the
        answer as it is written, `updates` carries the facts a token cannot
        show — which tool was called, and that the run stopped on an interrupt.

        Exactly one terminal is always yielded, last. A caller draining this
        for a reply (`AgentService.send`) must never have to decide what an
        exhausted stream with no terminal meant.
        """
        answer, proposal = "", None
        async for mode, chunk in stream:
            if mode == "messages":
                message, _metadata = chunk
                # AI messages only: the tools node publishes its ToolMessages
                # here too, and a tool's output is not the answer being written.
                if isinstance(message, AIMessage) and (text := _text_of(message)):
                    yield Token(text)
            elif mode == "updates":
                for node, value in chunk.items():
                    if node == _INTERRUPT:
                        proposal = value[0].value
                    elif not isinstance(value, dict):
                        continue
                    elif node == _MODEL_NODE:
                        for message in value.get("messages") or []:
                            # The whole message, not the accumulated tokens:
                            # this is the same value `messages[-1].content`
                            # had before the turn became a flow.
                            answer = _text_of(message)
                            for call in getattr(message, "tool_calls", None) or []:
                                yield ToolStarted(call["name"])
                    elif node == _TOOLS_NODE:
                        for message in value.get("messages") or []:
                            yield ToolFinished(getattr(message, "name", "") or "")
        if proposal is not None:
            yield Proposal(proposal.get("query"), proposal.get("rationale"))
        else:
            yield Message(answer)
