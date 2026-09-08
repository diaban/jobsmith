"""A turn is a flow: the events it emits, and what the REPL makes of them.

Two things had to be true before any of this could be written, and both are
pinned here rather than assumed:

- the job-notification middleware still injects into a **streamed** model
  call — it is how a finished job reaches the conversation, so a streaming
  path it silently skipped would be a regression disguised as a feature;
- `ScriptedChatModel` really emits several chunks. A model implementing
  `_generate` alone makes LangChain fall back to a one-chunk `astream`, and
  every test below would then pass while proving nothing.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_jobs import make_manager

from jobsmith.chat import (
    ChatRunner,
    ChatSession,
    Message,
    Proposal,
    Token,
    ToolFinished,
    ToolStarted,
)
from jobsmith.chat.session import NOTICE_MARKER
from jobsmith.cli.client import DaemonClient
from jobsmith.service import ChatStreamError, LocalAgentService

ANSWER = "A reasonably long answer that no single chunk should carry."


def make_agent(store, checkpointer, tmp_path, responses, **model_kwargs):
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=responses, **model_kwargs)
    return ChatSession(manager, model, checkpointer=MemorySaver()).build(), manager, model


async def collect(events) -> list[Any]:
    return [event async for event in events]


# ----------------------------------------------------- the two cleared risks


async def test_the_scripted_model_emits_the_answer_in_several_pieces(
    store, checkpointer, tmp_path
):
    """The fake must really stream, or nothing below measures anything.

    LangChain only reaches `_astream` when a streaming callback handler is
    attached, which is exactly what `stream_mode="messages"` installs — so
    this also pins that the runner asks for the mode that makes it happen.
    """
    agent, _, _ = make_agent(store, checkpointer, tmp_path, [AIMessage(content=ANSWER)])
    events = await collect(ChatRunner(agent).stream("s1", "hello"))

    tokens = [e for e in events if isinstance(e, Token)]
    assert len(tokens) > 1, "the answer arrived in one piece: the fake is not streaming"
    assert "".join(t.text for t in tokens) == ANSWER
    assert events[-1] == Message(ANSWER)


async def test_a_finished_job_is_still_announced_on_a_streamed_turn(
    store, checkpointer, tmp_path
):
    """`awrap_model_call` wraps a call that now streams its answer.

    The middleware is how a finished job reaches the conversation. A
    streaming path that bypassed it — or consumed the response before it
    could inject — would lose that quietly: the turn still answers, it just
    never mentions the job. So the notice is checked where it lands, in what
    the model was actually given.
    """
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=[AIMessage(content=ANSWER)])
    saver = MemorySaver()
    service = LocalAgentService(manager, lambda session_id=None: ChatSession(
        manager, model, session_id=session_id, checkpointer=saver))
    session_id = await service.new_session()
    job = await manager.create_job("older work", None, session_id=session_id)
    await manager.run_job(job.job_id)

    tokens = [e async for e in service.stream(session_id, "hi") if e["type"] == "token"]
    assert tokens, "the streamed turn produced no answer at all"
    notices = [m for call in model.calls for m in call
               if isinstance(m, SystemMessage) and NOTICE_MARKER in m.text]
    assert notices, "the finished job never reached the model on a streamed turn"
    assert (await manager.get_job(job.job_id)).announced is True


# ------------------------------------------------------ the event vocabulary


async def test_a_turn_that_proposes_a_job_ends_on_a_proposal(store, checkpointer, tmp_path):
    """The two terminals are the duality `_reply` always carried, and the
    stream always ends on exactly one of them — a caller draining for a reply
    must never have to interpret an exhausted stream."""
    agent, _, _ = make_agent(store, checkpointer, tmp_path, [
        launch_call("analyse it", "multi-step"),
        AIMessage(content=ANSWER),
    ])
    runner = ChatRunner(agent)

    events = await collect(runner.stream("s1", "please analyse it"))
    assert ToolStarted("launch_job") in events
    assert events[-1] == Proposal("analyse it", "multi-step")
    assert len([e for e in events if isinstance(e, Message | Proposal)]) == 1

    after = await collect(runner.resume("s1", True))
    assert [e.name for e in after if isinstance(e, ToolFinished)] == ["launch_job"]
    assert after[-1] == Message(ANSWER)
    # a tool's output is not the answer being written
    assert "launched in the background" not in "".join(
        e.text for e in after if isinstance(e, Token))


async def test_send_is_the_stream_drained(store, checkpointer, tmp_path):
    """One implementation of a turn. `send` must be the terminal of the very
    stream it drains, not a second path that could answer differently."""
    manager = make_manager(store, checkpointer, tmp_path)
    saver = MemorySaver()

    def session_factory(session_id=None):
        return ChatSession(manager, ScriptedChatModel(responses=[AIMessage(content=ANSWER)]),
                           session_id=session_id, checkpointer=saver)

    service = LocalAgentService(manager, session_factory)
    session_id = await service.new_session()
    streamed = [e async for e in service.stream(session_id, "hello")]
    assert streamed[-1] == {"type": "message", "content": ANSWER}
    assert await service.send(session_id, "hello again") == {"type": "message",
                                                             "content": ANSWER}


async def test_a_turn_with_no_terminal_is_refused_rather_than_answered():
    """A truncated turn must not read as a complete one.

    This is the no-drop rule at the port's edge: the last thing seen is not
    a reply, and returning it would hand the caller a sentence the model
    never finished.
    """
    from jobsmith.service import terminal_of

    async def cut_short():
        yield {"type": "token", "text": "half a sen"}

    with pytest.raises(ChatStreamError, match="cut short"):
        await terminal_of(cut_short())


# -------------------------------------------------- the transport never drops


def _chat_client(*lines: str) -> DaemonClient:
    """A DaemonClient whose streaming turn route answers with these SSE lines."""
    body = "".join(f"{line}\n" for line in lines)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    return DaemonClient("http://test", httpx.AsyncClient(
        transport=transport, base_url="http://test", timeout=None))


async def test_a_line_the_reader_cannot_decode_is_a_failure_not_a_skip():
    """The exact opposite of `/events`, and the asymmetry is the point.

    There, an unreadable line is skipped and the stream carries on: a
    progress tick is replaced by the next one a second later. Here the line
    was part of a sentence, so skipping it would deliver a shorter answer
    that reads as finished. Nobody can tell — which is why it raises.
    """
    client = _chat_client('data: {"type": "token", "text": "hi"}', "",
                          "data: {not json at all}", "")
    with pytest.raises(ChatStreamError, match="unreadable event"):
        await collect(client.stream("s1", "hello"))
    await client.aclose()


async def test_a_slow_reader_loses_nothing():
    """`subscribe` drops for a slow consumer; a turn must not.

    A queue with a drop policy would pass every other test here and fail
    this one: the reader takes its time and still gets every event, in
    order, because there is no buffer between the socket and it.
    """
    lines = [line for i in range(20)
             for line in (json.dumps({"type": "token", "text": str(i)}), "")]
    client = _chat_client(*[f"data: {line}" if line else "" for line in lines])
    try:
        seen = []
        async for event in client.stream("s1", "hello"):
            await asyncio.sleep(0.001)          # a reader that renders slowly
            seen.append(event)
        assert [e["text"] for e in seen] == [str(i) for i in range(20)]
    finally:
        await client.aclose()
