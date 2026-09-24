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
from jobsmith.cli.repl import TurnPrinter, job_lines, render_turn, run_repl, tool_activity
from jobsmith.service import ChatStreamError, LocalAgentService, ServiceUnavailable

ANSWER = "A reasonably long answer that no single chunk should carry."


def make_agent(store, checkpointer, tmp_path, responses, *, approval=False,
               **model_kwargs):
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=responses, **model_kwargs)
    session = ChatSession(manager, model, checkpointer=MemorySaver(),
                          approval_required=approval)
    return session.build(), manager, model


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
    must never have to interpret an exhausted stream.

    On the gate path since #83, which is the point of keeping the gate: this
    is the only shape of turn that ends on something other than a `Message`,
    and it is a mechanism a future irreversible capability will need.
    """
    agent, _, _ = make_agent(store, checkpointer, tmp_path, [
        launch_call("analyse it", "multi-step"),
        AIMessage(content=ANSWER),
    ], approval=True)
    runner = ChatRunner(agent)

    events = await collect(runner.stream("s1", "please analyse it"))
    assert ToolStarted("launch_job") in events
    assert events[-1] == Proposal("analyse it", "multi-step")
    assert len([e for e in events if isinstance(e, Message | Proposal)]) == 1

    after = await collect(runner.resume("s1", True))
    assert [e.name for e in after if isinstance(e, ToolFinished)] == ["launch_job"]
    # The terminal is the whole of what the turn produced — the approved run's
    # answer and then the model's sentence — not the model's message alone.
    # It was `Message(ANSWER)` while the two were the same string; #83 broke
    # that tie, and a terminal that stayed the model's message would hand a
    # caller that waits an answer with the run's result cut out of it.
    streamed = "".join(e.text for e in after if isinstance(e, Token))
    assert after[-1] == Message(streamed)
    assert streamed.endswith(ANSWER) and len(streamed) > len(ANSWER)
    # a tool's *result* is not the answer being written: what the model was
    # told about the run stays out of the stream, and only what the run
    # produced is written into it.
    assert "ALREADY been shown" not in streamed


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


# ------------------------------------------------------- what the REPL shows


def test_a_tool_is_named_in_prose_here_and_nowhere_else():
    """The event carries `launch_job`; a human is not asked to read that.

    The mapping lives with the presentation, so an unmapped tool still says
    something rather than leaking a bare identifier into a sentence.
    """
    assert tool_activity("launch_job") == "running the task"
    assert tool_activity("something_new") == "running something_new"


async def test_the_answer_goes_to_stdout_and_the_activity_to_stderr(capsys):
    """stdout is the conversation — piping `jobsmith chat` gives the answers
    and nothing else — and "what it is doing right now" goes where every
    other diagnostic in this layer goes."""
    async def events():
        for event in ({"type": "tool_started", "name": "launch_job"},
                      {"type": "tool_finished", "name": "launch_job"},
                      {"type": "token", "text": "all "},
                      {"type": "token", "text": "done"},
                      {"type": "message", "content": "all done"}):
            yield event

    terminal = await render_turn(events(), TurnPrinter())
    out, err = capsys.readouterr()
    assert terminal == {"type": "message", "content": "all done"}
    assert out == "  all done\n"          # printed once, as it arrived
    assert "… running the task" in err
    assert "✓ running the task" in err


async def test_the_repl_shows_what_the_run_will_do_and_how_to_stop_it(capsys):
    """The notice that replaced the card (#83), rendered.

    Three guarantees used to hang on an approval and now hang on this block:
    the reformulated query, the files it may open, the document it will
    write. The fourth line is the one the card could never carry — the job
    id, because cancellation is the undo the gate used to be, and an undo
    nobody is told about is not one.
    """
    async def events():
        for event in (
            {"type": "job_started", "job_id": "abcdef0123456789",
             "query": "compare the two chairs", "rationale": "several steps",
             "sources": ["/notes/chairs.md"], "document_name": "chairs",
             "document_title": "Comparatif", "formats": ["markdown", "html"]},
            {"type": "token", "text": "the answer"},
            {"type": "message", "content": "saved"},
        ):
            yield event

    await render_turn(events(), TurnPrinter())
    out, _ = capsys.readouterr()

    assert "running this as job abcdef01" in out
    assert "task     : compare the two chairs" in out
    assert "reads    : /notes/chairs.md" in out
    assert "titled   : Comparatif" in out
    assert "writes   : chairs.md, chairs.html" in out
    assert "stop it  : /cancel abcdef01" in out
    assert "the answer" in out


def test_the_repl_names_the_jobs_a_run_builds_on():
    """#104: an earlier job a run is handed is shown like a file it may open —
    by the short id `/job` and `/cancel` take and the start of its query, one
    line each, on the notice and on the proposal alike (one renderer)."""
    refs = [{"job_id": "0123456789abcdef", "query": "compare the two chairs"},
            {"job_id": "fedcba9876543210", "query": "price the standing desk"}]
    notice = {"type": "job_started", "job_id": "abcdef0123456789",
              "query": "a one-pager out of both", "rationale": "several steps",
              "sources": [], "formats": None, "from_jobs": refs}
    proposal = {k: v for k, v in notice.items() if k != "job_id"} | {"type": "proposal"}

    lines = job_lines(notice)
    assert lines[2:4] == ["    builds on: job 01234567 — compare the two chairs",
                          "             : job fedcba98 — price the standing desk"]
    assert job_lines(proposal) == lines

    # nothing referenced, nothing said — an empty line would read as a claim
    assert not any("builds on" in line
                   for line in job_lines(notice | {"from_jobs": []}))


async def test_the_repl_says_the_plan_as_activity_not_as_the_answer(
    store, checkpointer, tmp_path, capsys
):
    """#86, through the real service: the plan is said on stderr, beside the
    activity it sharpens, as the waves in order — two steps that run together
    side by side. stdout, the record of the turn (the notice, the answer),
    does not carry it."""
    from test_service import planned_manager, planned_service

    service = planned_service(planned_manager(store, checkpointer, tmp_path))
    session_id = await service.new_session()
    await render_turn(service.stream(session_id, "please analyse it"), TurnPrinter())
    out, err = capsys.readouterr()

    assert "  … plan: web_search + documents → research → analysis\n" in err
    assert "plan:" not in out and "→" not in out
    assert "running this as job" in out, "no job ran; this proves nothing"
    # said after the task started and before it finished
    assert err.index("… running the task") < err.index("… plan:") \
        < err.index("✓ running the task")


def test_a_plan_reads_as_its_waves():
    """One line, the drawings' columns: a step is placed after the longest
    chain of what it waits on, whatever order the plan lists it in."""
    from jobsmith.cli.repl import plan_line

    assert plan_line([
        {"capability": "critique", "depends_on": ["analysis"]},
        {"capability": "analysis", "depends_on": ["research"]},
        {"capability": "research", "depends_on": []},
        {"capability": "slide_deck", "depends_on": ["analysis"]},
    ]) == "research → analysis → critique + slide_deck"


async def test_the_repl_streams_a_turn_and_still_asks_for_approval(capsys, monkeypatch):
    """The human-in-the-loop round trip survives the turn becoming a flow:
    the proposal is still printed and answered, and the reply that follows is
    streamed rather than waited for."""
    typed = iter(["do the big thing", "y", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(typed))

    class Client:
        def __init__(self):
            self.approved = None

        async def stream(self, session_id, text):
            yield {"type": "tool_started", "name": "launch_job"}
            yield {"type": "proposal", "query": "the big thing", "rationale": "multi-step"}

        async def stream_approval(self, session_id, approved):
            self.approved = approved
            yield {"type": "token", "text": "launched"}
            yield {"type": "message", "content": "launched"}

    client = Client()
    await run_repl(client, "s1")           # type: ignore[arg-type]
    out, _ = capsys.readouterr()
    assert client.approved is True
    assert "task     : the big thing" in out
    assert "  launched\n" in out


async def test_the_repl_survives_a_backing_that_went_away(capsys, monkeypatch):
    """The REPL's half of the same narrowing the TUI uses.

    One `except`, and it names the port's own exception — a broad one would
    swallow this project's bugs to catch a daemon's absence. So a command
    against a daemon that died says so on stderr (where every diagnostic in
    this layer goes) and the prompt comes back; the `KeyError` below is not
    caught, and a defect in this code still ends the loop with a traceback.
    """
    typed = iter(["/jobs", "hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(typed))

    class Client:
        async def list_jobs(self, **kwargs):
            raise ServiceUnavailable.reaching("http://127.0.0.1:8000", ConnectionRefusedError())

        async def stream(self, session_id, text):
            raise ServiceUnavailable.reaching("http://127.0.0.1:8000", ConnectionRefusedError())
            yield {}                       # pragma: no cover - an async generator

    await run_repl(Client(), "s1")         # type: ignore[arg-type]
    out, err = capsys.readouterr()
    assert err.count("cannot reach the agent at http://127.0.0.1:8000") == 2
    assert out.rstrip().endswith("bye")    # ...and the loop kept going


async def test_a_defect_in_this_process_still_ends_the_repl_loudly(monkeypatch):
    """The rule the narrowing has to keep: only the named exception is
    answered. A `KeyError` from our own code is not a daemon that went away,
    and turning it into one would hide the bug behind a reconnect message."""
    typed = iter(["/jobs", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(typed))

    class Client:
        async def list_jobs(self, **kwargs):
            raise KeyError("a defect in this process")

    with pytest.raises(KeyError):
        await run_repl(Client(), "s1")     # type: ignore[arg-type]


async def test_a_cut_short_turn_is_announced_and_the_repl_survives_it(capsys, monkeypatch):
    """Half a printed answer plus silence is the defect the rule exists for.

    The REPL says the turn was cut short — loudly, on stderr — and stays
    usable, because the conversation itself is in the checkpointer.
    """
    typed = iter(["hello", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(typed))

    class Client:
        async def stream(self, session_id, text):
            yield {"type": "token", "text": "half a sen"}
            raise ChatStreamError("unreadable event in the reply")

        async def stream_approval(self, session_id, approved):
            yield {"type": "message", "content": ""}

    await run_repl(Client(), "s1")         # type: ignore[arg-type]
    out, err = capsys.readouterr()
    assert "half a sen" in out
    assert "cut short" in err
    assert out.rstrip().endswith("bye")    # the loop kept going


async def test_the_terminal_falls_back_only_when_nothing_was_streamed():
    """The one branch the transcript rule needs, and its bound.

    A model with no `_astream` of its own still reaches the `messages` mode
    through LangChain's one-chunk fallback, so this is a corner — but a model
    that emitted no chunk at all must not turn a written answer into an empty
    terminal. It can never mask a missing job answer, because that arrives as
    a Token: any turn a job answered in has a non-empty transcript and never
    reaches here.
    """
    from langchain_core.messages import AIMessage as AI

    from jobsmith.chat.runner import CUSTOM_ANSWER

    async def silent_model():
        yield "updates", {"model": {"messages": [AI(content="the whole answer")]}}

    async def a_job_answered():
        # a tool wrote into the turn; the model then streamed nothing
        yield "custom", {"event": CUSTOM_ANSWER, "text": "what the run produced"}
        yield "updates", {"model": {"messages": [AI(content="one sentence")]}}

    runner = ChatRunner(agent=None)
    assert [e async for e in runner._translate(silent_model())][-1] == \
        Message("the whole answer")
    # the job's answer is never traded for the model's message
    assert [e async for e in runner._translate(a_job_answered())][-1] == \
        Message("what the run produced")
