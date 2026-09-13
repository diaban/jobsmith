"""Chat layer: a task in the turn, promotion to the background, job notices.

The gate is not gone, it is off the nominal path (#83): the tests that drive
`interrupt()` build their session with `approval=True`, which is what
`$JOBSMITH_APPROVE_JOBS` sets in a deployment. They are kept because the
mechanism is kept — a future capability that spends money will want it, and
rebuilding an approval round trip through a port, an HTTP route and three
front-ends is the expensive half.
"""
from __future__ import annotations

import asyncio

import pytest
from conftest import FakeLLM, ScriptedChatModel, plan_json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from test_jobs import make_manager

from jobsmith.chat import ChatRunner, ChatSession, JobStarted, Token, ToolFinished
from jobsmith.chat.session import (
    NOTICE_MARKER,
    PROGRESS_MARKER,
    JobNotificationMiddleware,
)
from jobsmith.chat.tools import (
    DEFAULT_SYNC_TIMEOUT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TURNS,
    MAX_TURN_CHARS,
    pick_approval_required,
    pick_sync_timeout,
    progress_line,
    progress_signature,
    recent_conversation,
    running_steps,
)
from jobsmith.core.state import CONVERSATION_INPUT_KEY
from jobsmith.jobs.models import Job, JobOutput, JobStatus, now_iso


def launch_call(query: str, rationale: str, **args) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "launch_job",
            "args": {"query": query, "rationale": rationale, **args},
            "id": "call_1",
        }],
    )


def make_session(
    store, checkpointer, tmp_path, responses, *, llm=None,
    approval=False, sync_timeout=None,
) -> tuple[ChatSession, ScriptedChatModel]:
    manager = make_manager(store, checkpointer, tmp_path, llm=llm)
    model = ScriptedChatModel(responses=responses)
    session = ChatSession(manager, model, checkpointer=MemorySaver(),
                          approval_required=approval, sync_timeout=sync_timeout)
    return session, model


CFG = {"configurable": {"thread_id": "chat-1"}}


# ---------------- The nominal path: the task runs inside the turn ------------


async def test_a_task_runs_in_the_turn_and_nothing_is_asked_first(
    store, checkpointer, tmp_path
):
    """#83's whole claim, in one test: no card, a real run, an answer now.

    The engine used to be reachable only by handing the user a y/N and
    telling them to come back later. It is reached by asking, and the job
    exists — with its plan, its results and its deliverable — by the time the
    turn ends.
    """
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "needs several capability steps"),
        AIMessage(content="Done — the report is on disk."),
    ])
    agent = session.build()

    out = await agent.ainvoke(
        {"messages": [HumanMessage("please analyse the alpha data")]}, CFG)

    assert "__interrupt__" not in out, "the nominal path still asked for approval"
    (summary,) = await session.manager.list_jobs(session_id=session.session_id)
    job = await session.manager.get_job(summary.job_id)   # summaries carry no results
    assert job.status is JobStatus.DONE
    assert job.report_path is not None
    assert job.results, "the job engine did not actually run"


async def test_the_answer_is_delivered_verbatim_and_not_through_the_model(
    store, checkpointer, tmp_path
):
    """The decision the tool result forces (#83).

    A tool result is read by the model, which then writes the reply from it —
    so an answer handed back that way is an answer the model rewrites. It is
    written straight into the turn instead, and the model is told it has been
    delivered. Both halves are asserted: the reader gets the words, the model
    never sees them.
    """
    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "multi-step"),
        AIMessage(content="Saved."),
    ])
    runner = ChatRunner(session.build())

    events = [e async for e in runner.stream(session.session_id, "analyse the alpha data")]
    (job,) = await session.manager.list_jobs(session_id=session.session_id)

    delivered = "".join(e.text for e in events if isinstance(e, Token))
    assert job.final_answer and job.final_answer in delivered, \
        "the job's answer never reached the reader"
    tool_result = next(m for m in model.calls[-1] if isinstance(m, ToolMessage))
    assert job.final_answer not in tool_result.content, \
        "the answer went back through the model, which is where it gets rewritten"
    assert "ALREADY been shown" in tool_result.content
    assert job.report_path in tool_result.content   # ...and where the file is


async def test_the_notice_carries_the_three_guarantees_the_card_used_to(
    store, checkpointer, tmp_path
):
    """What approval was really for: the user *seeing* three decisions.

    The reformulated query (the engine never sees the thread), the files the
    run may open (#60), and what the document will be called, titled and
    written as (#55). Plus the one the card could not carry — the job id,
    because cancellation is the undo the gate used to be.
    """
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha cohort's Q3 churn", "multi-step",
                    source_files=["/notes/churn.md"], document_name="churn.md",
                    document_title="Churn Q3", formats=["markdown"]),
        AIMessage(content="Saved."),
    ])
    runner = ChatRunner(session.build())

    events = [e async for e in runner.stream(session.session_id, "analyse that")]

    (started,) = [e for e in events if isinstance(e, JobStarted)]
    assert started.query == "analyse the alpha cohort's Q3 churn"
    assert started.sources == ["/notes/churn.md"]
    assert started.document_name == "churn"          # the extension was dropped
    assert started.document_title == "Churn Q3"
    assert started.formats == ["markdown"]
    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert started.job_id == job.job_id, "the notice cannot name the job to cancel"
    # ...and it is said BEFORE the answer, which is the point of a notice
    assert events.index(started) < min(
        i for i, e in enumerate(events) if isinstance(e, Token))


async def test_a_slow_task_is_promoted_and_the_turn_says_so(store, checkpointer, tmp_path):
    """Promotion is "stop waiting", not "change how it runs".

    With the clock at zero the tool never waits, so the job is exactly what
    it always was — a background task — and the turn ends saying the answer
    is not coming in it.
    """
    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("a long one", "multi-step"),
        AIMessage(content="It is running in the background."),
    ], sync_timeout=0)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("do the long one")]}, CFG)

    tool_result = next(m for m in model.calls[-1] if isinstance(m, ToolMessage))
    assert "BACKGROUND" in tool_result.content
    assert "do not invent one" in tool_result.content
    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
    # ...and it was NOT cancelled by the turn ending: it runs to the end and
    # the completion notice is what brings it back.
    settled = await poll_until_settled(session.manager, job.job_id)
    assert settled.status is JobStatus.DONE
    assert settled.announced is False, "a promoted job must still be announced later"


async def test_a_turn_that_dies_mid_wait_does_not_take_the_job_with_it(
    store, checkpointer, tmp_path
):
    """The payoff of the implementation shape, and the reason for it.

    The task is `start_job` + wait, never `run_job`: a job is a background
    task from the first instant, so the only thing a cancelled turn cancels
    is the *waiting*. A UI killing its worker (`@work`), a client hanging up
    mid-stream, a REPL interrupted — none of them lose a run that is halfway
    through. Driving the run inside the turn instead would make every one of
    those a job silently thrown away.
    """
    from test_jobs import CountingEcho

    # Long enough to be cancelled inside, short enough to finish afterwards:
    # the claim is that it finishes, so it has to be allowed to.
    slow = CountingEcho("slow", delay=0.5)
    llm = FakeLLM({"planner": plan_json("slow")},
                  default="A sufficiently long final answer for the job test.")
    manager = make_manager(store, checkpointer, tmp_path, caps=[slow], llm=llm)
    model = ScriptedChatModel(responses=[launch_call("a long one", "multi-step"),
                                         AIMessage(content="Saved.")])
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    agent = session.build()

    turn = asyncio.create_task(
        agent.ainvoke({"messages": [HumanMessage("do the long one")]}, CFG))
    for _ in range(500):                       # wait until the step is really running
        await asyncio.sleep(0.01)
        if slow.runs:
            break
    assert slow.runs == 1, "the job never started, so nothing is being tested"

    turn.cancel()                              # the UI's worker, the client hanging up
    with pytest.raises(asyncio.CancelledError):
        await turn

    (job,) = await manager.list_jobs(session_id=session.session_id)
    settled = await poll_until_settled(manager, job.job_id)
    assert settled.status is JobStatus.DONE, "the turn's death killed the job"
    assert settled.announced is False, "nobody heard the answer: it must still be news"


async def test_a_task_answered_in_the_turn_is_never_announced_as_news(
    store, checkpointer, tmp_path
):
    """Where the two paths collide, and it bites inside a single turn.

    The completion notice exists to bring a finished job back into a later
    conversation, and it hands the model the whole answer with "give the user
    a short synthesis". A job that finished inside this turn is finished
    *before* the model writes its reply — so without being marked announced
    it is injected into that very call, telling the model to summarize the
    answer the tool result just told it not to touch. The paraphrase this
    change exists to prevent, arriving through the other door.
    """
    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "multi-step"),
        AIMessage(content="Saved."),
        AIMessage(content="Anything else?"),
    ])
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("analyse it")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert (await session.manager.get_job(job.job_id)).announced is True
    # the model call that wrote the reply saw no completion notice...
    assert not notices(model.calls[-1], NOTICE_MARKER)
    finished = await session.manager.get_job(job.job_id)
    assert finished.final_answer not in str(model.calls[-1]), \
        "the answer was handed to the model after all, as a notice"

    await agent.ainvoke({"messages": [HumanMessage("thanks")]}, CFG)
    assert not notices(model.calls[-1], NOTICE_MARKER)   # ...nor did the next one


async def test_a_task_that_fails_in_the_turn_says_so_and_offers_no_answer(
    store, checkpointer, tmp_path
):
    """The other half of delivering in the turn: there is nothing to deliver.

    `_delivered` mirrors `_notice_for`'s branches for the reasons #41 and #59
    gave — a stop is not a failure, a run with no answer must not be
    announced as one that has one — and it must never leave the model
    inventing a result to fill the silence.
    """
    llm = FakeLLM({"planner": "not json at all"})
    session, model = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "multi-step"),
        AIMessage(content="It failed, sorry."),
    ], llm=llm)
    runner = ChatRunner(session.build())

    events = [e async for e in runner.stream(session.session_id, "analyse it")]

    (summary,) = await session.manager.list_jobs(session_id=session.session_id)
    job = await session.manager.get_job(summary.job_id)
    assert job.status is JobStatus.FAILED
    # nothing was written into the turn before the tool answered: a run with
    # no answer must deliver silence, not an empty paragraph
    finished = next(i for i, e in enumerate(events) if isinstance(e, ToolFinished))
    assert not [e for e in events[:finished] if isinstance(e, Token)]
    tool_result = next(m for m in model.calls[-1] if isinstance(m, ToolMessage))
    assert "FAILED" in tool_result.content and job.error in tool_result.content
    assert "ALREADY been shown" not in tool_result.content
    assert job.announced is True, "a failure reported here must not be news again"


def test_pick_sync_timeout_follows_the_house_precedence(monkeypatch):
    monkeypatch.delenv("JOBSMITH_SYNC_TIMEOUT", raising=False)
    assert pick_sync_timeout() == DEFAULT_SYNC_TIMEOUT
    assert pick_sync_timeout(3) == 3.0
    monkeypatch.setenv("JOBSMITH_SYNC_TIMEOUT", "5")
    assert pick_sync_timeout() == 5.0
    assert pick_sync_timeout(1.5) == 1.5           # the argument still wins
    monkeypatch.setenv("JOBSMITH_SYNC_TIMEOUT", "0")
    assert pick_sync_timeout() == 0.0              # 0 is a value, not "unset"
    monkeypatch.setenv("JOBSMITH_SYNC_TIMEOUT", "soon")
    assert pick_sync_timeout() == DEFAULT_SYNC_TIMEOUT


def test_pick_approval_required_is_off_unless_a_deployment_says_so(monkeypatch):
    monkeypatch.delenv("JOBSMITH_APPROVE_JOBS", raising=False)
    assert pick_approval_required() is False
    assert pick_approval_required(True) is True
    monkeypatch.setenv("JOBSMITH_APPROVE_JOBS", "1")
    assert pick_approval_required() is True
    assert pick_approval_required(False) is False
    monkeypatch.setenv("JOBSMITH_APPROVE_JOBS", "no")
    assert pick_approval_required() is False


# ---------------- The gate, kept and off the nominal path --------------------


async def test_the_kept_gate_still_interrupts_and_runs_on_approval(
    store, checkpointer, tmp_path
):
    """`$JOBSMITH_APPROVE_JOBS` restores the whole round trip.

    Deliberately still here after #83: the mechanism is what a capability
    that spends money or does something irreversible will hang off, and it
    reaches from the tool through the port and the HTTP route into three
    front-ends. Deleting it would mean building all of that again.
    """
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "needs several capability steps"),
        AIMessage(content="Job launched — I'll share the report when it's done."),
    ], approval=True)
    agent = session.build()

    out = await agent.ainvoke({"messages": [HumanMessage("please analyse the alpha data")]}, CFG)
    # paused for human approval, nothing launched yet
    (intr,) = out["__interrupt__"]
    assert intr.value["action"] == "launch_job"
    assert intr.value["rationale"] == "needs several capability steps"
    assert await session.manager.list_jobs(session_id=session.session_id) == []

    out = await agent.ainvoke(Command(resume={"approved": True}), CFG)
    assert "share the report" in out["messages"][-1].content

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.query == "analyse the alpha data"
    for _ in range(200):  # the approved run still settles
        await asyncio.sleep(0.01)
        job = await session.manager.get_job(job.job_id)
        if job.status is JobStatus.DONE:
            break
    assert job.status is JobStatus.DONE
    assert job.report_path is not None


async def test_declined_launch_creates_no_job(store, checkpointer, tmp_path):
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("big task", "complex"),
        AIMessage(content="Understood, I won't launch it."),
    ], approval=True)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("do the big task")]}, CFG)
    out = await agent.ainvoke(Command(resume={"approved": False}), CFG)

    assert await session.manager.list_jobs(session_id=session.session_id) == []
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "DECLINED" in tool_msg.content


async def test_finished_job_injected_once_then_marked_announced(store, checkpointer, tmp_path):
    session, model = make_session(store, checkpointer, tmp_path, [
        AIMessage(content="Your analysis is ready — see the report."),
    ])
    # a session job finished before the user's next message
    job = await session.manager.create_job("crunch numbers", session_id=session.session_id)
    await session.manager.run_job(job.job_id)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("hi again")]}, CFG)
    injected = [
        m for m in model.calls[0]
        if isinstance(m, SystemMessage) and "background jobs finished" in m.content
    ]
    assert len(injected) == 1
    assert job.job_id[:8] in injected[0].content
    assert f"{job.job_id}.md" in injected[0].content  # report path for the link
    assert (await session.manager.get_job(job.job_id)).announced is True

    # next turn: nothing new to announce → no injection
    await agent.ainvoke({"messages": [HumanMessage("thanks")]}, CFG)
    assert not any(
        isinstance(m, SystemMessage) and "background jobs finished" in m.content
        for m in model.calls[-1]
    )


async def test_a_job_whose_report_failed_is_announced_honestly(
    store, checkpointer, tmp_path
):
    """DONE with no deliverable is a real state (the run answered, the write
    failed). Announcing "Report file: None" hands the user a null path and
    hides the one thing that explains it."""

    class Boom:
        format, extension = "markdown", "md"

        def write(self, job, directory):
            raise OSError("No space left on device")

    session, model = make_session(store, checkpointer, tmp_path,
                                  [AIMessage(content="Here is what came back.")])
    session.manager.reporter = Boom()
    job = await session.manager.create_job("crunch numbers", session_id=session.session_id)
    done = await session.manager.run_job(job.job_id)
    assert done.status is JobStatus.DONE and done.report_path is None

    agent = session.build()
    await agent.ainvoke({"messages": [HumanMessage("hi again")]}, CFG)

    (notice,) = notices(model.calls[0], NOTICE_MARKER)
    assert "None" not in notice.content                     # never a null path
    assert "No report file was saved" in notice.content
    assert "No space left on device" in notice.content      # why, in the notice
    assert done.final_answer in notice.content              # the answer survived


def test_the_notice_names_files_written_before_the_failure():
    """A partial write leaves real files behind even with the main one
    missing: `report_path` is None, but those paths are still worth having."""
    job = Job(job_id="abcdef0123", status=JobStatus.DONE, query="q",
              final_answer="The answer.",
              error="the html deliverable could not be written: OSError: nope",
              outputs=[JobOutput(path="/tmp/abcdef0123.md", role="alternate")])
    notice = JobNotificationMiddleware._notice_for(job)

    assert "None" not in notice
    assert "/tmp/abcdef0123.md" in notice
    assert "html deliverable could not be written" in notice
    assert "The answer." in notice


def test_a_failed_job_announces_the_files_its_steps_left_behind():
    """A job that stopped can still have produced files (#41). Announcing the
    failure and saying nothing about them recreates, in the conversation, the
    defect the DONE branch above was fixed for: a file nobody can find."""
    job = Job(job_id="abcdef0123", status=JobStatus.FAILED, query="q",
              error="the model refused",
              outputs=[JobOutput(path="/tmp/abcdef0123/chart.svg", format="svg",
                                 role="annex", produced_by="chart")])
    notice = JobNotificationMiddleware._notice_for(job)

    assert "FAILED: the model refused" in notice          # why, first
    assert "/tmp/abcdef0123/chart.svg" in notice
    assert "not as a report" in notice                    # what they are not
    assert "Report file" not in notice


def test_a_failed_job_with_no_files_is_announced_exactly_as_before():
    job = Job(job_id="abcdef0123", status=JobStatus.FAILED, query="q",
              error="the model refused")
    notice = JobNotificationMiddleware._notice_for(job)
    assert notice == "Job abcdef01 ('q') FAILED: the model refused"


def test_a_cancelled_job_is_announced_as_cancelled_not_failed():
    """`cancel_job` is one of the model's own tools, so the same actor stops a
    job and reports on it. Calling that stop a failure would put an untruth in
    the conversation; saying nothing about the files would hide them."""
    job = Job(job_id="abcdef0123", status=JobStatus.CANCELLED, query="q",
              error="1 file(s) a step reported producing are missing: chart → gone.svg",
              outputs=[JobOutput(path="/tmp/abcdef0123/chart.svg", format="svg",
                                 role="annex", produced_by="chart")])
    notice = JobNotificationMiddleware._notice_for(job)

    assert "was CANCELLED before it finished" in notice
    assert "FAILED" not in notice
    assert "/tmp/abcdef0123/chart.svg" in notice
    assert "not as a report" in notice
    # a cancelled job has no failure message, but it can have a delivery one
    assert "gone.svg" in notice


async def test_a_cancelled_job_reaches_the_conversation(store, checkpointer, tmp_path):
    """A job that ends in silence is the defect the DONE branch was fixed for."""
    session, model = make_session(store, checkpointer, tmp_path,
                                  [AIMessage(content="I stopped that one.")])
    job = await session.manager.create_job("crunch numbers",
                                           session_id=session.session_id)
    await session.manager.cancel_job(job.job_id)          # never started: tombstone
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("what happened to it?")]}, CFG)

    (notice,) = notices(model.calls[0], NOTICE_MARKER)
    assert job.job_id[:8] in notice.content and "CANCELLED" in notice.content
    assert (await session.manager.get_job(job.job_id)).announced is True


async def test_a_resumed_job_is_news_again(store, checkpointer, tmp_path):
    """The trap in making a cancellation announceable: `announced` is set when
    the STOP is announced, so a job resumed to DONE afterwards would be
    filtered out of `list_finished_unannounced` and its answer would never
    reach the conversation that asked for it. `_begin_resume` unmarks it, for
    the same reason it clears `job.error`: a resumed job is news again."""
    from test_jobs import CountingEcho

    alpha, slow = CountingEcho("alpha"), CountingEcho("slow", delay=30.0)
    llm = FakeLLM(
        {"planner": plan_json("alpha", "slow", deps={"slow": ["alpha"]})},
        default="A sufficiently long final answer for the job test.",
    )
    manager = make_manager(store, checkpointer, tmp_path, caps=[alpha, slow], llm=llm)
    model = ScriptedChatModel(responses=[AIMessage(content="I stopped it."),
                                         AIMessage(content="Here it is at last.")])
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    job = await manager.create_job("a job worth resuming", session_id=session.session_id)
    manager.start_job(job.job_id)
    for _ in range(500):                       # wait until `slow` is really running
        await asyncio.sleep(0.01)
        if slow.runs:
            break
    await manager.cancel_job(job.job_id)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("stop that")]}, CFG)
    (stop_notice,) = notices(model.calls[0], NOTICE_MARKER)
    assert "CANCELLED" in stop_notice.content
    assert (await manager.get_job(job.job_id)).announced is True

    slow.delay = 0.0                           # let the interrupted step finish
    done = await manager.resume_job(job.job_id)
    assert done.status is JobStatus.DONE and done.announced is False

    await agent.ainvoke({"messages": [HumanMessage("and now?")]}, CFG)
    (end_notice,) = notices(model.calls[-1], NOTICE_MARKER)
    assert "is DONE" in end_notice.content
    assert done.report_path in end_notice.content      # the answer finally lands


def test_the_notice_still_gives_the_path_when_there_is_one():
    job = Job(job_id="abcdef0123", status=JobStatus.DONE, query="q",
              final_answer="The answer.",
              outputs=[JobOutput(path="/tmp/abcdef0123.md")])
    notice = JobNotificationMiddleware._notice_for(job)
    assert "Report file: /tmp/abcdef0123.md" in notice
    assert "No report file" not in notice


async def test_job_tools_are_session_scoped(store, checkpointer, tmp_path):
    """job_status must not resolve another session's job."""
    session, _ = make_session(store, checkpointer, tmp_path, [
        AIMessage(
            content="",
            tool_calls=[{"name": "job_status", "args": {"job_id_prefix": ""}, "id": "c1"}],
        ),
        AIMessage(content="done"),
    ])
    foreign = await session.manager.create_job("someone else's job", session_id="other-session")
    agent = session.build()
    out = await agent.ainvoke({"messages": [HumanMessage("status?")]}, CFG)
    tool_msg = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "No unique job" in tool_msg.content
    assert foreign.job_id[:8] not in tool_msg.content


# ---------------- Carrying the conversation's referent into the job ----------

async def poll_until_settled(manager, job_id):
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await manager.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            return job
    raise AssertionError("job never settled")


async def test_launch_carries_referent_from_an_earlier_turn(store, checkpointer, tmp_path):
    """The bug: after a few turns the model writes "analyse that", and the job
    engine — which never sees the thread — plans against a request whose
    referent is gone. The recent turns must travel in `inputs`."""
    session, _ = make_session(store, checkpointer, tmp_path, [
        AIMessage(content="Right — the Q3 churn spike in the alpha cohort."),
        launch_call("analyse that", "several capability steps"),
        AIMessage(content="Launched."),
    ])
    agent = session.build()

    await agent.ainvoke(
        {"messages": [HumanMessage("we saw a Q3 churn spike in the alpha cohort")]}, CFG
    )
    await agent.ainvoke({"messages": [HumanMessage("analyse that")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.query == "analyse that"          # the model's wording is untouched
    excerpt = job.inputs[CONVERSATION_INPUT_KEY]
    assert "Q3 churn spike in the alpha cohort" in excerpt   # the referent travelled
    assert "user: analyse that" in excerpt
    assert "assistant: Right" in excerpt


async def test_planner_prompt_receives_the_conversation(store, checkpointer, tmp_path):
    """End to end: what the chat tool attaches reaches the planner's prompt."""
    llm = FakeLLM(
        {"planner": plan_json("alpha")},
        default="A sufficiently long final answer for the job test.",
    )
    session, _ = make_session(store, checkpointer, tmp_path, [
        AIMessage(content="Noted: the beta migration rollback."),
        launch_call("analyse it", "multi-step"),
        AIMessage(content="Launched."),
    ], llm=llm)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("the beta migration rollback")]}, CFG)
    await agent.ainvoke({"messages": [HumanMessage("analyse it")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    await poll_until_settled(session.manager, job.job_id)

    planner_call = next(
        c for c in llm.calls
        if any("planner" in (m.get("content") or "")
               for m in c["messages"] if m["role"] == "system")
    )
    user_msg = next(m["content"] for m in planner_call["messages"] if m["role"] == "user")
    assert "beta migration rollback" in user_msg
    assert "Request to plan for:\nanalyse it" in user_msg


async def test_proposal_shows_the_context_that_will_travel(store, checkpointer, tmp_path):
    """What the user approves must include what is being attached.

    Kept on the gate path: `context` is the one field of that payload the
    notice does NOT carry, because a notice is read while the answer is being
    written and the excerpt is machinery, not a decision to check.
    """
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse that", "multi-step"),
        AIMessage(content="Launched."),
    ], approval=True)
    agent = session.build()

    out = await agent.ainvoke({"messages": [HumanMessage("look into the alpha data")]}, CFG)
    (intr,) = out["__interrupt__"]
    assert intr.value["query"] == "analyse that"        # unchanged HITL shape
    assert intr.value["rationale"] == "multi-step"
    assert "look into the alpha data" in intr.value["context"]


async def test_model_supplied_inputs_survive_the_attachment(store, checkpointer, tmp_path):
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the deck", "multi-step", inputs={"image_s3_keys": ["k1"]}),
        AIMessage(content="Launched."),
    ])
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("the deck I uploaded")]}, CFG)

    (job,) = await session.manager.list_jobs(session_id=session.session_id)
    assert job.inputs["image_s3_keys"] == ["k1"]
    assert "the deck I uploaded" in job.inputs[CONVERSATION_INPUT_KEY]


# ---------------- The excerpt is bounded and noise-free ----------------------

def test_excerpt_keeps_only_the_last_turns():
    messages = []
    for i in range(20):
        messages.append(HumanMessage(f"question {i}"))
        messages.append(AIMessage(content=f"answer {i}"))

    excerpt = recent_conversation(messages)
    assert len(excerpt.splitlines()) == MAX_CONTEXT_TURNS
    assert "answer 19" in excerpt
    assert "question 0" not in excerpt
    # chronological, not reversed
    assert excerpt.index("question 17") < excerpt.index("answer 19")


def test_excerpt_is_char_bounded_per_turn_and_overall():
    messages = [HumanMessage("x" * 5000) for _ in range(MAX_CONTEXT_TURNS)]
    excerpt = recent_conversation(messages)
    assert len(excerpt) <= MAX_CONTEXT_CHARS
    for line in excerpt.splitlines():
        assert len(line) <= MAX_TURN_CHARS + len("user: ") + 1
        assert line.endswith("…")


def test_excerpt_drops_machinery_not_prose():
    messages = [
        SystemMessage("[job update] background jobs finished: job 1234"),
        HumanMessage("the alpha cohort churn"),
        launch_call("previous task", "why"),
        ToolMessage(content="Job abcd1234 launched in the background", tool_call_id="call_1"),
        AIMessage(content="I launched it."),
    ]
    excerpt = recent_conversation(messages)
    assert excerpt == "user: the alpha cohort churn\nassistant: I launched it."


def test_excerpt_flattens_content_blocks_and_skips_empty_turns():
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]),
        AIMessage(content=""),
        AIMessage(content="Sure."),
    ]
    assert recent_conversation(messages) == "user: look at this\nassistant: Sure."


def test_empty_conversation_yields_nothing_to_attach():
    """A job launched outside a chat carries nothing extra."""
    assert recent_conversation([]) == ""


# ---------------- In-flight progress: pushed on change, never accumulated ----


async def make_running_job(
    manager, session_id, query, steps, *, done=(), deps=None, status=JobStatus.RUNNING
):
    """A job in mid-flight, written straight to the repository — the point is
    the *rendering* of persisted progress, not another run of the engine."""
    job = await manager.create_job(query, session_id=session_id)
    job.status = status
    job.step_finished_at = {name: now_iso() for name in done}
    await manager.repo.save_summary(job)
    await manager.repo.save_plan(job.job_id, {
        "steps": [{"capability": s, "depends_on": (deps or {}).get(s, [])} for s in steps],
        "rationale": "test plan",
    })
    return job


async def advance(manager, job, capability):
    """One more step lands."""
    fresh = await manager.get_job(job.job_id)
    fresh.step_finished_at[capability] = now_iso()
    await manager.repo.save_summary(fresh)
    return fresh


def notices(call, marker):
    return [m for m in call if isinstance(m, SystemMessage) and marker in m.content]


async def test_running_job_progress_reaches_the_model(store, checkpointer, tmp_path):
    """The wait must not be opaque: a job that is mid-flight says so."""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    await make_running_job(
        session.manager, session.session_id, "analyse the alpha data",
        ["research", "analysis", "critique"],
        done=["research"], deps={"analysis": ["research"], "critique": ["analysis"]},
    )
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("anything new?")]}, CFG)

    (notice,) = notices(model.calls[0], PROGRESS_MARKER)
    assert "1/3 steps done (research)" in notice.content
    assert "running analysis" in notice.content       # the ready wave, from the DAG
    assert "critique" not in notice.content.split("running")[1]   # still blocked
    assert "elapsed" in notice.content


async def test_progress_is_pushed_again_only_when_it_moved(store, checkpointer, tmp_path):
    """Re-sending the same line every turn would spend tokens to say nothing."""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    job = await make_running_job(
        session.manager, session.session_id, "analyse the alpha data",
        ["research", "analysis"], deps={"analysis": ["research"]},
    )
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("start")]}, CFG)
    assert notices(model.calls[-1], PROGRESS_MARKER)          # first sighting

    await agent.ainvoke({"messages": [HumanMessage("and now?")]}, CFG)
    assert not notices(model.calls[-1], PROGRESS_MARKER)      # nothing moved

    await advance(session.manager, job, "research")
    await agent.ainvoke({"messages": [HumanMessage("and now?")]}, CFG)
    (notice,) = notices(model.calls[-1], PROGRESS_MARKER)     # a step landed
    assert "1/2 steps done (research)" in notice.content


async def test_progress_notices_never_accumulate_in_the_thread(store, checkpointer, tmp_path):
    """Five turns of progress must leave zero stale notices behind."""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    job = await make_running_job(
        session.manager, session.session_id, "analyse the alpha data",
        ["a", "b", "c", "d"],
    )
    agent = session.build()

    for step in ["a", "b", "c", "d"]:
        await agent.ainvoke({"messages": [HumanMessage(f"turn {step}")]}, CFG)
        await advance(session.manager, job, step)
        assert len(notices(model.calls[-1], PROGRESS_MARKER)) == 1   # exactly one, fresh

    persisted = (await agent.aget_state(CFG)).values["messages"]
    assert not [m for m in persisted if isinstance(m, SystemMessage)]
    # ... and the last request carried one notice, not four stacked ones
    assert len(notices(model.calls[-1], PROGRESS_MARKER)) == 1


async def both_notices_turn(store, checkpointer, tmp_path):
    """One turn carrying both notices at once: a finished job to announce and
    a running one to report on."""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    done = await session.manager.create_job("crunch numbers", session_id=session.session_id)
    await session.manager.run_job(done.job_id)
    await make_running_job(
        session.manager, session.session_id, "analyse the alpha data",
        ["research", "analysis"], done=["research"], deps={"analysis": ["research"]},
    )
    agent = session.build()
    await agent.ainvoke({"messages": [HumanMessage("unrelated question")]}, CFG)
    return model.calls[-1]


async def test_notices_stay_adjacent_to_the_leading_system_prompt(
    store, checkpointer, tmp_path
):
    """The placement is forced by the provider, not by taste: Anthropic rejects
    any SystemMessage that is not adjacent to the leading system block."""
    call = await both_notices_turn(store, checkpointer, tmp_path)

    systems = [i for i, m in enumerate(call) if isinstance(m, SystemMessage)]
    assert systems == list(range(len(systems)))         # one contiguous leading block
    assert NOTICE_MARKER in call[1].content             # completion: act on it now
    assert PROGRESS_MARKER in call[2].content           # progress: background
    # the conversation itself is untouched — the user's turn is still the turn
    # being answered, so a status line is never mistaken for the thing to reply to
    assert isinstance(call[-1], HumanMessage)
    assert call[-1].content == "unrelated question"


async def test_notices_survive_the_real_provider_formatters(store, checkpointer, tmp_path):
    """The bug this class of test exists for: every other test scripts the
    model, so no notice's *formatting* was ever exercised. Both notices used to
    be emitted non-adjacent, which `langchain_anthropic` refuses outright —
    silently breaking the chat layer's headline feature on Claude."""
    anthropic = pytest.importorskip("langchain_anthropic.chat_models")
    openai = pytest.importorskip("langchain_openai.chat_models.base")
    call = await both_notices_turn(store, checkpointer, tmp_path)

    system, formatted = anthropic._format_messages(call)   # raises on the old placement
    hoisted = " ".join(block["text"] for block in system)
    assert NOTICE_MARKER in hoisted and PROGRESS_MARKER in hoisted
    assert [m["role"] for m in formatted] == ["user"]      # only the real turn remains

    as_dicts = [openai._convert_message_to_dict(m) for m in call]
    assert [d["role"] for d in as_dicts[:3]] == ["system"] * 3
    assert as_dicts[-1] == {"role": "user", "content": "unrelated question"}


def test_a_tool_calling_thread_also_formats(store, checkpointer, tmp_path):
    """Same guarantee on the messiest thread shape: mid-launch, with a tool
    call and its result between the prompt and the latest turn."""
    anthropic = pytest.importorskip("langchain_anthropic.chat_models")
    thread = [
        SystemMessage("system prompt"),
        HumanMessage("analyse the alpha data"),
        launch_call("analyse the alpha data", "multi-step"),
        ToolMessage(content="Job abcd1234 launched", tool_call_id="call_1"),
        HumanMessage("meanwhile, what is the alpha cohort?"),
    ]
    notice = SystemMessage(f"[job progress] {PROGRESS_MARKER}: 1/2 steps done")

    injected = JobNotificationMiddleware._inject(thread, [notice])
    system, formatted = anthropic._format_messages(injected)

    assert PROGRESS_MARKER in " ".join(block["text"] for block in system)
    assert [m["role"] for m in formatted] == ["user", "assistant", "user"]


async def test_settled_jobs_are_announced_not_reported_as_running(store, checkpointer, tmp_path):
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    job = await session.manager.create_job("crunch numbers", session_id=session.session_id)
    await session.manager.run_job(job.job_id)
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("hi")]}, CFG)

    assert len(notices(model.calls[0], NOTICE_MARKER)) == 1
    assert not notices(model.calls[0], PROGRESS_MARKER)


async def test_progress_is_scoped_to_this_session(store, checkpointer, tmp_path):
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    foreign = await make_running_job(
        session.manager, "other-session", "someone else's job", ["research"],
    )
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("hi")]}, CFG)
    assert not notices(model.calls[0], PROGRESS_MARKER)
    assert foreign.job_id[:8] not in str(model.calls[0])


async def test_no_jobs_means_no_injection_at_all(store, checkpointer, tmp_path):
    """The common case — plain chat — pays nothing."""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="hello")])
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("hi")]}, CFG)
    assert not notices(model.calls[0], PROGRESS_MARKER)
    assert not notices(model.calls[0], NOTICE_MARKER)
    assert len(model.calls[0]) == 2      # the system prompt and the user turn, nothing else


async def test_progress_notice_is_dropped_from_a_launch_excerpt(store, checkpointer, tmp_path):
    """Machinery must not travel into the job engine as conversation."""
    excerpt = recent_conversation([
        SystemMessage(f"[job progress] {PROGRESS_MARKER}: 1a2b3c4d 1/3 steps done"),
        HumanMessage("the alpha cohort"),
    ])
    assert excerpt == "user: the alpha cohort"


# ---------------- Progress rendering, derived from persisted job data -------


def make_job(**kwargs) -> Job:
    base = {"job_id": "abcd1234ef", "status": JobStatus.RUNNING, "query": "q", "created_at": ""}
    return Job(**(base | kwargs))


def test_running_steps_is_the_ready_wave():
    job = make_job(
        plan={"steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "analysis", "depends_on": ["research"]},
            {"capability": "critique", "depends_on": ["analysis"]},
        ], "rationale": ""},
        step_finished_at={"research": now_iso()},
    )
    assert running_steps(job) == ["analysis"]


def test_a_settled_job_has_no_running_steps():
    """A cancelled run leaves unfinished steps; none of them is running."""
    job = make_job(
        status=JobStatus.CANCELLED,
        plan={"steps": [{"capability": "research", "depends_on": []}], "rationale": ""},
    )
    assert running_steps(job) == []


def test_progress_line_before_the_plan_exists():
    job = make_job(status=JobStatus.QUEUED, plan=None)
    assert "queued, planning" in progress_line(job)


def test_progress_signature_ignores_elapsed_time_only():
    job = make_job(
        plan={"steps": [{"capability": "research", "depends_on": []}], "rationale": ""},
    )
    before = progress_signature(job)
    assert progress_signature(make_job(plan=job.plan, created_at="2020-01-01T00:00:00+00:00")) \
        == before                                       # age alone is not news
    job.step_finished_at = {"research": now_iso()}
    assert progress_signature(job) != before            # a landed step is
