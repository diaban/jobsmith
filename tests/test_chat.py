"""Chat layer: HITL job launch, decline, session scoping, job notices."""
from __future__ import annotations

import asyncio

from conftest import FakeLLM, ScriptedChatModel, plan_json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from test_jobs import make_manager

from jobsmith.chat import ChatSession
from jobsmith.chat.session import NOTICE_MARKER, PROGRESS_MARKER
from jobsmith.chat.tools import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TURNS,
    MAX_TURN_CHARS,
    progress_line,
    progress_signature,
    recent_conversation,
    running_steps,
)
from jobsmith.core.state import CONVERSATION_INPUT_KEY
from jobsmith.jobs.models import Job, JobStatus, now_iso


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
    store, checkpointer, tmp_path, responses, *, llm=None
) -> tuple[ChatSession, ScriptedChatModel]:
    manager = make_manager(store, checkpointer, tmp_path, llm=llm)
    model = ScriptedChatModel(responses=responses)
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    return session, model


CFG = {"configurable": {"thread_id": "chat-1"}}


async def test_launch_job_interrupts_then_runs_on_approval(store, checkpointer, tmp_path):
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse the alpha data", "needs several capability steps"),
        AIMessage(content="Job launched — I'll share the report when it's done."),
    ])
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
    for _ in range(200):  # background task → poll to completion
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
    ])
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
    await agent.ainvoke(Command(resume={"approved": True}), CFG)

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
    await agent.ainvoke(Command(resume={"approved": True}), CFG)

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
    """What the user approves must include what is being attached."""
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("analyse that", "multi-step"),
        AIMessage(content="Launched."),
    ])
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
    await agent.ainvoke(Command(resume={"approved": True}), CFG)

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


async def test_progress_sits_before_the_latest_turn(store, checkpointer, tmp_path):
    """Placement is deliberate: progress is background, so the user's message
    stays the last one — a status line must not read as the thing to reply to.
    (The completion notice is the opposite: an instruction to announce now.)"""
    session, model = make_session(store, checkpointer, tmp_path, [AIMessage(content="ok")])
    await make_running_job(
        session.manager, session.session_id, "analyse the alpha data", ["research"],
    )
    agent = session.build()

    await agent.ainvoke({"messages": [HumanMessage("unrelated question")]}, CFG)
    call = model.calls[0]
    assert isinstance(call[-1], HumanMessage)
    assert PROGRESS_MARKER in call[-2].content


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
