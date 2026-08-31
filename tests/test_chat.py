"""Chat layer: HITL job launch, decline, session scoping, completion notices."""
from __future__ import annotations

import asyncio

from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from test_jobs import make_manager

from agent_oo.chat import ChatSession
from agent_oo.jobs.models import JobStatus


def launch_call(query: str, rationale: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "launch_job",
            "args": {"query": query, "rationale": rationale},
            "id": "call_1",
        }],
    )


def make_session(store, checkpointer, tmp_path, responses) -> tuple[ChatSession, ScriptedChatModel]:
    manager = make_manager(store, checkpointer, tmp_path)
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
