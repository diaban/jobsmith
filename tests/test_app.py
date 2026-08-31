"""Global agent composition: build_app + fakes run the whole product keyless."""
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent_oo.app import build_app
from agent_oo.app.providers import KeywordChatModel, KeywordLLM
from agent_oo.jobs.models import JobStatus


def make_app(tmp_path):
    return build_app(
        llm=KeywordLLM(),
        chat_model=KeywordChatModel(),
        reports_dir=str(tmp_path / "artifacts"),
    )


async def test_default_pack_job_runs_keyless(tmp_path):
    app = make_app(tmp_path)
    job = await app.manager.create_job("study the topic in depth")
    done = await app.manager.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    # KeywordLLM chains every registered capability from the planner prompt
    assert set(done.results) == {"research", "analysis", "critique"}
    assert [s["capability"] for s in done.plan["steps"]] == ["research", "analysis", "critique"]
    assert done.report_path is not None


async def test_chat_session_proposes_and_launches(tmp_path):
    app = make_app(tmp_path)
    session = app.new_session()
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}

    out = await agent.ainvoke({"messages": [HumanMessage("please research topic X")]}, cfg)
    assert "__interrupt__" in out  # complexity detected → HITL proposal

    out = await agent.ainvoke(Command(resume={"approved": True}), cfg)
    assert "launched in the background" in out["messages"][-1].content
    (job,) = await app.manager.list_jobs(session_id=session.session_id)
    assert job.query == "please research topic X"


async def test_direct_answer_stays_in_chat(tmp_path):
    app = make_app(tmp_path)
    session = app.new_session()
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}
    out = await agent.ainvoke({"messages": [HumanMessage("hi there!")]}, cfg)
    assert "__interrupt__" not in out
    assert await app.manager.list_jobs() == []
