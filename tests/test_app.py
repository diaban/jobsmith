"""Global agent composition: build_app + fakes run the whole product keyless."""
from __future__ import annotations

from conftest import registered_capabilities
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from jobsmith.app import build_app
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.jobs.models import JobStatus


async def make_app(tmp_path, *, db: str = "memory"):
    return await build_app(
        llm=KeywordLLM(),
        chat_model=KeywordChatModel(),
        db=db,
        reports_dir=str(tmp_path / "artifacts"),
    )


async def test_default_pack_job_runs_keyless(tmp_path):
    app = await make_app(tmp_path)
    job = await app.manager.create_job("study the topic in depth")
    done = await app.manager.run_job(job.job_id)
    assert done.status is JobStatus.DONE
    # KeywordLLM chains every registered capability from the planner prompt —
    # and which ones those are depends on what is installed and configured
    caps = registered_capabilities(app)
    assert caps[:3] == ["research", "analysis", "critique"]
    assert [s["capability"] for s in done.plan["steps"]] == caps
    assert set(done.results) == set(caps)
    assert done.report_path is not None


async def test_chat_session_runs_the_task_in_the_turn(tmp_path):
    """The composed product, on the nominal path (#83): complexity detected,
    the engine reached, the answer back inside the turn — no card."""
    app = await make_app(tmp_path)
    session = app.new_session()
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}

    out = await agent.ainvoke({"messages": [HumanMessage("please research topic X")]}, cfg)

    assert "__interrupt__" not in out
    (job,) = await app.manager.list_jobs(session_id=session.session_id)
    assert job.query == "please research topic X"
    assert job.status is JobStatus.DONE
    assert job.report_path is not None


async def test_the_kept_approval_gate_composes_too(tmp_path):
    """The same product with `$JOBSMITH_APPROVE_JOBS` on: the interrupt is
    back, and the approved run still settles. Composed rather than unit-built,
    because that flag has to reach the tools through `build_app`'s own
    session factory."""
    app = await make_app(tmp_path)
    session = app.new_session()
    session.approval_required = True
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}

    out = await agent.ainvoke({"messages": [HumanMessage("please research topic X")]}, cfg)
    assert "__interrupt__" in out
    assert await app.manager.list_jobs(session_id=session.session_id) == []

    await agent.ainvoke(Command(resume={"approved": True}), cfg)
    (job,) = await app.manager.list_jobs(session_id=session.session_id)
    assert job.query == "please research topic X"


async def test_direct_answer_stays_in_chat(tmp_path):
    app = await make_app(tmp_path)
    session = app.new_session()
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}
    out = await agent.ainvoke({"messages": [HumanMessage("hi there!")]}, cfg)
    assert "__interrupt__" not in out
    assert await app.manager.list_jobs() == []


def test_the_fake_chat_model_answers_the_tool_the_way_the_prompt_asks():
    """The keyless demo is what a first-time reader sees, so the fake must
    model the behaviour rather than relay the instruction (#83).

    It used to reply `Noted — <tool result>`, which after this change means
    pasting "do NOT repeat the answer" into the conversation, directly under
    the answer it is talking about.
    """
    from langchain_core.messages import ToolMessage

    from jobsmith.app.providers import KeywordChatModel

    def reply(text: str) -> str:
        result = KeywordChatModel()._generate(
            [ToolMessage(content=text, tool_call_id="c1")])
        return result.generations[0].message.text

    delivered = reply(
        "Job abcd1234 finished. Its answer has ALREADY been shown to the user, in "
        "full and word for word — do NOT repeat it.\n"
        "The deliverable is saved at: artifacts/abcd1234.md — worth naming.")
    assert delivered == "Saved to artifacts/abcd1234.md."
    assert "do NOT repeat" not in delivered

    promoted = reply("Job abcd1234 is still running after 20s, so it has been "
                     "moved to the BACKGROUND (short id abcd1234).")
    assert "background" in promoted
    assert "abcd1234" not in promoted     # no result, and no instruction either
