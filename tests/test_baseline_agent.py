"""The ReAct baseline gathers its own material through the retrieval ports and
reasons alone; it is what every compiler step is measured against
(→ docs/design/compiler-v1.md, "Measurement (step −1)", #206)."""
from __future__ import annotations

import json

from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage, ToolMessage

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import DefaultResources, LocalFiles
from jobsmith.agents.react import MAX_CALL_CHARS, REACT_AGENT, baseline_tools, items_payload
from jobsmith.dag.prior_jobs import PriorJob, PriorJobUnavailable, PriorStep
from jobsmith.engine.models import JobStatus

RETRIEVAL = {"read_prior_job", "read_file", "search_documents", "web_search"}


class Priors:
    def __init__(self, *jobs: PriorJob):
        self.jobs = {job.job_id: job for job in jobs}

    async def load(self, job_id: str) -> PriorJob:
        if job_id not in self.jobs:
            raise PriorJobUnavailable(f"{job_id!r}: no such job")
        return self.jobs[job_id]


def tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_a_tool_is_offered_only_when_a_port_backs_it(tmp_path):
    """A capability nothing can serve stays out of the registry; so does a tool.
    And the tools are the retrieval ports only: no reasoning capability, which
    before step 1 would read an empty state and rig the comparison."""
    assert baseline_tools(AgentContext(llm=None)) == []
    backed = AgentContext(llm=None, resources=DefaultResources(
        documents=LocalFiles(tmp_path), web=LocalFiles(tmp_path), documents_root=str(tmp_path)),
        prior_jobs=Priors())
    assert {t.name for t in baseline_tools(backed)} == RETRIEVAL


def test_every_item_has_one_shape_and_a_call_fits_the_budget():
    long = "word " * 20_000
    payload = json.loads(items_payload(
        [{"id": f"d#{i}", "source": "s", "title": "t", "text": long} for i in range(3)]
        + [{"id": "short", "source": "s", "title": "t", "text": "short"}]))
    assert all(set(item) == {"id", "source", "title", "text"} for item in payload["items"])
    assert sum(len(item["text"]) for item in payload["items"]) <= MAX_CALL_CHARS
    assert payload["items"][-1]["text"] == "short"
    assert "[truncated" in payload["items"][0]["text"]


async def test_a_refusal_is_material_not_silence(tmp_path):
    ctx = AgentContext(llm=None, readable_roots=(str(tmp_path),), prior_jobs=Priors(
        PriorJob(job_id="aaaaaaaa1111", answer="Yes.", status="done",
                 steps=(PriorStep("analysis", "why", ok=False),))))
    tools = baseline_tools(ctx)

    refused = json.loads(await tool(tools, "read_file").ainvoke({"path": "../etc/passwd"}))
    assert refused["items"] == [] and refused["refused"]
    missing = json.loads(await tool(tools, "read_prior_job").ainvoke({"job_id": "nope"}))
    assert missing["items"] == [] and "nope" in missing["unavailable"][0]
    prior = json.loads(await tool(tools, "read_prior_job").ainvoke({"job_id": "aaaaaaaa1111"}))
    assert [i["id"] for i in prior["items"]] == ["aaaaaaaa#answer", "aaaaaaaa#analysis"]
    assert "FAILED" in prior["items"][1]["title"]


async def test_the_baseline_searches_then_answers_as_a_job(tmp_path):
    """Composed by the same `build_app`, run by `run_for`: no planner, no DAG."""
    from jobsmith.app.agent import build_app
    from jobsmith.app.providers import make_llm

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "chairs.md").write_text("The Aeron chair supports 159 kg.")
    model = ScriptedChatModel(responses=[
        AIMessage("", tool_calls=[{"id": "c1", "name": "search_documents",
                                   "args": {"query": "Aeron load"}}]),
        AIMessage("159 kg [chairs.md#0].")])
    app = await build_app(agent=REACT_AGENT, llm=make_llm("fake"), chat_model=model,
                          db="memory", reports_dir=str(tmp_path / "reports"),
                          resources=DefaultResources(documents=LocalFiles(docs),
                                                     documents_root=str(docs)))
    try:
        job = await app.engine.create_job(
            {"messages": [{"role": "user", "content": "How much does the Aeron hold?"}]},
            label="Aeron")
        done = await app.engine.run_for(job.job_id, 10)
    finally:
        await app.aclose()

    assert (done.status, done.graph, done.result) == (JobStatus.DONE, "react", "159 kg [chairs.md#0].")
    found = json.loads(next(m for m in model.calls[-1] if isinstance(m, ToolMessage)).content)
    assert "159 kg" in found["items"][0]["text"]
    assert app.dag is None



def test_one_search_reads_what_one_query_of_the_dag_reads():
    """A baseline that read more per query would win on retrieval, not on
    orchestration (→ 0208)."""
    import inspect

    from jobsmith.agents.default import DocumentsCapability
    from jobsmith.agents.react import MAX_HITS

    assert MAX_HITS == inspect.signature(DocumentsCapability).parameters["per_query"].default
