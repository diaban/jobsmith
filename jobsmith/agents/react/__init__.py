"""The ReAct baseline: one model that gathers its own material and reasons alone.

Every step of the compiler (docs/design/compiler-v1.md) is measured against
this agent on the same tasks — first-try success, cost, latency, calls — or
"compiled is better" stays a claim (#206).

**Its tools are the default agent's retrieval ports, and only those.** Not
the reasoning capabilities: today `analysis` and `critique` take no arguments
and read their material out of the DAG's state by name, so wrapped as tools
they would find nothing and reason "from the request alone" — a baseline
weakened by construction, and a comparison rigged for the compiler. So the
model does the analysis and the critique itself: a compiled orchestration
against an agent that orchestrates and reasons alone.

**Same material, same bounds.** The ports are the ones the default agent
opens (`open_default_resources`, `readable_roots`, the job history), each
tool registered only when something backs it, as a capability would be. A
call returns `{"items": [{id, source, title, text}]}` — the one shape the
compiler's retrieval ops will share — cut to the budget `research` reads its
material within, and a refusal travels as material (0060), never as silence.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from ...dag.prior_jobs import PriorJobSource, PriorJobUnavailable
from ...engine.graph import GraphSpec
from ..base import AgentContext, AgentDefinition
from ..default import DefaultResources, open_default_resources, readable_roots
from ..default.sources import DocumentReader, DocumentSource, DocumentUnavailable, LocalFileReader

#: What one tool call hands the model, at most — the budget `research` reads
#: all its material within (`ResearchCapability.MAX_MATERIAL_CHARS`), so one
#: call of the baseline sees no more than one step of the DAG does.
MAX_CALL_CHARS = 32_000
#: Hits per search call, as `DocumentsCapability.max_documents`.
MAX_HITS = 10
TRUNCATION_NOTE = "\n\n…[truncated: only the first {kept} characters of this item]"

SYSTEM_PROMPT = (
    "You answer the user's request. Gather the material it needs with your tools "
    "when it needs any, then analyse it yourself: the answer first, drawn from the "
    "material, then what in the material limits or contradicts it. Cite an item by "
    "its id when a point comes from it. Say plainly when the material does not "
    "settle the question, and never invent a source."
)


def _bounded(text: str, budget: int) -> str:
    """`text` within `budget`, cut on a word near the end and marked."""
    if len(text) <= budget:
        return text
    head = text[:max(budget, 0)]
    boundary = max(head.rfind("\n"), head.rfind(" "))
    if boundary > int(budget * 0.8):
        head = head[:boundary]
    head = head.rstrip()
    return head + TRUNCATION_NOTE.format(kept=len(head))


def items_payload(items: list[dict[str, str]], *, budget: int = MAX_CALL_CHARS,
                  **extra: Any) -> str:
    """The tool's answer: `items` shared out within `budget`, plus `extra`.

    Share-out, item by item: each takes what is left divided by how many
    remain, so short items hand their share back to the rest.
    """
    shared: list[dict[str, str]] = []
    for index, item in enumerate(items):
        text = _bounded(item.get("text", ""), budget // (len(items) - index))
        budget -= len(text)
        shared.append({"id": item.get("id", ""), "source": item.get("source", ""),
                       "title": item.get("title", ""), "text": text})
    return json.dumps({"items": shared, **extra}, ensure_ascii=False)


def _search_tool(name: str, what: str, source: DocumentSource) -> StructuredTool:
    async def search(query: str) -> str:
        hits = await source.search(query, limit=MAX_HITS)
        return items_payload([{"id": d.id, "source": d.source, "title": d.title,
                               "text": d.text} for d in hits])

    return StructuredTool.from_function(
        coroutine=search, name=name,
        description=f"Search {what} with keyword queries. Returns passages, each "
                    "with a quotable id. Call it again with other keywords if the "
                    "first results miss.")


def _read_file_tool(reader: DocumentReader) -> StructuredTool:
    async def read_file(path: str) -> str:
        try:
            doc = await reader.read(path)
        except DocumentUnavailable as refused:
            return items_payload([], refused=[str(refused) or f"{path!r}: unavailable"])
        return items_payload([{"id": doc.id, "source": doc.source, "title": doc.title,
                               "text": doc.text}], refused=[])

    return StructuredTool.from_function(
        coroutine=read_file, name="read_file",
        description="Read one file the request names, by its path as written. "
                    "Says why when the file cannot be read.")


def _read_prior_job_tool(source: PriorJobSource) -> StructuredTool:
    async def read_prior_job(job_id: str) -> str:
        try:
            job = await source.load(job_id)
        except PriorJobUnavailable as unavailable:
            return items_payload([], unavailable=[str(unavailable)])
        short = job.job_id[:8]
        items = []
        if job.answer.strip():
            items.append({"id": f"{short}#answer", "source": f"job {job.job_id}",
                          "title": f"answer of job {short}", "text": job.answer.strip()})
        items += [{"id": f"{short}#{step.capability}", "source": f"job {job.job_id}",
                   "title": step.capability if step.ok else f"{step.capability} — FAILED",
                   "text": step.text.strip()} for step in job.steps if step.text.strip()]
        return items_payload(items, status=job.status, unavailable=[])

    return StructuredTool.from_function(
        coroutine=read_prior_job, name="read_prior_job",
        description="Read what an earlier job of this product produced, by its full "
                    "job id: its answer and each of its steps.")


def baseline_tools(ctx: AgentContext) -> list[StructuredTool]:
    """The retrieval ports this deployment backs, as tools; none is also valid."""
    resources: DefaultResources = ctx.resources or DefaultResources()
    tools: list[StructuredTool] = []
    if ctx.prior_jobs is not None:
        tools.append(_read_prior_job_tool(ctx.prior_jobs))
    if roots := readable_roots(ctx, resources):
        tools.append(_read_file_tool(LocalFileReader(roots)))
    if resources.documents is not None:
        tools.append(_search_tool("search_documents", "the local documents",
                                  resources.documents))
    if resources.web is not None:
        tools.append(_search_tool("web_search", "the web", resources.web))
    return tools


def _answer(output: dict[str, Any]) -> str:
    return output["messages"][-1].text


def react_graph(ctx: AgentContext,
                tools: Callable[[AgentContext], list[StructuredTool]] = baseline_tools
                ) -> GraphSpec:
    graph = create_agent(ctx.chat_model, tools=tools(ctx), system_prompt=SYSTEM_PROMPT,
                         checkpointer=ctx.checkpointer)
    return GraphSpec("react", graph, result=_answer)


REACT_AGENT = AgentDefinition(
    name="react",
    description=("Baseline for the compiler's measurements: one ReAct loop over the "
                 "retrieval tools, reasoning alone."),
    graph=react_graph,
    open_resources=open_default_resources,
)

__all__ = ["MAX_CALL_CHARS", "REACT_AGENT", "SYSTEM_PROMPT", "baseline_tools",
           "items_payload", "react_graph"]
