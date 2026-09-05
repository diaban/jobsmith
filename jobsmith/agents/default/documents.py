"""DOCUMENTS capability: retrieve real material for the request.

This is the capability that stops a job from being the model talking to
itself. Internal shape: derive search terms from the request (leniently) →
query the `DocumentSource` port → emit the passages, each with an id a later
step can quote.

It depends on the PORT, never on what is behind it: the same capability
serves local files today and a web or vector backend tomorrow.
"""
from __future__ import annotations

import json
from typing import Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.deps import LLMClient
from ...core.state import CapabilityResult
from .sources import Document, DocumentSource


class DocumentsState(CapabilityBaseState, total=False):
    queries: list[str]
    found: list[dict]


class DocumentsCapability(Capability):
    """Search the configured sources and return the relevant passages."""

    spec = CapabilitySpec(
        name="documents",
        description=(
            "retrieve relevant passages from the configured document sources "
            "(real material, quotable by id) — use it whenever the request "
            "concerns specific documents, files or facts rather than general "
            "knowledge"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "source": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
            },
        },
    )

    QUERIES_SYSTEM = (
        "Produce keyword search queries that would surface documents answering "
        'the user\'s request. Return JSON: {"queries": ["<query>", ...]}, 1 to 3 '
        "entries, keywords rather than sentences. No prose, no markdown."
    )

    def __init__(
        self,
        llm: LLMClient,
        source: DocumentSource,
        *,
        max_queries: int = 3,
        per_query: int = 6,
        max_documents: int = 10,
    ):
        self.llm = llm
        self.source = source
        self.max_queries = max_queries
        self.per_query = per_query
        self.max_documents = max_documents

    # -------------------- Nodes --------------------

    async def plan_queries(self, state: DocumentsState) -> dict:
        """Keyword queries beat the raw sentence on a term-based index."""
        try:
            raw = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.QUERIES_SYSTEM},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            queries = [str(q) for q in json.loads(raw).get("queries", []) if str(q).strip()]
        except Exception:
            queries = []
        # lenient by design: an unusable reply degrades to the request itself,
        # which still retrieves something rather than failing the step
        return {"queries": queries[: self.max_queries] or [state["query"]]}

    async def retrieve(self, state: DocumentsState) -> dict:
        # plan_queries always writes a non-empty list (it degrades to the
        # request itself); the same fallback keeps this node honest on its own.
        queries = state.get("queries") or [state["query"]]
        best: dict[str, Document] = {}
        for query in queries:
            try:
                hits = await self.source.search(query, limit=self.per_query)
            except Exception:
                continue                     # one bad query must not lose the others
            for hit in hits:
                kept = best.get(hit.id)
                if kept is None or hit.score > kept.score:
                    best[hit.id] = hit
        ranked = sorted(best.values(), key=lambda d: (-d.score, d.id))[: self.max_documents]
        return {"found": [
            {"id": d.id, "title": d.title, "source": d.source, "text": d.text}
            for d in ranked
        ]}

    async def emit_success(self, state: DocumentsState) -> dict:
        # Reached only when the router saw a non-empty `found`; `queries` was
        # written before it, by plan_queries.
        found = state.get("found") or []
        return self._emit_success(
            {"documents": found},
            meta={"document_count": len(found), "queries": state.get("queries") or []},
        )

    async def emit_failure(self, state: DocumentsState) -> dict:
        return self._emit_failure(
            "no document matched the request in the configured sources"
        )

    # -------------------- Router --------------------

    def route_after_retrieve(self, state: DocumentsState) -> Literal["success", "failure"]:
        return "success" if state.get("found") else "failure"

    # -------------------- Rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        """For the model: the passages, each labelled with a quotable id."""
        found = result.get("data", {}).get("documents") or []
        if not found:
            return None
        blocks = [
            f"## [{d['id']}] {d['title']}\n\n{d['text']}" for d in found
        ]
        return "# Retrieved documents\n\n" + "\n\n".join(blocks)

    def render_report(self, result: CapabilityResult) -> str | None:
        """For the human: where the material came from, not the material."""
        if not result.get("ok"):
            return f"_{result.get('error') or 'no detail'}_"
        found = result.get("data", {}).get("documents") or []
        if not found:
            return "_no source retrieved_"
        return "\n".join(f"- `{d['id']}` — {d['source']}" for d in found)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(DocumentsState)
        g.add_node("plan_queries", self.plan_queries)
        g.add_node("retrieve", self.retrieve)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("plan_queries")
        g.add_edge("plan_queries", "retrieve")
        g.add_conditional_edges("retrieve", self.route_after_retrieve, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()


class WebSearchCapability(DocumentsCapability):
    """The same retrieval, pointed at the web.

    Subclassed rather than parameterised because the ONLY thing that differs
    is the spec — and the spec is what the planner reads to choose between
    "the user's own files" and "what the web says today". Blurring the two
    into one description would take that choice away from it.
    """

    spec = CapabilitySpec(
        name="web_search",
        description=(
            "search the public web for current, external information — use it "
            "for recent events, third-party facts, prices, versions or anything "
            "the local documents cannot contain; prefer `documents` when the "
            "request concerns the user's own material"
        ),
        output_schema=DocumentsCapability.spec.output_schema,
    )

    def render_report(self, result: CapabilityResult) -> str | None:
        """Web ids are URLs, so a link reads better than `id` — source."""
        if not result.get("ok"):
            return f"_{result.get('error') or 'no detail'}_"
        found = result.get("data", {}).get("documents") or []
        if not found:
            return "_no source retrieved_"
        return "\n".join(f"- [{d['title']}]({d['source']})" for d in found)

    async def emit_failure(self, state: DocumentsState) -> dict:
        return self._emit_failure("the web search returned nothing usable")
