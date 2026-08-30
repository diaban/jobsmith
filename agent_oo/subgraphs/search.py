"""SEARCH (RAG) sub-graph — object-oriented version.

Pattern:
- The class owns `deps` and any configuration (max retries, prompts).
- Each node is an async method, bound to the instance.
- Routers are methods too (sync, since LangGraph routers are sync).
- `build()` returns the compiled graph.

The class is NOT a node itself — instances are not callable. They expose
`build()` which produces a CompiledGraph that the parent graph adds as a node.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Literal

from langgraph.constants import END
from langgraph.graph import StateGraph

from ..deps import Deps
from ..state import NodeError, SearchResult, SearchSubState, SubgraphName


class SearchSubgraph:
    """RAG sub-graph: rewrite query → search → retry w/ back-off → fallback → emit."""

    QUERY_REWRITE_SYSTEM = (
        'Rewrite the banker\'s question into a concise search query for a '
        'banking knowledge base. Return JSON: {"query": "<rewritten>"}.\n'
        'No prose, no markdown.'
    )

    def __init__(self, deps: Deps, *, max_retries: int = 2, top_k: int = 10):
        self.deps = deps
        self.max_retries = max_retries
        self.top_k = top_k

    # -------------------- Nodes --------------------

    async def generate_query(self, state: SearchSubState) -> dict:
        try:
            raw = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.QUERY_REWRITE_SYSTEM},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(raw)
            generated = (parsed.get("query") or "").strip() or state["query"]
        except (json.JSONDecodeError, KeyError):
            generated = state["query"]
        return {
            "generated_query": generated,
            "retry_count": 0,
            "max_retries": self.max_retries,
        }

    async def search_call(self, state: SearchSubState) -> dict:
        try:
            docs = await self.deps.search.search(
                state["generated_query"], top_k=self.top_k
            )
            return {"raw_docs": docs}
        except Exception:
            return {"raw_docs": []}

    async def retry_search(self, state: SearchSubState) -> dict:
        n = state.get("retry_count", 0) + 1
        # Exponential back-off with jitter
        delay = (2 ** (n - 1)) * 0.5 + random.uniform(0, 0.25)
        await asyncio.sleep(delay)
        try:
            docs = await self.deps.search.search(
                state["generated_query"], top_k=self.top_k
            )
            return {"raw_docs": docs, "retry_count": n}
        except Exception:
            return {"raw_docs": [], "retry_count": n}

    async def fallback_search(self, state: SearchSubState) -> dict:
        try:
            docs = await self.deps.search.search_cached(
                state["generated_query"], top_k=self.top_k
            )
            return {"raw_docs": docs}
        except Exception:
            return {"raw_docs": []}

    # -------------------- Terminal nodes --------------------

    async def emit_success(self, state: SearchSubState) -> dict:
        result: SearchResult = {
            "docs": state.get("raw_docs", []),
            "query_used": state.get("generated_query", ""),
            "via_fallback": False,
        }
        return {
            "search_result": result,
            "completed_subgraphs": [SubgraphName.SEARCH.value],
        }

    async def emit_success_via_fallback(self, state: SearchSubState) -> dict:
        result: SearchResult = {
            "docs": state.get("raw_docs", []),
            "query_used": state.get("generated_query", ""),
            "via_fallback": True,
        }
        return {
            "search_result": result,
            "completed_subgraphs": [SubgraphName.SEARCH.value],
        }

    async def emit_failure(self, state: SearchSubState) -> dict:
        err: NodeError = {
            "subgraph": SubgraphName.SEARCH.value,
            "kind": "search_fail",
            "detail": "search engine exhausted and fallback failed",
            "recoverable": True,
        }
        return {
            "completed_subgraphs": [SubgraphName.SEARCH.value],
            "errors": [err],
        }

    # -------------------- Routers (sync methods) --------------------

    @staticmethod
    def _has_docs(state: SearchSubState) -> bool:
        return bool(state.get("raw_docs"))

    def route_after_search(self, state: SearchSubState) -> Literal["success", "retry"]:
        return "success" if self._has_docs(state) else "retry"

    def route_after_retry(
        self, state: SearchSubState
    ) -> Literal["retry_ok", "exhausted", "retry_again"]:
        if self._has_docs(state):
            return "retry_ok"
        if state.get("retry_count", 0) >= state.get("max_retries", self.max_retries):
            return "exhausted"
        return "retry_again"

    def route_after_fallback(
        self, state: SearchSubState
    ) -> Literal["fallback_ok", "fallback_fail"]:
        return "fallback_ok" if self._has_docs(state) else "fallback_fail"

    # -------------------- Compilation --------------------

    def build(self):
        """Return the compiled sub-graph (without checkpointer — parent owns it)."""
        g = StateGraph(SearchSubState)

        # Bound methods are valid LangGraph nodes / routers.
        g.add_node("generate_query", self.generate_query)
        g.add_node("search_call", self.search_call)
        g.add_node("retry_search", self.retry_search)
        g.add_node("fallback_search", self.fallback_search)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_success_fb", self.emit_success_via_fallback)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("generate_query")
        g.add_edge("generate_query", "search_call")

        g.add_conditional_edges("search_call", self.route_after_search, {
            "success": "emit_success",
            "retry": "retry_search",
        })
        g.add_conditional_edges("retry_search", self.route_after_retry, {
            "retry_ok": "emit_success",
            "retry_again": "retry_search",
            "exhausted": "fallback_search",
        })
        g.add_conditional_edges("fallback_search", self.route_after_fallback, {
            "fallback_ok": "emit_success_fb",
            "fallback_fail": "emit_failure",
        })

        g.add_edge("emit_success", END)
        g.add_edge("emit_success_fb", END)
        g.add_edge("emit_failure", END)

        return g.compile()
