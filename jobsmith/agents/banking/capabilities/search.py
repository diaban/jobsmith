"""SEARCH (RAG) capability.

Internal shape: rewrite query → search → retry w/ back-off → fallback → emit.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Literal

from langgraph.constants import END

from ....core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ....core.deps import LLMClient
from ....core.state import CapabilityResult
from ..deps import SearchEngine


class SearchState(CapabilityBaseState, total=False):
    generated_query: str
    raw_docs: list[dict[str, Any]]
    retry_count: int
    max_retries: int


class SearchCapability(Capability):
    """RAG over the internal knowledge base."""

    spec = CapabilitySpec(
        name="search",
        description="RAG over the internal knowledge base",
        output_schema={
            "type": "object",
            "properties": {
                "docs": {"type": "array", "items": {"type": "object"}},
                "query_used": {"type": "string"},
            },
        },
    )

    QUERY_REWRITE_SYSTEM = (
        'Rewrite the banker\'s question into a concise search query for a '
        'banking knowledge base. Return JSON: {"query": "<rewritten>"}.\n'
        'No prose, no markdown.'
    )

    def __init__(self, llm: LLMClient, search: SearchEngine, *, max_retries: int = 2, top_k: int = 10):
        self.llm = llm
        self.search = search
        self.max_retries = max_retries
        self.top_k = top_k

    # -------------------- Nodes --------------------

    async def generate_query(self, state: SearchState) -> dict:
        try:
            raw = await self.llm.chat(
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

    async def search_call(self, state: SearchState) -> dict:
        try:
            docs = await self.search.search(state["generated_query"], top_k=self.top_k)
            return {"raw_docs": docs}
        except Exception:
            return {"raw_docs": []}

    async def retry_search(self, state: SearchState) -> dict:
        n = state.get("retry_count", 0) + 1
        # Exponential back-off with jitter
        delay = (2 ** (n - 1)) * 0.5 + random.uniform(0, 0.25)
        await asyncio.sleep(delay)
        try:
            docs = await self.search.search(state["generated_query"], top_k=self.top_k)
            return {"raw_docs": docs, "retry_count": n}
        except Exception:
            return {"raw_docs": [], "retry_count": n}

    async def fallback_search(self, state: SearchState) -> dict:
        try:
            docs = await self.search.search_cached(state["generated_query"], top_k=self.top_k)
            return {"raw_docs": docs}
        except Exception:
            return {"raw_docs": []}

    # -------------------- Terminal nodes --------------------

    def _success_data(self, state: SearchState) -> dict[str, Any]:
        return {
            "docs": state.get("raw_docs", []),
            "query_used": state.get("generated_query", ""),
        }

    async def emit_success(self, state: SearchState) -> dict:
        return self._emit_success(self._success_data(state), meta={"via_fallback": False})

    async def emit_success_via_fallback(self, state: SearchState) -> dict:
        return self._emit_success(self._success_data(state), meta={"via_fallback": True})

    async def emit_failure(self, state: SearchState) -> dict:
        return self._emit_failure("search engine exhausted and fallback failed")

    # -------------------- Routers (sync methods) --------------------

    @staticmethod
    def _has_docs(state: SearchState) -> bool:
        return bool(state.get("raw_docs"))

    def route_after_search(self, state: SearchState) -> Literal["success", "retry"]:
        return "success" if self._has_docs(state) else "retry"

    def route_after_retry(
        self, state: SearchState
    ) -> Literal["retry_ok", "exhausted", "retry_again"]:
        if self._has_docs(state):
            return "retry_ok"
        if state.get("retry_count", 0) >= state.get("max_retries", self.max_retries):
            return "exhausted"
        return "retry_again"

    def route_after_fallback(
        self, state: SearchState
    ) -> Literal["fallback_ok", "fallback_fail"]:
        return "fallback_ok" if self._has_docs(state) else "fallback_fail"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        docs = result.get("data", {}).get("docs", [])
        if not docs:
            return None
        parts = ["# Search results"]
        for i, d in enumerate(docs):
            parts.append(f"[{d.get('id', f'doc_{i}')}] {d.get('text', '')}")
        return "\n\n".join(parts)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(SearchState)

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
