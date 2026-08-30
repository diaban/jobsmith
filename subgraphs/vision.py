"""VISION sub-graph — object-oriented version (same pattern as SearchSubgraph)."""
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.constants import END
from langgraph.graph import StateGraph

from ..deps import Deps
from ..state import NodeError, SubgraphName, VisionResult


class VisionSubState(TypedDict, total=False):
    query: str
    image_s3_keys: list[str]
    image_bytes: bytes
    description: str
    retry_count: int
    vision_result: Any
    completed_subgraphs: Any
    errors: Any


class VisionSubgraph:
    def __init__(self, deps: Deps, *, max_retries: int = 1):
        self.deps = deps
        self.max_retries = max_retries

    async def fetch_image(self, state: VisionSubState) -> dict:
        keys = state.get("image_s3_keys") or []
        if not keys:
            return {"image_bytes": b""}
        try:
            return {"image_bytes": await self.deps.s3.get_object(keys[0])}
        except Exception:
            return {"image_bytes": b""}

    async def image_analysis(self, state: VisionSubState) -> dict:
        img = state.get("image_bytes") or b""
        if not img:
            return {"description": ""}
        try:
            desc = await self.deps.llm.vision(img, prompt=state["query"])
            return {"description": desc}
        except Exception:
            return {"description": ""}

    async def retry_analysis(self, state: VisionSubState) -> dict:
        out = await self.image_analysis(state)
        return {**out, "retry_count": state.get("retry_count", 0) + 1}

    async def emit_success(self, state: VisionSubState) -> dict:
        result: VisionResult = {
            "description": state.get("description", ""),
            "image_s3_key": (state.get("image_s3_keys") or [""])[0],
        }
        return {
            "vision_result": result,
            "completed_subgraphs": [SubgraphName.VISION.value],
        }

    async def emit_failure(self, state: VisionSubState) -> dict:
        err: NodeError = {
            "subgraph": SubgraphName.VISION.value,
            "kind": "vision_fail",
            "detail": "image fetch or analysis failed after retries",
            "recoverable": True,
        }
        return {
            "completed_subgraphs": [SubgraphName.VISION.value],
            "errors": [err],
        }

    def route_after_analysis(
        self, state: VisionSubState
    ) -> Literal["ok", "retry", "exhausted"]:
        if state.get("description"):
            return "ok"
        if state.get("retry_count", 0) >= self.max_retries:
            return "exhausted"
        return "retry"

    def build(self):
        g = StateGraph(VisionSubState)
        g.add_node("fetch_image", self.fetch_image)
        g.add_node("image_analysis", self.image_analysis)
        g.add_node("retry_analysis", self.retry_analysis)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("fetch_image")
        g.add_edge("fetch_image", "image_analysis")
        g.add_conditional_edges("image_analysis", self.route_after_analysis, {
            "ok": "emit_success",
            "retry": "retry_analysis",
            "exhausted": "emit_failure",
        })
        g.add_conditional_edges("retry_analysis", self.route_after_analysis, {
            "ok": "emit_success",
            "retry": "emit_failure",
            "exhausted": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
