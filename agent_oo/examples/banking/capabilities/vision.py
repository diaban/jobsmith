"""VISION capability — analyse an attached image."""
from __future__ import annotations

from typing import Literal

from langgraph.constants import END

from ....core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ....core.state import CapabilityResult
from ..deps import S3Client, VisionClient


class VisionState(CapabilityBaseState, total=False):
    image_bytes: bytes
    description: str
    retry_count: int


class VisionCapability(Capability):
    """Describe an attached image so its content can inform the answer."""

    spec = CapabilitySpec(
        name="vision",
        description="Analyse an attached image",
        requires_inputs=("image_s3_keys",),
        output_schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "image_s3_key": {"type": "string"},
            },
        },
    )

    def __init__(self, vision: VisionClient, s3: S3Client, *, max_retries: int = 1):
        self.vision = vision
        self.s3 = s3
        self.max_retries = max_retries

    # -------------------- Nodes --------------------

    @staticmethod
    def _keys(state: VisionState) -> list[str]:
        return (state.get("inputs") or {}).get("image_s3_keys") or []

    async def fetch_image(self, state: VisionState) -> dict:
        keys = self._keys(state)
        if not keys:
            return {"image_bytes": b""}
        try:
            return {"image_bytes": await self.s3.get_object(keys[0])}
        except Exception:
            return {"image_bytes": b""}

    async def image_analysis(self, state: VisionState) -> dict:
        img = state.get("image_bytes") or b""
        if not img:
            return {"description": ""}
        try:
            desc = await self.vision.vision(img, prompt=state["query"])
            return {"description": desc}
        except Exception:
            return {"description": ""}

    async def retry_analysis(self, state: VisionState) -> dict:
        out = await self.image_analysis(state)
        return {**out, "retry_count": state.get("retry_count", 0) + 1}

    async def emit_success(self, state: VisionState) -> dict:
        return self._emit_success({
            "description": state.get("description", ""),
            "image_s3_key": (self._keys(state) or [""])[0],
        })

    async def emit_failure(self, state: VisionState) -> dict:
        return self._emit_failure("image fetch or analysis failed after retries")

    # -------------------- Router --------------------

    def route_after_analysis(
        self, state: VisionState
    ) -> Literal["ok", "retry", "exhausted"]:
        if state.get("description"):
            return "ok"
        if state.get("retry_count", 0) >= self.max_retries:
            return "exhausted"
        return "retry"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        desc = result.get("data", {}).get("description", "")
        if not desc:
            return None
        return f"# Image analysis\n\n{desc}"

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(VisionState)
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
