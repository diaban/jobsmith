"""ANALYSIS capability (LLM-only): findings, tensions, implications."""
from __future__ import annotations

from ...core.capability import CapabilitySpec
from ._step import SingleStepCapability


class AnalysisCapability(SingleStepCapability):
    """Analyse the research notes (or the bare request)."""

    spec = CapabilitySpec(
        name="analysis",
        description=(
            "analyse the available material: key findings, tensions, and "
            "implications — plan it after research when research is used"
        ),
        output_schema={
            "type": "object",
            "properties": {"analysis": {"type": "string"}},
        },
    )

    SYSTEM = (
        "You are an analyst. From the request and the provided material, "
        "extract the key findings, the tensions or contradictions, and the "
        "practical implications. Concise markdown."
    )
    HEADING = "Analysis"
    OUTPUT_KEY = "analysis"
    UPSTREAM = (("research", "notes"),)
