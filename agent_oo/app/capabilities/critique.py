"""CRITIQUE capability (LLM-only): gaps, weak assumptions, counter-arguments."""
from __future__ import annotations

from ...core.capability import CapabilitySpec
from ._step import SingleStepCapability


class CritiqueCapability(SingleStepCapability):
    """Critically review the analysis (or the research notes)."""

    spec = CapabilitySpec(
        name="critique",
        description=(
            "critically review the produced material: gaps, questionable "
            "assumptions, counter-arguments, concrete improvements — plan it "
            "after analysis when analysis is used"
        ),
        output_schema={
            "type": "object",
            "properties": {"critique": {"type": "string"}},
        },
    )

    SYSTEM = (
        "You are a critical reviewer. Challenge the provided material: list "
        "the gaps, questionable assumptions, and counter-arguments, then "
        "suggest concrete improvements. Concise markdown."
    )
    HEADING = "Critique"
    OUTPUT_KEY = "critique"
    UPSTREAM = (("analysis", "analysis"), ("research", "notes"))
