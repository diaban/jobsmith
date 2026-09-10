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
    #: What the block IS, in the material the generator reads — the same
    #: move `slide_deck` makes with its `MATERIAL` labels (#58). This step is
    #: agent-facing by design: under a heading reading plainly "Critique", its
    #: gaps and suggested improvements arrived as one more section of subject
    #: material, and came back out as a section of the deliverable. The rule
    #: for a block of this kind is in the generator's prompt; this says which
    #: block it is.
    HEADING = "Internal review of the work — not of the subject"
    OUTPUT_KEY = "critique"
    UPSTREAM = (("analysis", "analysis"), ("research", "notes"))
