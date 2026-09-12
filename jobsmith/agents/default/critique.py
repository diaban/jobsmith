"""CRITIQUE capability (LLM-only): gaps, weak assumptions, counter-arguments."""
from __future__ import annotations

from ...core.capability import CapabilitySpec
from ...core.state import CapabilityResult
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
    #: What this block IS. It labelled the block in the generator's material
    #: (#58) and now labels it for the human reading the job record (#73):
    #: whoever opens the per-step material must not read a review of the work
    #: as a verdict on the subject.
    HEADING = "Internal review of the work — not of the subject"
    OUTPUT_KEY = "critique"
    UPSTREAM = (("analysis", "analysis"), ("research", "notes"))

    def render_context(self, result: CapabilityResult) -> str | None:
        """Nothing. This step does not feed the generator (#73).

        #58 had the choice between withholding this material and labelling it,
        and chose the label: `HEADING` says the block is a review OF THE WORK
        and the generator's prompt says to use it as evidence and never as
        voice. A run falsified that choice. Handed hedged notes and one
        impeccably structured methodology review, a weak model wrote the
        review: the deliverable's sections matched `critique`'s nearly one for
        one — a data-collection plan, a verification plan, a blank template —
        while the specifications and conclusions that existed upstream never
        crossed. A label says what a block *is*; it does not stop it being
        copied by a model looking for a shape to follow.

        What it costs: this step no longer reaches the generator or the refine
        cycle. What remains is the human — `render_report`, the annexes of a
        self-contained report, `GET /jobs/{id}` and `jobsmith job <id>` — plus
        `slide_deck`, which reads `results` directly and keeps its own labelled
        block: no observed run shows a deck copying the review, and changing a
        decision on analogy rather than on evidence is what put it here.
        """
        return None

    def render_report(self, result: CapabilityResult) -> str | None:
        """The human's copy, under the label that says what it is.

        `render_context` is now silent, so this is the only place the review
        reaches anybody — and the reader of a job record is exactly as able to
        mistake a review of the work for a verdict on the subject as the model
        was.
        """
        text = super().render_report(result)
        return f"_{self.HEADING}._\n\n{text}" if text else text
