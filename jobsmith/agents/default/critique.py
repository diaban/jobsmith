"""CRITIQUE capability (LLM-only): the findings checked against the material.

**What this step reviews changed** (#82). It used to review the *work* —
"gaps, questionable assumptions, concrete improvements" is a memo to whoever
would redo the job — and that register is what #58 spent an apparatus
containing (a heading saying the block is agent-facing, a role field in the
deck's material, a ban on it becoming the document's voice) and what #73
finally withdrew from the generator altogether: handed a hedged set of notes
and one impeccably structured methodology review, a weak model wrote the
review.

What that left was a step nothing acted on. On a run with no slide deck it
made an LLM call, spent its tokens (6.7k in the measured job, an eighth of
the run) and produced text whose only destinations were the report's
provenance section and the job record. A step that costs a call per run and
whose output nothing reads is a bill, not a capability.

So it is pointed at the subject instead: which claims the material does not
support, where its sources disagree, what the request asks about and the
material does not cover. That is ordinary subject material with an ordinary
consumer — the caveats belong on the statements they bear on, which is
exactly what `GLOBAL_GENERATOR_PROMPT` already asks the generator to do with
partial material — and it is the argument #58 made when it chose to keep the
step: a review that says a claim is unsupported is worth having *before* the
answer is written.

Three things make that a different proposition from what #73 falsified. The
output is **about the subject**, so a model copying its shape copies
statements about the subject rather than a work plan. It is **bullets, few
and short**, so there is no document skeleton to copy in the first place.
And since #81 the notes it checks are written from retrieved passages and
carry their ids, so "the material does not support this" is a claim about
evidence rather than about the model's own memory.
"""
from __future__ import annotations

from typing import ClassVar

from ...core.capability import CapabilitySpec
from ...core.state import CapabilityResult
from ._step import SingleStepCapability, StepState


class CritiqueCapability(SingleStepCapability):
    """Check the findings against the material they were drawn from."""

    spec = CapabilitySpec(
        name="critique",
        description=(
            "check the findings against the material they were drawn from: "
            "claims the material does not support, points where its sources "
            "disagree, and parts of the request it does not cover — the "
            "caveats that belong beside the answer, about the subject and "
            "never about the work; plan it after analysis when analysis is "
            "used"
        ),
        output_schema={
            "type": "object",
            "properties": {"critique": {"type": "string"}},
        },
    )

    SYSTEM = (
        "You are checking the findings about the subject against the material "
        "they were drawn from, before they are written up. Report on the "
        "SUBJECT:\n"
        "- claims the material does not support, and what it does support "
        "instead;\n"
        "- points where the material disagrees with itself, giving both "
        "readings and which is better supported;\n"
        "- parts of the request the material does not cover, with whatever is "
        "known about them anyway.\n"
        "Each point is one short bullet, a statement about the subject, "
        "written so it can be read beside the finding it bears on — name the "
        "thing, the figure or the claim it is about. Never write about the "
        "method, the process, the steps that produced this, or what should be "
        "done next: no recommendations for further work, no plan for "
        "obtaining more material, no template, no questions. Where the "
        "material supports the findings, say so in one line rather than "
        "inventing doubt. At most 8 bullets. Concise markdown."
    )
    #: What this block IS, for the model (`render_context`) and for the human
    #: (`render_report`). It named a review OF THE WORK until #82; it names
    #: caveats about the subject now, because that is what the step produces.
    HEADING = "Caveats on the findings — checked against the material"
    OUTPUT_KEY = "critique"

    #: Both blocks, and that is the point of the change: you cannot say a
    #: claim is unsupported while seeing only the claim. `UPSTREAM`'s
    #: first-match rule (`_step.py`) is a priority chain over restatements of
    #: one thing — the analysis, else the notes it came from — which is right
    #: for a step that reasons *onward* and wrong for one that checks one
    #: against the other. Same shape and same reason as
    #: `ResearchCapability.GROUNDING`: fixed order, every entry read.
    #:
    #: Nothing is bounded here, unlike `research`'s retrieved material. Both
    #: blocks are model-written summaries, bounded by their own producers;
    #: the run's large payload — up to 80 000 characters of retrieved pages
    #: (#75) — never reaches this step.
    MATERIAL: ClassVar[tuple[tuple[str, str, str], ...]] = (
        ("analysis", "analysis", "the findings to check"),
        ("research", "notes", "the notes they were drawn from, with source ids"),
    )
    #: Kept for the base class's fallback: with no `results` at all (a step
    #: planned alone, or its upstream failed) `_material` says so, and this
    #: declares which upstreams the planner should put in front of it.
    UPSTREAM = (("analysis", "analysis"), ("research", "notes"))

    def _material(self, state: StepState) -> str:
        """The findings AND the notes behind them, each labelled with what it is."""
        results = state.get("results", {})
        blocks: list[str] = []
        for name, key, role in self.MATERIAL:
            result = results.get(name)
            if not result or not result.get("ok"):
                continue
            # every CapabilityResult key is NotRequired: a failed step has no
            # `data` at all
            text = (result.get("data") or {}).get(key)
            if text:
                blocks.append(f"[{name} — {role}]\n{text}")
        if not blocks:
            return super()._material(state)
        return "\n\n".join(blocks)

    def render_report(self, result: CapabilityResult) -> str | None:
        """The human's copy, under the same label the generator was given."""
        text = super().render_report(result)
        return f"_{self.HEADING}._\n\n{text}" if text else text
