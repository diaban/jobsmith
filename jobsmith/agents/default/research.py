"""RESEARCH capability: notes on the request, from the material or from memory.

Internal shape: decompose the request into aspects → produce structured
notes per aspect → emit.

**Where the notes come from is decided at runtime** (#81). If a retrieval
step ran earlier in the plan, its passages are the source and the notes are
written from them; if none did, the step falls back to the model's own
knowledge, which is what it always was. The result says which of the two
happened (`meta["grounded_on"]`), because "the Aeron seats 159 kg" written
from a spec sheet and the same sentence written from memory are not the same
claim, and nothing downstream could tell them apart.

Why this step and not every step: `research` is the one the planner already
puts between the retrieval and the reasoning, so grounding it makes the edge
the plan draws (`web_search → research → analysis`) carry what a reader of
that diagram assumes it carries. Before this, `documents`, `web_search` and
`read_files` had exactly one consumer — the final generator — and everything
in between reasoned from memory.
"""
from __future__ import annotations

import json
from typing import ClassVar, Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.deps import LLMClient
from ...core.state import CapabilityResult
from ._step import SUBJECT_ONLY_RULE

TRUNCATION_NOTE = "\n\n…[truncated: only the first {kept} characters of this document]"


class ResearchState(CapabilityBaseState, total=False):
    aspects: list[str]
    notes: str
    grounded_on: list[str]


#: `read_files`' rule, one step earlier (→ 0081): a refusal is material, so a
#: file that could not be opened is declared, never written up from memory.
UNREADABLE_RULE = (
    "A document listed as impossible to read was asked for and never opened: "
    "name it in the notes as a gap in the material, and never write it up "
    "from your own knowledge as though it had been read."
)


class ResearchCapability(Capability):
    """Break the request into key aspects and write research notes."""

    spec = CapabilitySpec(
        name="research",
        description=(
            "break the request into its key aspects and produce structured "
            "research notes — from the passages a retrieval step found when "
            "the plan has one (plan it after `read_files`, `documents` or "
            "`web_search` and it reads what they retrieved), and from the "
            "model's own knowledge when it does not"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "aspects": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
        },
    )

    #: The upstream results that are *retrieved material*, and the data key
    #: carrying it. All three are read, not the first that matches — which is
    #: the opposite of `SingleStepCapability._material` (`_step.py`) and
    #: deliberately so: there, the tuple is a priority chain over things that
    #: say the same thing at different removes (the analysis, else the notes
    #: it came from), and taking the second as well would hand the model the
    #: same content twice. Here the entries are *sources* — a file the user
    #: named and what the web says today are complementary, and dropping one
    #: because another matched first would lose material nobody can recover
    #: later. Fixed order, so the prompt is deterministic (the `results` dict
    #: arrives in wave order — see the determinism caveat in `core/state.py`).
    #:
    #: `prior_jobs` (#74) is here for the reason #81 exists at all: the
    #: material of an earlier run reaching only the final generator is the
    #: same defect as retrieved passages reaching only it, and a follow-up
    #: whose notes are written from memory about a job it was handed is the
    #: exact failure this pack keeps producing. It sits beside `read_files`
    #: because they are the two things the request POINTED AT — a file, a run
    #: — as against the two that were searched for it.
    GROUNDING: ClassVar[tuple[tuple[str, str], ...]] = (
        ("read_files", "documents"),
        ("prior_jobs", "documents"),
        ("documents", "documents"),
        ("web_search", "documents"),
    )

    #: The other half of a retrieval step's result: what was asked for and
    #: could NOT be obtained. It is `read_files`'s rule, stated in its own
    #: module docstring — "a refusal is material, not silence" — applied one
    #: step earlier than it was written for: it put the refusal in the
    #: generation context because "a model told nothing about the missing
    #: file writes confidently over the hole", and #81 put a step that writes
    #: sourced notes in between. A gap the notes never mention is a gap the
    #: deliverable inherits with no way of knowing.
    #:
    #: Only `read_files` has one, and that is a fact about the two ports
    #: rather than an omission here: a search that matched nothing returned
    #: nothing (the step fails, and "the sources say nothing about X" is not
    #: a document anyone named), while a file that could not be read was
    #: pointed at by the user and is missing from the answer they expect.
    #: `documents` and `web_search` are deliberately absent rather than given
    #: an invented equivalent.
    #:
    #: `prior_jobs` has one for the same reason `read_files` does and not by
    #: analogy: a job the request named and that could not be read is a gap
    #: the reader expects to be filled, and notes written up from memory in
    #: its place are indistinguishable from notes written from the run.
    REFUSALS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("read_files", "unreadable"),
        ("prior_jobs", "unavailable"),
    )

    #: What the retrieved material may spend of the prompt, in characters.
    #:
    #: The worst case handed to this step is real: since #75 `web_search`
    #: returns up to 10 documents of 8 000 characters (80 000 ≈ 20 000
    #: tokens), `documents` can add 10 more and `read_files` up to 5 files of
    #: 40 000 — and every one of those blocks is *already* re-sent to the
    #: generator by `render_context`, so feeding them here whole would double
    #: the run's largest payload before a single note is written. 32 000
    #: characters ≈ 8 000 tokens: room for a page or two of real material per
    #: source, small enough that the notes and the aspects still fit
    #: comfortably beside it in any window this project composes against.
    #:
    #: The budget is shared out document by document (each gets what is left
    #: divided by how many remain), so a long first page cannot crowd out the
    #: last one and short documents hand their unused share back. Every
    #: document reaches the step: nine sources and a silent tenth is how a
    #: contradiction goes unnoticed. What each cut costs is written into the
    #: text, the rule `TavilySource._bounded` and `LocalFileReader` both
    #: follow — a document silently shortened reads as a whole one.
    MAX_MATERIAL_CHARS = 32_000

    DECOMPOSE_SYSTEM = (
        "Identify the key aspects to investigate to fulfil the user's request. "
        'Return JSON: {"aspects": ["<short aspect>", ...]} with 2 to 5 entries. '
        "No prose, no markdown."
    )
    #: "Flag any uncertainty explicitly" is where a wholesale hedge is born
    #: (#73). Asked for it without being told *where* to put it, the model
    #: opened its notes with "these values are indicative and must all be
    #: verified" and marked every figure `[to confirm]` — including the ones
    #: it was sure of. Downstream, that reads as material nobody can use: the
    #: generator of the run this came from declared it could not answer over
    #: 14k characters of sourced specifications. So the instruction now says
    #: where the mark goes and what must not be marked.
    #:
    #: It is the prompt for the ungrounded run, and #81 is why that is worth
    #: saying: the hedge it fights was *honest* for a step writing from
    #: memory, which is all this step could do. What a run with real material
    #: is held to is `GROUNDED_NOTES_SYSTEM`, and it is a different standard.
    NOTES_SYSTEM = (
        "Write structured research notes for the request: one short markdown "
        "section per listed aspect, from your own knowledge. Be factual, and "
        "state what you know as what you know. Where a specific point is "
        "uncertain, mark the doubt on that point ('reported as X, "
        "unconfirmed'). Do not open with a general warning that the notes are "
        "unverified, and do not mark what you are confident about: a blanket "
        "hedge makes usable notes unusable."
    )
    #: The same step with a source, and it is held to a stricter standard than
    #: the one above. Three things it asks for that the recall prompt cannot:
    #: the material is the source (not a hint to be checked against memory),
    #: every point carries the id it came from (the ids exist precisely so a
    #: later step can quote them), and doubt is marked where the *material* is
    #: thin rather than where the model is. The last one is #73's rule applied
    #: to a case #73 did not have: a run with sourced figures that hedges them
    #: all is describing its own memory of the subject, not what it was given.
    GROUNDED_NOTES_SYSTEM = (
        "Write structured research notes for the request: one short markdown "
        "section per listed aspect. The retrieved material above is your "
        "source — report what it says, keeping its figures, names and terms, "
        "and put the id of the document a point comes from next to it, like "
        "[id]. Where the material says nothing about an aspect, say so in one "
        "line and add what you know yourself, marked as your own knowledge. "
        "Mark doubt only where the material is thin on a point or its "
        "documents disagree — never as a general warning, and never on a "
        "point the material states plainly. "
    ) + UNREADABLE_RULE

    def __init__(self, llm: LLMClient, *, max_aspects: int = 5,
                 max_material_chars: int | None = None):
        self.llm = llm
        self.max_aspects = max_aspects
        self.max_material_chars = max_material_chars or self.MAX_MATERIAL_CHARS

    # -------------------- Nodes --------------------

    async def decompose(self, state: ResearchState) -> dict:
        try:
            raw = await self.llm.chat(
                messages=[
                    {"role": "system",
                     "content": self.DECOMPOSE_SYSTEM + SUBJECT_ONLY_RULE},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            aspects = [str(a) for a in json.loads(raw).get("aspects", []) if str(a).strip()]
        except Exception:
            aspects = []
        # lenient by design: an unparseable reply degrades to one broad aspect
        return {"aspects": aspects[: self.max_aspects] or [state["query"]]}

    def _retrieved(self, state: ResearchState) -> tuple[str, list[str], list[str]]:
        """What the retrieval found, what it could not, and which steps ran.

        `results` is NotRequired and read as such: the executor seeds every
        sub-graph with the parent state, so it is there whenever a step ran
        before this one — and absent for the first wave of a plan, for a plan
        that has no retrieval step, and for a graph driven outside a job.
        Empty means "nothing was retrieved", which is a fact about the run and
        the reason the recall prompt still exists.
        """
        found = [
            (name, doc)
            for name, key in self.GROUNDING
            if (result := (state.get("results") or {}).get(name)) and result.get("ok")
            for doc in ((result.get("data") or {}).get(key) or [])
            if isinstance(doc, dict) and (doc.get("text") or "").strip()
        ]
        blocks: list[str] = []
        budget = self.max_material_chars
        for index, (_, doc) in enumerate(found):
            text = self._bounded(str(doc["text"]).strip(), budget // (len(found) - index))
            budget -= len(text)
            blocks.append(f"## [{doc.get('id') or '?'}] {doc.get('title') or ''}\n\n{text}")
        # Read from the SAME results the material came from — a step that
        # failed outright reaches neither this nor `ContextMerger`, which is
        # the boundary `render_context` already draws and not one to move
        # here. In practice a refusal only ever travels beside material: one
        # unreadable file among three is what leaves the step `ok`.
        refused = [
            str(why)
            for name, key in self.REFUSALS
            if (result := (state.get("results") or {}).get(name)) and result.get("ok")
            for why in ((result.get("data") or {}).get(key) or [])
            if str(why).strip()
        ]
        return ("\n\n".join(blocks), refused,
                list(dict.fromkeys(name for name, _ in found)))

    def _bounded(self, text: str, budget: int) -> str:
        """`text` within its share of the budget, cut on a word and marked."""
        if len(text) <= budget:
            return text
        head = text[:budget]
        boundary = max(head.rfind("\n"), head.rfind(" "))
        # Same rule as `TavilySource._bounded`: honour a boundary only when it
        # is near the end, or a block with no whitespace loses real material.
        if boundary > int(budget * 0.8):
            head = head[:boundary]
        head = head.rstrip()
        return head + TRUNCATION_NOTE.format(kept=len(head))

    async def investigate(self, state: ResearchState) -> dict:
        # `aspects` is written by decompose, which always returns a non-empty
        # list — the fallback here is the same one it uses, so an aspect list
        # that somehow never arrived degrades to the request itself.
        aspects = state.get("aspects") or [state["query"]]
        material, refused, grounded_on = self._retrieved(state)
        # The material first, the task last. A prompt whose instruction sits
        # in front of 32 000 characters of documents is an instruction the
        # model has to hold across all of them; behind them, it is the last
        # thing read and the one the notes are written against.
        parts = []
        if material:
            parts.append(f"Retrieved material:\n{material}")
            if refused:
                # Its own labelled block, never mixed into the material: what
                # is missing is not something to reason from, it is something
                # to declare. Same shape as `ReadFilesCapability.render_context`.
                parts.append("Documents that could NOT be read:\n"
                             + "\n".join(f"- {why}" for why in refused))
        parts += [f"Request: {state['query']}",
                  "Aspects:\n" + "\n".join(f"- {a}" for a in aspects)]
        user = "\n\n".join(parts)
        system = self.GROUNDED_NOTES_SYSTEM if material else self.NOTES_SYSTEM
        try:
            notes = await self.llm.chat(
                messages=[
                    {"role": "system", "content": system + SUBJECT_ONLY_RULE},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
        except Exception:
            notes = ""
        return {"notes": notes, "grounded_on": grounded_on}

    async def emit_success(self, state: ResearchState) -> dict:
        # Reached only through route_after_notes == "success", i.e. with
        # non-empty notes; decompose has likewise already written the aspects,
        # and investigate the steps it read (an empty list when there were
        # none, which is exactly what the reader has to be able to tell).
        aspects = state.get("aspects") or []
        return self._emit_success(
            {"aspects": aspects, "notes": state.get("notes") or ""},
            meta={"aspect_count": len(aspects),
                  "grounded_on": state.get("grounded_on") or []},
        )

    async def emit_failure(self, state: ResearchState) -> dict:
        # The same fact on the failing path, for the same reason a step that
        # wrote a file and then broke still declares it (#41): "it had
        # material and produced nothing" and "it had nothing" are two
        # different defects.
        return self._emit_failure(
            "research produced no notes",
            meta={"grounded_on": state.get("grounded_on") or []},
        )

    # -------------------- Router --------------------

    def route_after_notes(self, state: ResearchState) -> Literal["success", "failure"]:
        return "success" if (state.get("notes") or "").strip() else "failure"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        """The notes, under a heading that says what they are made of.

        The generator weighs "notes from the retrieved documents" and "notes
        from the model's own knowledge" differently, and until #81 the second
        was the only thing this step could produce while the heading said
        neither. It is the same distinction `meta["grounded_on"]` records for
        the job's reader; here it costs one clause.
        """
        notes = result.get("data", {}).get("notes")
        if not notes:
            return None
        return f"# Research notes ({self._provenance(result)})\n\n{notes}"

    def render_report(self, result: CapabilityResult) -> str | None:
        """For the human: the notes, and where they came from."""
        if not result.get("ok"):
            return f"_{result.get('error') or 'no detail'}_"
        notes = result.get("data", {}).get("notes") or ""
        return f"_{self._provenance(result).capitalize()}._\n\n{notes}"

    @staticmethod
    def _provenance(result: CapabilityResult) -> str:
        grounded_on = (result.get("meta") or {}).get("grounded_on") or []
        if not grounded_on:
            return "from the model's own knowledge"
        return "from the material retrieved by " + ", ".join(grounded_on)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(ResearchState)
        g.add_node("decompose", self.decompose)
        g.add_node("investigate", self.investigate)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("decompose")
        g.add_edge("decompose", "investigate")
        g.add_conditional_edges("investigate", self.route_after_notes, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
