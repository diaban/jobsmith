"""PRIOR_JOBS capability: what an earlier run of this conversation produced.

`read_files` reads a file. This reads a **job** (#74). The two referents look
alike in a sentence — *"make a one-pager out of that report"* — and are not
the same thing at all: one is a document the user names and this product did
not write, the other is a run of this product, whose answer and whose
per-step material are in the store, addressable by id.

Serving the second with the first is what this replaces, and the round trip
cost five things. The report is a **deliverable, not a trace**: per-step
material is deliberately not inlined into it, so re-reading it hands the
follow-up the thinnest representation of the run. It brings this product's
own scaffolding back as subject matter — *About this job*, the plan table,
the timestamps. It assumes a file that exists, is text and is on this
machine. It needs a path policy for a case that has no path. And since #84 a
run leaves a document only because the request asked for one, so the file may
never have been written.

Three decisions, and they are `read_files`' own for the same reasons:

**It calls no model.** Loading what a run established is not a judgement.
Its constructor takes the `PriorJobSource` port and nothing else.

**A reference that cannot be served is material, not silence.** One
unloadable job among two does not fail the step, and the refusal travels into
the generation context as well as the report — a model told nothing about the
missing run writes confidently over the hole, and #81 is why it also has to
reach `research` rather than only the final generator.

**Nothing here knows where job records live.** The port answers, this
reports. A capability reaching into the jobs layer for a store namespace
would be the layering mistake the port exists to prevent.

What it emits is shaped like a retrieval step's result — `documents` with
quotable ids, `unavailable` beside it — which is not a coincidence and not
an overload: a previous run's answer and each of its steps *are* pieces of
material with an id a later step can quote, and the shape is what lets
`research` read them through the `GROUNDING` machinery #81 already built
instead of growing a second one.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.prior_jobs import PriorJob, PriorJobSource
from ...core.state import FROM_JOBS_INPUT_KEY, AgentState, CapabilityResult

TRUNCATION_NOTE = "\n\n…[truncated: only the first {kept} characters of this material]"


class PriorJobsState(CapabilityBaseState, total=False):
    loaded: list[dict]
    unavailable: list[str]


def referenced_jobs(inputs: dict[str, Any] | None) -> list[str]:
    """The job ids an `inputs` dict carries, tolerating its shape.

    Read exactly as `named_files` reads `source_files`, and for the same
    reason: `inputs` is an open dict a model fills through `launch_job`, so a
    bare string where a list was documented is a thing that happens. Anything
    else answers "no job was referenced", which is a fact about the request
    and not an error to raise on.
    """
    declared = (inputs or {}).get(FROM_JOBS_INPUT_KEY)
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list | tuple):
        return []
    return [ref for ref in (str(d).strip() for d in declared) if ref]


class PriorJobsCapability(Capability):
    """Load the material of the earlier jobs the request builds on."""

    spec = CapabilitySpec(
        name="prior_jobs",
        description=(
            "load what an earlier job produced — the answer it reached and the "
            "material each of its steps gathered — when the request builds on "
            "one ('from that job', 'out of the report you just wrote'); it "
            "reads the run's own records, so it gives more than the document "
            "that run wrote and works even when it wrote none; use "
            "`read_files` instead for a file the user names, and `documents` "
            "to search the available material for a topic"
        ),
        requires_inputs=(FROM_JOBS_INPUT_KEY,),
        output_schema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "source": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
                "unavailable": {"type": "array", "items": {"type": "string"}},
            },
        },
    )

    #: What a previous run's material may spend of the prompt, in characters,
    #: all referenced jobs together.
    #:
    #: The number is set by what is actually there: one measured job carried
    #: 44 000 characters across four steps, and this block is re-sent by every
    #: downstream step that reads the merged context (`research` since #81,
    #: then the generator). 24 000 ≈ 6 000 tokens — comfortably more than the
    #: deliverable a re-read of the file would have given, comfortably less
    #: than a second run's worth of everything.
    #:
    #: It is shared out block by block (each gets what is left divided by how
    #: many remain), so a long analysis cannot crowd out the last step and
    #: short blocks hand their unused share back. Every block reaches the
    #: step: three steps and a silent fourth is how a contradiction goes
    #: unnoticed. What each cut costs is written into the text — the rule
    #: `TavilySource._bounded`, `LocalFileReader` and `research` all follow,
    #: because material silently shortened reads as material that is whole.
    MAX_MATERIAL_CHARS = 24_000

    #: The answer's first claim on that budget. It is the one thing a re-read
    #: of the file WOULD have given, so it must not be the part squeezed out
    #: by the steps behind it; half is enough for any deliverable this product
    #: has produced and leaves the material its share even then.
    ANSWER_SHARE = 0.5

    #: Referenced jobs beyond this are dropped and named as such. A request
    #: builds on one or two runs; a list of ten is a model listing the
    #: session, and the budget above would leave each of them a paragraph.
    MAX_JOBS = 3

    def __init__(self, source: PriorJobSource, *,
                 max_jobs: int | None = None,
                 max_material_chars: int | None = None):
        self.source = source
        self.max_jobs = max_jobs or self.MAX_JOBS
        self.max_material_chars = max_material_chars or self.MAX_MATERIAL_CHARS

    # -------------------- Planner integration --------------------

    def is_applicable(self, state: AgentState) -> bool:
        """The key being present is not enough — it has to name a job.

        `super()` asks whether `from_jobs` is in `inputs`, and
        `{"from_jobs": []}` satisfies that while referencing nothing. Same
        override, same reason, as `read_files`.
        """
        return bool(referenced_jobs(state.get("inputs")))

    # -------------------- Nodes --------------------

    async def load_all(self, state: PriorJobsState) -> dict:
        """Load each referenced job; a refusal is recorded, never raised."""
        refs = referenced_jobs(state.get("inputs"))
        unavailable = [f"{ref}: not loaded (at most {self.max_jobs} jobs are read)"
                       for ref in refs[self.max_jobs:]]
        jobs: list[PriorJob] = []
        for ref in refs[: self.max_jobs]:
            try:
                jobs.append(await self.source.load(ref))
            except Exception as missing:
                # One unloadable reference must not cost the others — the
                # same isolation `read_files` gives a file it cannot open.
                unavailable.append(str(missing) or f"{ref!r}: unavailable")
                continue
        loaded, empty = self._documents(jobs)
        return {"loaded": loaded, "unavailable": unavailable + empty}

    def _documents(self, jobs: list[PriorJob]) -> tuple[list[dict], list[str]]:
        """The blocks these runs produced, bounded, plus what produced none.

        A job that exists and has nothing yet — queued, or stopped before a
        step landed — is not a failure to load: it is a run with nothing to
        say so far, and saying which of the two it is is the whole reason
        `PriorJob` carries a status.
        """
        blocks: list[dict] = []
        empty: list[str] = []
        for job in jobs:
            own = self._blocks(job)
            if not own:
                empty.append(f"job {job.job_id[:8]} ({job.status}) has produced "
                             f"no material yet")
                continue
            blocks += own
        if not blocks:
            return [], empty
        # The answer first, because it is what a re-read of the file would
        # have given; its budget is capped so the steps still get theirs.
        answers = [b for b in blocks if b["kind"] == "answer"]
        rest = [b for b in blocks if b["kind"] != "answer"]
        answer_budget = int(self.max_material_chars * self.ANSWER_SHARE) if rest \
            else self.max_material_chars
        spent = self._fit(answers, answer_budget)
        self._fit(rest, self.max_material_chars - spent)
        return [{k: v for k, v in b.items() if k != "kind"} for b in blocks], empty

    def _fit(self, blocks: list[dict], budget: int) -> int:
        """Cut `blocks` in place to fit `budget`; returns what they spend.

        Share-out, block by block: each takes what is left divided by how
        many remain, so short ones hand their share back to the rest.
        """
        for index, block in enumerate(blocks):
            text = self._bounded(block["text"], max(budget // (len(blocks) - index), 0))
            budget -= len(text)
            block["text"] = text
        return sum(len(b["text"]) for b in blocks)

    def _bounded(self, text: str, budget: int) -> str:
        """`text` within its share of the budget, cut on a word and marked."""
        if len(text) <= budget:
            return text
        head = text[:budget]
        boundary = max(head.rfind("\n"), head.rfind(" "))
        # Same rule as `research._bounded`: honour a boundary only when it is
        # near the end, or a block with no whitespace loses real material.
        if boundary > int(budget * 0.8):
            head = head[:boundary]
        head = head.rstrip()
        return head + TRUNCATION_NOTE.format(kept=len(head))

    def _blocks(self, job: PriorJob) -> list[dict]:
        """One job's material as quotable blocks, the answer first.

        The id is `<short job id>#answer` / `<short job id>#<step>`, the same
        shape retrieved passages carry (`path#chunk`), so a later step can
        cite where a point came from. The source says `job <id>` and not a
        path: nothing here is a file, and a reader who wants the record has
        `jobsmith job <id>`.
        """
        short = job.job_id[:8]
        blocks: list[dict] = []
        if job.answer.strip():
            blocks.append({
                "kind": "answer",
                "id": f"{short}#answer",
                "title": f"answer of job {short} — {job.query[:60]}",
                "source": f"job {job.job_id}",
                "text": job.answer.strip(),
            })
        for step in job.steps:
            if not step.text.strip():
                continue
            label = f"{step.capability} (job {short})"
            blocks.append({
                "kind": "step",
                "id": f"{short}#{step.capability}",
                "title": label if step.ok else f"{label} — this step FAILED",
                "source": f"job {job.job_id}",
                "text": step.text.strip(),
            })
        return blocks

    async def emit_success(self, state: PriorJobsState) -> dict:
        # Reached only when the router saw a non-empty `loaded`, written by
        # load_all — which writes both channels or neither.
        loaded = state.get("loaded") or []
        unavailable = state.get("unavailable") or []
        return self._emit_success(
            {"documents": loaded, "unavailable": unavailable},
            meta={"block_count": len(loaded), "unavailable": unavailable},
        )

    async def emit_failure(self, state: PriorJobsState) -> dict:
        unavailable = state.get("unavailable") or []
        detail = "; ".join(unavailable) or "no job was referenced"
        return self._emit_failure(
            f"none of the referenced jobs could be read: {detail}",
            meta={"unavailable": unavailable},
        )

    # -------------------- Router --------------------

    def route_after_load(self, state: PriorJobsState) -> Literal["success", "failure"]:
        return "success" if state.get("loaded") else "failure"

    # -------------------- Rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        """For the model: the earlier runs' own material, and what is missing.

        The heading says where this came from, for the reason `research`'s
        says whether its notes are grounded: "the Aeron seats 159 kg" carried
        over from a previous run is not a claim this run established, and
        nothing downstream could tell them apart otherwise.
        """
        data = result.get("data") or {}
        loaded = data.get("documents") or []
        unavailable = data.get("unavailable") or []
        if not loaded and not unavailable:
            return None
        blocks = [f"## [{d['id']}] {d['title']}\n\n{d['text']}" for d in loaded]
        if unavailable:
            blocks.append(
                "## Earlier jobs that could NOT be read\n\n"
                + "\n".join(f"- {why}" for why in unavailable)
                + "\n\nSay so in the answer rather than working around the gap."
            )
        return ("# Material from earlier jobs (produced by previous runs, not by "
                "this one)\n\n" + "\n\n".join(blocks))

    def render_report(self, result: CapabilityResult) -> str | None:
        """For the human: which runs were read, and what was not."""
        data = result.get("data") or {}
        loaded = data.get("documents") or []
        unavailable = data.get("unavailable") or list((result.get("meta") or {}).get(
            "unavailable") or [])
        lines = [f"- `{d['id']}` — {d['source']}" for d in loaded]
        lines += [f"- _not read: {why}_" for why in unavailable]
        if not lines:
            return f"_{result.get('error') or 'no earlier job was read'}_"
        return "\n".join(lines)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(PriorJobsState)
        g.add_node("load_all", self.load_all)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("load_all")
        g.add_conditional_edges("load_all", self.route_after_load, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
