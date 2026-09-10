"""READ_FILES capability: the documents the request named, read in full.

`documents` searches. This reads. The distinction is the whole point of the
step: a request that says *"make a one-pager out of the report you wrote"*
carries a path, and a ranked search over a directory answers a different
question — often plausibly enough that nobody notices the agent reasoned from
memory about a file it was handed.

Three decisions worth keeping:

**It calls no model.** Reading a named file is not a judgement, and a step
that asked an LLM which of the named files to read would be inventing a
decision the user already made. Its constructor therefore takes the
`DocumentReader` port and nothing else — exactly the clients it needs.

**A refusal is material, not silence.** One unreadable file among three does
not fail the step (the other two are still the answer's grounding), but the
refusal travels into the generation context as well as the report: a model
told nothing about the missing file will happily write around the gap, and
the reader of the deliverable would never learn that a document they named
was never opened.

**Nothing here knows what a path is allowed to be.** The port refuses, the
capability reports what it was told. That rule lives in `core/paths.py` and
is the deployment's to configure — a capability that could widen it would be
a capability that could read anything.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.state import SOURCE_FILES_INPUT_KEY, AgentState, CapabilityResult
from .sources import DocumentReader


class ReadFilesState(CapabilityBaseState, total=False):
    read: list[dict]
    refused: list[str]


def named_files(inputs: dict[str, Any] | None) -> list[str]:
    """The file references an `inputs` dict carries, tolerating its shape.

    `inputs` is an open dict filled by a model through `launch_job`, so a
    single string where a list was documented is a thing that happens; it is
    read generously and cleaned, exactly as `artifact_refs` reads `meta`.
    Anything else answers "no file was named", which is a fact about the
    request and not an error to raise on.
    """
    declared = (inputs or {}).get(SOURCE_FILES_INPUT_KEY)
    if isinstance(declared, str):
        declared = [declared]
    if not isinstance(declared, list | tuple):
        return []
    return [ref for ref in (str(d).strip() for d in declared) if ref]


class ReadFilesCapability(Capability):
    """Read the files the request explicitly named."""

    spec = CapabilitySpec(
        name="read_files",
        description=(
            "read, in full, the files the request explicitly names — a path the "
            "user gave, or a deliverable an earlier job produced — so the answer "
            "works from the actual document; use it whenever the request points "
            "at a specific file, and prefer `documents` when it asks what the "
            "available material says about a topic instead of naming a file"
        ),
        requires_inputs=(SOURCE_FILES_INPUT_KEY,),
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
                "unreadable": {"type": "array", "items": {"type": "string"}},
            },
        },
    )

    def __init__(self, reader: DocumentReader, *, max_files: int = 5):
        self.reader = reader
        self.max_files = max_files

    # -------------------- Planner integration --------------------

    def is_applicable(self, state: AgentState) -> bool:
        """The key being present is not enough — it has to name something.

        `super()` asks whether `source_files` is in `inputs`, and
        `{"source_files": []}` satisfies that while naming no file at all.
        Planning a step whose only possible report is "nothing was given" is
        exactly what dropping an inapplicable step exists to prevent.
        """
        return bool(named_files(state.get("inputs")))

    # -------------------- Nodes --------------------

    async def read_all(self, state: ReadFilesState) -> dict:
        """Read each named file; a refusal is recorded, never raised."""
        read: list[dict] = []
        refused: list[str] = []
        for ref in named_files(state.get("inputs"))[: self.max_files]:
            try:
                document = await self.reader.read(ref)
            except Exception as unavailable:
                # One unreadable file must not cost the others — same shape as
                # `documents` isolating a failing query. The port's message
                # already names the file it is about.
                refused.append(str(unavailable) or f"{ref!r}: unavailable")
                continue
            read.append({"id": document.id, "title": document.title,
                         "source": document.source, "text": document.text})
        return {"read": read, "refused": refused}

    async def emit_success(self, state: ReadFilesState) -> dict:
        # Reached only when the router saw a non-empty `read`, written by
        # read_all — which writes both channels or neither.
        read = state.get("read") or []
        refused = state.get("refused") or []
        return self._emit_success(
            {"documents": read, "unreadable": refused},
            meta={"file_count": len(read), "unreadable": refused},
        )

    async def emit_failure(self, state: ReadFilesState) -> dict:
        refused = state.get("refused") or []
        detail = "; ".join(refused) or "no file was named"
        return self._emit_failure(
            f"none of the named files could be read: {detail}",
            meta={"unreadable": refused},
        )

    # -------------------- Router --------------------

    def route_after_read(self, state: ReadFilesState) -> Literal["success", "failure"]:
        return "success" if state.get("read") else "failure"

    # -------------------- Rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        """For the model: the documents themselves, and what was NOT read.

        The refusals are in the context on purpose. A model handed two of the
        three files it was told about writes a confident answer over the hole;
        told which one is missing, it can say so.
        """
        data = result.get("data") or {}
        read = data.get("documents") or []
        refused = data.get("unreadable") or []
        if not read and not refused:
            return None
        blocks = [f"## [{d['id']}] {d['title']}\n\n{d['text']}" for d in read]
        if refused:
            blocks.append(
                "## Files that could NOT be read\n\n"
                + "\n".join(f"- {why}" for why in refused)
                + "\n\nSay so in the answer rather than working around the gap."
            )
        return "# Named documents\n\n" + "\n\n".join(blocks)

    def render_report(self, result: CapabilityResult) -> str | None:
        """For the human: which files were opened, and which were refused."""
        data = result.get("data") or {}
        read = data.get("documents") or []
        refused = data.get("unreadable") or list((result.get("meta") or {}).get(
            "unreadable") or [])
        lines = [f"- `{d['id']}` — {d['source']}" for d in read]
        lines += [f"- _not read: {why}_" for why in refused]
        if not lines:
            return f"_{result.get('error') or 'no file was read'}_"
        return "\n".join(lines)

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(ReadFilesState)
        g.add_node("read_all", self.read_all)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("read_all")
        g.add_conditional_edges("read_all", self.route_after_read, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
