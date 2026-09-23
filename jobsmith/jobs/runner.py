"""Driving a graph run, and translating it into domain updates.

This is the **only** module that knows the shape of LangGraph's
`astream(stream_mode="updates")` events — `{node_name: state_update}`, node
names like `cap_<capability>`, terminal node names. Everything above it reacts
to the small typed updates below, so the JobManager never parses graph output
and a test can drive it with a fake runner.

The stream is incremental: a mounted sub-graph is published at the superstep
it completes. What it publishes is its whole output state, though — for a
capability that means the *accumulated* `results`, since `Send` seeds it with
the parent's — so which step just finished is read from the node name and
never from the payload's keys.

Reading progress from the stream (rather than instrumenting nodes) is what
keeps graph nodes job-agnostic: they do not know a Job exists.

There are two ways in — `stream()` starts a run from the job's query,
`resume()` re-enters the thread's checkpoint — and both are translated by the
same code, so the JobManager folds a resumed run exactly like a first one.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..core.state import CapabilityResult, NodeError, Plan

_TERMINAL_NODES = ("post_process", "unanswered", "escalate", "user_error")


@dataclass(frozen=True)
class PlanReady:
    """The planner validated a DAG."""
    plan: Plan


@dataclass(frozen=True)
class StepFinished:
    """One capability produced its result (ok or not)."""
    capability: str
    result: CapabilityResult


@dataclass(frozen=True)
class FormatsChosen:
    """The engine's document step has finished reading the request (#90).

    Three states, the same three `Job.formats` has: a list of names, `[]` for
    "no document at all", and **`None` for "it wrote nothing"**. The last one
    is emitted since #96, because it became a decision: a request nobody
    named a format for and whose sentence asked for no file gets no file,
    and this is the moment that is known. Whether `None` is that silence or
    a caller who had already spoken (the node returns without writing when
    the channel was seeded) is the manager's to tell — it holds the record
    the graph was seeded from, and this module only knows the stream.
    """
    formats: list[str] | None


@dataclass(frozen=True)
class NodeErrors:
    """Errors a node accumulated; recoverable ones do not stop the run."""
    errors: list[NodeError]


@dataclass(frozen=True)
class Terminal:
    """The run reached a terminal node."""
    terminal_kind: str | None
    final_answer: str | None
    user_error_message: str | None


JobUpdate = PlanReady | FormatsChosen | StepFinished | NodeErrors | Terminal


class GraphRunner:
    """Runs a job's graph and yields what happened, in domain terms."""

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def _config(job_id: str) -> dict[str, Any]:
        # job_id doubles as the LangGraph thread_id: the checkpoint of a
        # cancelled or interrupted run stays addressable for a future resume.
        return {"configurable": {"thread_id": job_id}}

    async def stream(
        self,
        job_id: str,
        query: str,
        inputs: dict[str, Any],
        formats: list[str] | None = None,
    ) -> AsyncIterator[JobUpdate]:
        """Start a run from the query.

        `formats` is what the CALLER already asked the document to be, seeded
        into the state so the graph's own document step knows whether anyone
        has spoken (#90). `None` — the request said nothing — is what lets it
        decide; anything else silences it before a single model call.
        """
        async for update in self._translate(self.graph.astream(
            {"query": query, "inputs": inputs, "job_id": job_id,
             "document_formats": formats},
            config=self._config(job_id),
            stream_mode="updates",
        )):
            yield update

    async def resume(self, job_id: str) -> AsyncIterator[JobUpdate]:
        """Re-enter the thread's checkpoint instead of starting a new run.

        `None` as input is LangGraph's "carry on from where you stopped": the
        last completed superstep is replayed from the checkpoint and only the
        tasks that were still pending are executed. A capability interrupted
        mid-flight therefore runs again from its start, while the steps that
        had already finished are *not* re-emitted — which is why the caller
        must keep the results it loaded from the repository.

        Only call this when `pending()` is non-empty: on a thread with no
        checkpoint LangGraph raises (it has no input to start from).
        """
        async for update in self._translate(
            self.graph.astream(None, config=self._config(job_id), stream_mode="updates")
        ):
            yield update

    async def pending(self, job_id: str) -> tuple[str, ...]:
        """Nodes the thread would run next — what a resume would execute.

        Empty means there is nothing to resume: either no checkpoint exists
        (the run never started) or the graph already reached a terminal node.
        """
        snapshot = await self.graph.aget_state(self._config(job_id))
        return tuple(snapshot.next or ())

    async def _translate(self, stream: AsyncIterator[dict]) -> AsyncIterator[JobUpdate]:
        """LangGraph `updates` events → the domain updates above."""
        async for update in stream:
            for node, value in update.items():
                if node == "document_intent" and not (
                    isinstance(value, dict) and "document_formats" in value
                ):
                    # Finished and wrote nothing — LangGraph publishes a node
                    # that returned `{}` as `None`. Announced all the same:
                    # since #96 silence settles the document question too.
                    yield FormatsChosen(None)
                    continue
                if not isinstance(value, dict):
                    continue
                if value.get("errors"):
                    yield NodeErrors(list(value["errors"]))
                if node == "planner" and value.get("plan"):
                    yield PlanReady(value["plan"])
                elif node == "document_intent" and "document_formats" in value:
                    # The NODE NAME again, and the key's PRESENCE: the channel
                    # was seeded at entry with what the caller asked for, so a
                    # value in it is not news. `document_intent` writes it only
                    # when it decided something itself, and `[]` — no document
                    # at all — is exactly such a decision, which is why the
                    # test is `in` and not truthiness.
                    yield FormatsChosen(list(value["document_formats"] or []))
                elif node.startswith("cap_"):
                    # The NODE NAME says which step this is; the update's
                    # `results` does not. A capability sub-graph is seeded with
                    # the whole parent state (`Send(node, state)`) and its
                    # output schema carries `results`, so what LangGraph
                    # publishes here is the union of every step so far — not
                    # this step's contribution. Reading each key of it
                    # re-announced every earlier step on every wave (#53).
                    capability = node[len("cap_"):]
                    result = (value.get("results") or {}).get(capability)
                    if result is not None:
                        yield StepFinished(capability, result)
                elif node in _TERMINAL_NODES:
                    yield Terminal(
                        value.get("terminal_kind"),
                        value.get("final_answer"),
                        value.get("user_error_message"),
                    )
