"""What an earlier run established, addressable by its job id.

**The second referent.** A request can point at two different things, and
until now one mechanism served both: *"this file on disk"* — a path the user
names, read by `DocumentReader` (#60) — and *"what job X produced"*, which
had to be spelled as the path of the file that job happened to write. That
round trip costs more than elegance. The report is a deliverable and not a
trace, so re-reading it hands the follow-up the thinnest representation of
the run — the final prose, never the notes or the passages behind it — and
since #85 it is thinner still: a title, the answer and one line naming the
job (`jobs/report.py::job_reference`), everything else about the run being
recorded rather than recited. The file now carries nothing `final_answer`
does not, so reading it back is reading the answer through a filesystem and
losing the material on the way. It also assumes a file that may be on
another machine, may have been deleted, may be a PDF nobody can read back,
and since #84 may never have been written at all: a run leaves a document
because the request asked for one.

So this is a port of its own, and its vocabulary is jobs, not files:
`load(job_id)` answers with what the run produced — the answer it reached and
the material its steps gathered, in plan order — and touches no filesystem.

**Why it lives in `core/` and not next to the capability that consumes it.**
`DocumentSource` and `DocumentReader` sit in `agents/default/sources.py`
because the *agent* opens them (`open_resources` picks the backend). This one
is supplied by the composition root, exactly like `ArtifactStore`: the job
records belong to the deployment's persistence, not to any agent's idea of a
document index. `AgentContext` therefore has to be able to name it, and
`agents/base.py` may not import one agent's module — the same argument that
put `ArtifactStore` in `core/artifacts.py`, and it produces the same shape: a
port here, an adapter over the jobs layer in `jobs/prior.py`, a capability
that imports neither.

Nothing here names a store, a namespace or a schema. Where a job's records
live is `jobs/repository.py`'s secret, and a capability reaching into the
jobs layer for them would be the layering mistake this project avoids
everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PriorJobUnavailable(Exception):
    """A referenced job cannot be served, and the message says why.

    Raised rather than answered with an empty `PriorJob`, for the reason
    `DocumentUnavailable` gives about a named file: "there is no such job" is
    a different fact from "that job established nothing", and an empty answer
    flattens them into the second — which is the one reading that is never
    true of a run somebody deliberately pointed at.

    A job that genuinely produced nothing *yet* is NOT this: it exists, its
    status says why (queued, running, failed before any step landed), and
    that is an answer rather than a refusal. `PriorJob` carries both, so the
    consumer can say which of the two it is looking at.
    """


@dataclass(frozen=True)
class PriorStep:
    """One step of an earlier run, already rendered as text.

    Text rather than the raw `CapabilityResult`: what the consumer needs is
    material it can put in front of a model, and a port shaped by the need is
    a port another backing can satisfy — a daemon answering over HTTP, a
    warehouse of finished runs — without reproducing this framework's result
    schema. `ok` travels because a step that failed established nothing and
    must not read as if it had (see `PriorJob.steps`).
    """

    capability: str
    text: str
    ok: bool = True


@dataclass(frozen=True)
class PriorJob:
    """What one earlier run produced, as far as anyone else needs to know.

    `steps` is in **plan order** — the only deterministic order a run has
    (`results` is filled by parallel waves, see the caveat in `core/state.py`)
    — and carries failed steps too: "the web search failed in that run" is
    material for whoever builds on it, the same reasoning `read_files` gives
    for carrying its refusals rather than staying silent about them.

    `status` and `answer` are separate facts and both are needed: a job that
    is still running, or that ran to the end without being able to answer
    (#59), has steps worth reading and no answer, and a consumer that could
    not tell the two apart would present a partial run as a finished one.
    """

    job_id: str
    query: str = ""
    answer: str = ""
    status: str = ""
    steps: tuple[PriorStep, ...] = ()


class PriorJobSource(Protocol):
    """The port. One method, because that is all the capability needs.

    `load` takes the job id as it was handed over — resolved and scoped by
    whoever accepted the reference (`chat/tools.py` resolves a prefix against
    the session's own jobs), never a prefix to be matched here: a port that
    searched would be a port that can answer with somebody else's run.
    """

    async def load(self, job_id: str) -> PriorJob: ...


__all__ = ["PriorJob", "PriorJobSource", "PriorJobUnavailable", "PriorStep"]
