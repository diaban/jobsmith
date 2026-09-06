"""Files a capability produces — the port, and the local adapter behind it.

A capability that draws a chart or exports a table has to put the bytes
somewhere, and `Job.outputs` has always had a `role="annex"` /
`produced_by` slot for exactly that. What was missing is the seam between
the two, and it is two small things:

    ArtifactStore   the port: "keep these bytes for this job" (write.py-shaped
                    by the need, not by a filesystem)
    ArtifactRef     what the capability then *declares*, in its result's
                    `meta["artifacts"]` — the key `core/state.py` already
                    documented as carrying "artifact refs"

The manager turns those refs into `JobOutput`s next to the deliverables it
already writes (`jobs/manager.py`), which is the only place that knows where
a job's files live and how a failed write is survived.

**Why a port rather than raw filesystem I/O in the capability.** Same reason
`documents` talks to `DocumentSource` instead of `open()`: where the bytes go
is a deployment decision (a directory today, object storage when the daemon
stops being the only reader), a capability that calls `Path.write_bytes` can
never be pointed elsewhere without editing it, and a fake store makes the
capability testable without a disk. The layout stays the composition root's
business — a capability names its file, never its path.

Two edges of the mechanism, stated rather than hidden: a declaration rides on
a **successful** result (`_emit_failure` takes no `meta`), and the manager
reads declarations only for a job that reached an answer. So a file written by
a step that then failed, or by a run that was cancelled, stays on disk
unrecorded — recording it would mean deciding what a partial deliverable is
worth, and nothing here pretends to have decided that.

`LocalArtifactStore` is the first adapter: `<root>/<job_id>/<name>`, rooted at
the same directory the manager writes deliverables into. The per-job directory
is what keeps two jobs' `chart.svg` apart; keeping names unique *within* a job
is the agent's business (see `write`).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# `CapabilityResult.meta` key under which a capability declares what it wrote.
# A convention, like CONVERSATION_INPUT_KEY in state.py: the manager reads it,
# capabilities write it, and nothing else in the framework interprets `meta`.
ARTIFACTS_META_KEY = "artifacts"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A file a capability produced, as it declares it to the job.

    `path` is whatever the store returned — the capability does not build it.
    `title` is for a human reading `jobsmith outputs`; `format` defaults to the
    file's extension, since that is what it is nine times out of ten.
    """

    path: str
    title: str = ""
    format: str = ""

    def __post_init__(self) -> None:
        if not str(self.path).strip():
            raise ValueError("an artifact ref needs a path")

    @property
    def file_format(self) -> str:
        """Declared format, or the extension the file actually has."""
        return self.format or Path(self.path).suffix.lstrip(".") or "file"

    def to_dict(self) -> dict[str, str]:
        """JSON-safe form — `meta` is persisted and served over HTTP."""
        return {"path": self.path, "title": self.title, "format": self.file_format}

    @classmethod
    def from_dict(cls, data: Any) -> ArtifactRef | None:
        """Read one ref back, tolerating anything a capability may have put there.

        `meta` is an open dict written by code the framework does not control,
        so a malformed entry is dropped rather than raised on: it must not turn
        a job that answered into a job that failed at reporting time.
        """
        if isinstance(data, str):                    # a bare path is generous but clear
            data = {"path": data}
        if not isinstance(data, dict):
            return None
        path = str(data.get("path") or "").strip()
        if not path:
            return None
        return cls(path=path, title=str(data.get("title") or ""),
                   format=str(data.get("format") or ""))


def artifact_meta(*refs: ArtifactRef) -> dict[str, Any]:
    """The `meta` fragment a capability passes to `_emit_success`.

        return self._emit_success(data, meta=artifact_meta(ref))
    """
    return {ARTIFACTS_META_KEY: [ref.to_dict() for ref in refs]}


def artifact_refs(meta: dict[str, Any] | None) -> list[ArtifactRef]:
    """The refs declared in one result's `meta` — never raises, never guesses."""
    declared = (meta or {}).get(ARTIFACTS_META_KEY)
    if isinstance(declared, dict | str):     # one ref, not wrapped in a list
        declared = [declared]
    if not isinstance(declared, list):
        return []
    return [ref for ref in (ArtifactRef.from_dict(d) for d in declared) if ref is not None]


@runtime_checkable
class ArtifactStore(Protocol):
    """Where a capability's file goes. One method, because that is all it needs.

    Async because a remote store is (the local one just does not await
    anything), and returning the path because the capability has to declare
    what it wrote and cannot construct that path itself.
    """

    async def write(self, job_id: str, name: str, data: bytes | str) -> str: ...


class LocalArtifactStore:
    """`ArtifactStore` over a directory: `<root>/<job_id>/<name>`.

    Rooted at the manager's `reports_dir`, so a job's annexes sit next to its
    deliverable and `jobsmith outputs` lists paths under one tree.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    async def write(self, job_id: str, name: str, data: bytes | str) -> str:
        """Write one file for a job and return its path.

        `name` is a filename, not a path: only its last component is kept, so
        a capability cannot escape the job's directory with `../` — it does
        not know the layout and must not be able to reach outside it.

        The same name twice within one job overwrites, exactly like reusing a
        filename anywhere else; a capability that produces several files names
        them apart (its own `spec.name` is the obvious prefix).
        """
        directory = self.root / _safe_segment(job_id, "job id")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _safe_segment(Path(str(name)).name, "artifact name")
        payload = data.encode("utf-8") if isinstance(data, str) else data
        # to_thread: capability waves run in parallel and a big file is a
        # blocking write; the event loop keeps the other branches moving.
        await asyncio.to_thread(path.write_bytes, payload)
        return str(path)


def _safe_segment(value: str, what: str) -> str:
    """One path component that cannot climb out of its parent."""
    cleaned = (value or "").strip().strip("/\\")
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"unusable {what}: {value!r}")
    return cleaned


__all__ = [
    "ARTIFACTS_META_KEY",
    "ArtifactRef",
    "ArtifactStore",
    "LocalArtifactStore",
    "artifact_meta",
    "artifact_refs",
]
