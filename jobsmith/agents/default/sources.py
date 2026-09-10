"""Where the default agent's material comes from.

**Two ports, because there are two questions.**

`DocumentSource.search(query, limit)` answers *"what do my files say about
X"*: it ranks. `DocumentReader.read(ref)` answers *"give me THAT document"*:
it does not rank, it either hands the file over or says why it cannot. A
`search` that happened to match a filename would be a coincidence, not an
answer, which is why the second is a port of its own rather than an argument
to the first — and why the capabilities consuming them are two steps the
planner chooses between (`documents` / `read_files`) rather than one step
with a mode.

Both are deliberately not shaped like any particular backend: a web-search
API, a vector store, object storage or the local filesystem all fit behind
them, and swapping one for another must not touch the capability.

Adapters here are the local ones, because they need no key, no network and no
service — which is what makes them usable in tests and in CI:

- `LocalFiles` for `DocumentSource` — the files in a directory, ranked by
  term overlap. Deliberately NOT semantic: this is keyword scoring, and the
  docstrings say so rather than implying retrieval quality the code does not
  have. A vector-store adapter is the next implementation of that port.
- `LocalFileReader` for `DocumentReader` — a file named by the request, if it
  lands inside a root the deployment declared readable (`core/paths.py`).
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...core.paths import PathRefused, resolve_within

# Text-ish files worth reading. Binary formats (pdf, docx) need a parser and
# belong in their own adapter, not in a widening list here.
DEFAULT_SUFFIXES = (".md", ".txt", ".rst", ".py", ".json", ".yaml", ".yml", ".toml", ".csv")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", "dist", "build", ".claude"}

_WORD = re.compile(r"[a-z0-9_]{3,}")


@dataclass(frozen=True)
class Document:
    """One piece of retrieved material, whatever produced it."""

    id: str                 # stable, quotable in an answer
    text: str
    title: str = ""
    source: str = ""        # a path, a URL — where a human would go to check
    score: float = 0.0


class DocumentSource(Protocol):
    """The port. One method, because that is all the capability needs."""

    async def search(self, query: str, *, limit: int = 8) -> list[Document]: ...


class DocumentUnavailable(Exception):
    """A named document cannot be served, and the message says why.

    Raised rather than answered with nothing, for the same reason
    `TavilySource` raises on an HTTP error: "there is no such file", "it is
    outside the readable area" and "it is not text" are three different facts
    about a file someone deliberately pointed at, and an empty result flattens
    all three into "that document said nothing" — which is the one reading
    that is never true.
    """


class DocumentReader(Protocol):
    """The port for *this document*, named by the request.

    `read` takes the reference exactly as the request wrote it — a path, as
    far as the local adapter is concerned, but the capability never assumes
    that: an adapter over object storage takes a key, and one over a document
    system takes an id, and neither changes the capability.
    """

    async def read(self, ref: str) -> Document: ...


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _chunks(text: str, *, target: int = 1200) -> list[str]:
    """Split on blank lines, then glue paragraphs up to a target size.

    Paragraph boundaries keep a chunk readable when it lands in a prompt; the
    target size keeps a whole file from swamping the ranking.
    """
    out: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > target:
            out.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        out.append(current)
    return out


class LocalFiles:
    """`DocumentSource` over a directory tree — no key, no network.

    Ranking is term-frequency overlap with the query, not semantics: a chunk
    scores by how many query terms it contains and how often, normalised by
    length so a long file does not win on volume alone.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        suffixes: tuple[str, ...] = DEFAULT_SUFFIXES,
        max_files: int = 500,
        max_bytes: int = 400_000,
    ):
        self.root = Path(root).expanduser().resolve()
        self.suffixes = suffixes
        self.max_files = max_files
        self.max_bytes = max_bytes

    def _files(self) -> list[Path]:
        found: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            if len(found) >= self.max_files:
                break
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            if SKIP_DIRS & set(path.relative_to(self.root).parts):
                continue
            found.append(path)
        return found

    async def search(self, query: str, *, limit: int = 8) -> list[Document]:
        wanted = set(_terms(query))
        if not wanted:
            return []
        scored: list[Document] = []
        for path in self._files():
            try:
                if path.stat().st_size > self.max_bytes:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            relative = path.relative_to(self.root).as_posix()
            for index, chunk in enumerate(_chunks(text)):
                counts = Counter(t for t in _terms(chunk) if t in wanted)
                if not counts:
                    continue
                # distinct terms matter more than repetition; length-normalised
                coverage = len(counts) / len(wanted)
                density = sum(counts.values()) / (len(chunk) / 1000 + 1)
                scored.append(Document(
                    id=f"{relative}#{index}",
                    text=chunk,
                    title=relative,
                    source=str(path),
                    score=round(coverage * 2 + density, 4),
                ))
        scored.sort(key=lambda d: (-d.score, d.id))
        return scored[:limit]


class LocalFileReader:
    """`DocumentReader` over a fixed set of readable roots.

    The roots come from the composition root (`AgentContext.readable_roots`
    plus whatever the agent already exposes), and `core.paths.resolve_within`
    is the entire access rule: a reference is resolved — symlinks followed,
    `..` collapsed — and then has to land inside one of them. Nothing here
    inspects the spelling of a path, so `../../etc/passwd`, a symlink out of
    the tree and an absolute path to somewhere else are one refusal with one
    message, and a root the deployment did add is reachable however the user
    happens to write it.

    Whole files, not chunks: the request named this document, so ranking part
    of it against the request would answer a question nobody asked. What the
    reader does impose is a **budget** — a file longer than `max_chars` is cut
    and the cut is written into the text itself, where both the model and the
    human reading the report can see it. Silence there would be the same
    defect as a dropped token: a shorter document than the one on disk, with
    nothing saying so.
    """

    def __init__(self, roots: Iterable[str | Path], *, max_chars: int = 40_000):
        self.roots = tuple(Path(r).expanduser() for r in roots)
        self.max_chars = max_chars

    async def read(self, ref: str) -> Document:
        try:
            path = resolve_within(ref, self.roots)
        except PathRefused as refused:
            # The policy speaks in paths; the port speaks in documents. One
            # exception type reaches the capability, whatever went wrong.
            raise DocumentUnavailable(str(refused)) from refused
        # to_thread for the same reason `LocalArtifactStore.write` uses it:
        # capability waves run in parallel and a large file is a blocking read.
        text, cut = await asyncio.to_thread(self._read_text, path, ref)
        if cut:
            text += (f"\n\n…[truncated: only the first {self.max_chars} characters "
                     f"of this file were read]")
        return Document(id=ref, text=text, title=Path(ref).name or ref, source=str(path))

    def _read_text(self, path: Path, ref: str) -> tuple[str, bool]:
        """The file's text and whether it was cut short. Refuses, never guesses."""
        if not path.is_file():
            raise DocumentUnavailable(f"{ref!r}: no such file")
        budget = self.max_chars * 4 + 1          # utf-8 is at most 4 bytes a character
        try:
            with path.open("rb") as handle:
                raw = handle.read(budget)
                cut = handle.read(1) != b""
        except OSError as broken:
            raise DocumentUnavailable(
                f"{ref!r}: cannot be read ({broken.strerror or broken})") from broken
        text = self._decode(raw, ref, cut=cut)
        if len(text) > self.max_chars:
            text, cut = text[: self.max_chars], True
        if not text.strip():
            raise DocumentUnavailable(f"{ref!r}: the file is empty")
        return text, cut

    @staticmethod
    def _decode(raw: bytes, ref: str, *, cut: bool) -> str:
        """utf-8, strictly — a decode that "succeeds" by ignoring bytes lies.

        `errors="ignore"` is what retrieval uses (`LocalFiles`), and it is
        right there: one unreadable file among five hundred should not stop a
        search. Here the user named THIS file, so mojibake would be handed
        back as its contents; refusing says the true thing instead.
        """
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as bad:
            # A read stopped at a byte budget can split a character in two;
            # a failure anywhere earlier means the file simply is not text.
            if cut and bad.start >= len(raw) - 3:
                return raw[: bad.start].decode("utf-8")
            raise DocumentUnavailable(
                f"{ref!r}: not UTF-8 text — this reader serves text documents"
            ) from bad
