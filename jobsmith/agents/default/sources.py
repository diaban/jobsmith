"""Where the default agent's material comes from.

`DocumentSource` is the **port**: what the `documents` capability needs, said
in its own terms. It is deliberately not shaped like any particular backend —
a web-search API, a vector store or the local filesystem all fit behind it,
and swapping one for another must not touch the capability.

`LocalFiles` is the first adapter: the files in a directory, ranked by term
overlap. It needs no key, no network and no service, which is what makes it
usable in tests and in CI.

Deliberately NOT semantic: this is keyword scoring, and the docstrings say so
rather than implying retrieval quality the code does not have. A vector-store
adapter is the next implementation of the same port.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
