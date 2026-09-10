"""What a path is allowed to be — answered once, for everything that asks.

Two different questions reach a filesystem in this project, and conflating
them is how one of them ends up unguarded:

    a NAME      one component, chosen by a model or a user, appended to a
                directory THIS code picked — `safe_name`. It carries no
                separator and is never `.` or `..`, so it cannot climb out of
                the directory it is written into. `LocalArtifactStore` names a
                capability's file with it, and a requested report filename
                (#55) is the same question with the same answer: the caller
                says what the file is called, never where it goes.

    a LOCATION  a whole path, given by a model or a user, pointing at a file
                that already exists — `resolve_within`. It is refused unless
                it **lands** inside one of the roots the deployment declared
                readable.

`resolve_within` resolves before it compares, and that is the whole of the
rule. `..` is normalised away and every symlink on the way is followed, so a
link pointing out of the root is refused exactly like `../../etc/passwd`: the
test is on where the path *lands*, never on how it is spelled. Rejecting
spellings is a blacklist, and a blacklist is a race against whoever writes
the next spelling.

**An absolute path is not refused on sight.** This product prints paths — a
finished job's report, the TUI's artifacts pane, `jobsmith outputs` — and the
path a user pastes back is the path they were shown. Some of those are
absolute, and refusing them would refuse the product's own output while
teaching nobody anything about traversal. So an absolute path takes exactly
the same test: resolve, then be inside a declared root. One that lands
outside is refused with the same message a relative one gets.

**Which roots those are is a deployment decision.** It is made in the
composition root and handed down (`AgentContext.readable_roots`), never
invented by a capability and never widened by one — a capability that could
choose its own root would be a capability that could choose `/`.

A refusal always says why, in the terms the person used: it will be read by
someone who named a file and got nothing back, and "not permitted" without a
reason is indistinguishable from a bug.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class PathRefused(ValueError):
    """A path may not be used, and the message says why.

    A `ValueError` because that is what it always was where this started
    (`LocalArtifactStore` refusing a job id with a slash in it), so callers
    that already catch one keep working.
    """


def safe_name(value: str, what: str = "name") -> str:
    """One path component that cannot climb out of its parent.

    Whitespace and leading/trailing separators are stripped — a model writing
    `"/report.md"` means the name, not the root — and anything still holding a
    separator is refused rather than quietly flattened: a caller that meant a
    *location* must be told so, not handed a file somewhere else.
    """
    cleaned = (value or "").strip().strip("/\\")
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise PathRefused(f"unusable {what}: {value!r}")
    return cleaned


def resolve_within(ref: str, roots: Iterable[str | Path]) -> Path:
    """The file `ref` names, if it lands inside one of `roots`. Else refuse.

    A relative `ref` is taken against each root in turn (the first that
    contains it wins); an absolute one is tested as it stands. Both are
    resolved — symlinks followed, `..` collapsed — before the containment
    check, which is what makes a symlink out of the tree indistinguishable
    from a `..` out of it: both land outside, both are refused.

    Existence is deliberately NOT required here. Whether the file is there,
    is a file at all, and is readable is the adapter's business and produces
    a different message; this function answers one question only, and answers
    it for a path that does not exist yet as readily as for one that does.
    """
    given = str(ref or "").strip()
    if not given:
        raise PathRefused("no file was named")
    if "\x00" in given:
        raise PathRefused(f"{given!r} is not a usable path")
    candidate = Path(given).expanduser()

    allowed: list[str] = []
    for root in roots:
        try:
            base = Path(root).expanduser().resolve()
            landed = (candidate if candidate.is_absolute() else base / candidate).resolve()
        except OSError:                 # an unreachable root refuses, never raises
            continue
        allowed.append(str(base))
        if landed == base or landed.is_relative_to(base):
            return landed
    where = ", ".join(allowed) if allowed else "nothing is readable here"
    raise PathRefused(f"{given!r} is outside the readable area ({where})")


__all__ = ["PathRefused", "resolve_within", "safe_name"]
