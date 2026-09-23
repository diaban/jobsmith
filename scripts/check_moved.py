"""Prove that moving history out of CLAUDE.md dropped nothing.

    python scripts/check_moved.py [OLD_REF] [--condensed FILE]

Splits ``CLAUDE.md`` as it was at OLD_REF (default ``main``) into blocks — a
heading, a paragraph, one bullet, a table, a fenced block — and checks that
each one appears, whitespace- and list-marker-normalized, somewhere in the
current ``CLAUDE.md`` or ``docs/decisions/*.md``. Every ``#NN`` cited by the
old file must be cited by the new set too.

A block deliberately condensed into a rule rather than moved is listed in the
``--condensed`` file, one ``<first 60 characters of the block> -> <destination>``
per line; it is then reported, not failed.

Used by the migration of #102 and by the ``scribe`` agent whenever it moves
history out of CLAUDE.md to hold the budget. Stdlib only; exits 1 on a loss.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = re.compile(r"^\s*(?:[-*] |\d+\. )", re.M)


def blocks(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    fence = False
    for line in text.split("\n"):
        stripped = line.strip()
        if fence:
            cur.append(line)
            if stripped.startswith("```"):
                fence = False
                out.append("\n".join(cur))
                cur = []
            continue
        if stripped.startswith("```"):
            if cur:
                out.append("\n".join(cur))
            cur, fence = [line], True
            continue
        if not stripped:
            if cur:
                out.append("\n".join(cur))
            cur = []
            continue
        starts = re.match(r"^(#|\s*- |\s*\d+\. |\|)", line)
        if starts and not (line.startswith("|") and cur and cur[0].startswith("|")):
            if cur:
                out.append("\n".join(cur))
            cur = [line]
            continue
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return [b for b in out if b.strip()]


def normalize(text: str) -> str:
    text = MARKER.sub(" ", text)
    text = re.sub(r"^#+\s*", " ", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def main(argv: list[str]) -> int:
    ref = next((a for a in argv if not a.startswith("--")), "main")
    condensed: dict[str, str] = {}
    if "--condensed" in argv:
        path = Path(argv[argv.index("--condensed") + 1])
        for line in path.read_text().splitlines():
            if "->" in line:
                head, dest = line.split("->", 1)
                condensed[normalize(head)] = dest.strip()
    old = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:CLAUDE.md"],
        capture_output=True, text=True, check=True,
    ).stdout
    new = [(ROOT / "CLAUDE.md").read_text()]
    new += [p.read_text() for p in sorted((ROOT / "docs" / "decisions").glob("*.md"))]
    haystack = normalize("\n\n".join(new))

    old_blocks = blocks(old)
    missing, reported = [], []
    for block in old_blocks:
        needle = normalize(block)
        if needle in haystack:
            continue
        dest = next((d for h, d in condensed.items() if needle.startswith(h)), None)
        (reported if dest else missing).append((needle[:90], dest))

    refs_old = set(re.findall(r"#\d+\b", old))
    refs_new = set(re.findall(r"#\d+\b", "\n".join(new)))
    lost_refs = sorted(refs_old - refs_new, key=lambda r: int(r[1:]))

    moved = len(old_blocks) - len(missing) - len(reported)
    print(f"{len(old_blocks)} blocks in CLAUDE.md@{ref}: {moved} found verbatim, "
          f"{len(reported)} condensed, {len(missing)} missing")
    print(f"{len(refs_old)} issue references: {len(lost_refs)} lost")
    for needle, dest in reported:
        print(f"  condensed: {needle!r} -> {dest}")
    for needle, _ in missing:
        print(f"  MISSING:   {needle!r}")
    for ref_ in lost_refs:
        print(f"  LOST REF:  {ref_}")
    return 1 if missing or lost_refs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
