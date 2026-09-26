"""The decision records stay reachable and short, checked instead of reviewed.

Every record has its row in the index, every `→ NNNN` in CLAUDE.md names a
record that exists, and a record written under the effort budget stays within
it (→ 0102, 0130).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "docs" / "decisions"
RECORDS = sorted(p for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))
#: Records from this number on are written under 0130's budget.
BUDGETED_FROM = 130
MAX_LINES = 20


def _number(path: Path) -> str:
    return path.name[:4]


@pytest.mark.parametrize("record", RECORDS, ids=_number)
def test_every_record_has_its_row_in_the_index(record):
    index = (DECISIONS / "README.md").read_text(encoding="utf-8")
    assert f"]({record.name})" in index, f"{record.name} has no row in docs/decisions/README.md"


def test_every_arrow_in_claude_md_names_a_record_that_exists():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    cited = {n for refs in re.findall(r"→ ((?:\d{4}(?:, )?)+)", text)
             for n in re.findall(r"\d{4}", refs)}
    missing = sorted(cited - {_number(r) for r in RECORDS})
    assert not missing, f"CLAUDE.md cites records that do not exist: {missing}"


@pytest.mark.parametrize(
    "record", [r for r in RECORDS if int(_number(r)) >= BUDGETED_FROM], ids=_number)
def test_a_record_under_the_budget_is_at_most_twenty_lines(record):
    lines = record.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) <= MAX_LINES, f"{record.name}: {len(lines)} lines (budget {MAX_LINES}, → 0130)"
