"""CLAUDE.md stays under its size budget (#102).

Every session loads CLAUDE.md and carries it through every turn, so its size
is paid on every tool call of every agent. It grew from 59k to 174k characters
in ~25 PRs because each PR added its reasoning there; a budget nobody checks is
how that happened. The rules stay in CLAUDE.md, the history of how they were
decided goes to ``docs/decisions/`` — see the budget paragraph at its top.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = 40_000


def test_claude_md_is_under_its_budget():
    size = len((ROOT / "CLAUDE.md").read_text(encoding="utf-8"))
    assert size <= BUDGET, (
        f"CLAUDE.md is {size} characters, over its {BUDGET} budget. Move history "
        "(what was measured, the alternatives, why) into a decision record in "
        "docs/decisions/ and leave a one-line rule with `→ NNNN` here. Do not "
        "raise the budget."
    )


def test_every_rule_pointer_names_a_record():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    cited = {
        number
        for group in re.findall(r"→ ((?:\d{4}(?:, )?)+)", text)
        for number in group.split(", ")
        if number
    }
    records = {p.name[:4] for p in (ROOT / "docs" / "decisions").glob("[0-9][0-9][0-9][0-9]-*.md")}
    assert cited, "no `→ NNNN` pointer found: the pattern no longer matches CLAUDE.md"
    assert cited <= records, f"CLAUDE.md points at missing records: {sorted(cited - records)}"
