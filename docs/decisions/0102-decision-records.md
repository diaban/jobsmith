# 0102 — Decisions are recorded in `docs/decisions/`, and `CLAUDE.md` keeps the rules

- **Issue:** #102 · **PR:** #103
- **Status:** accepted
- **Reconstructed** by the scribe from PR #103 on 2026-09-24 — not written by the agent that took the decision. Built only from PR #103's body, issue #102's body and the diff; no measurement below was taken by the scribe.

## Context

`CLAUDE.md` went from 59k characters to 158k (~40k tokens) over roughly 25 PRs.
Every agent loads it at the start of a session and carries it through every
turn, so every delegated task pays for it on every tool call — measured on
this repo's own agents: **10 minutes for #76, 38 minutes for #96**, the
instruction file being part of the reason. It grows because it plays two roles
at once: the **rules** an agent must not break (`reportTypedDictNotRequiredAccess`
is on; a capability nothing can serve stays out of the registry; `build()`
must use `self.state_graph(...)`; blank the keys for the deterministic eval
tier) and the **history** of each decision (what was observed, the options,
the numbers, why this one) — valuable, but read only by whoever touches that
area, and every PR was adding to the rules file because the convention said a
PR documents its reasoning there. The file also went stale in ways nobody was
responsible for catching — **three times in one session**: the *About this
job* line after #85, the notice-placement bullet after langchain-anthropic
1.7.3, and #74's argument against a file that #85 had already emptied.

## Decision

**Decisions are recorded as they are taken, in `docs/decisions/`, and
`CLAUDE.md` keeps the rules.**

- `docs/decisions/NNNN-<slug>.md`, numbered by issue, one per decision:
  context, the decision, the alternatives and why not, what was measured,
  consequences, and what it supersedes. A `README.md` index, one line each.
- **The agent that implements an issue writes its decision record** in the
  same PR — it is the one that measured, falsified and chose; reconstructing
  the *why* from a diff afterwards loses exactly that. `CLAUDE.md` gains a
  line only when the decision creates or changes a **rule**, and that line
  points at the record (`→ NNNN`).
- **A `scribe` agent** (`.claude/agents/scribe.md`) keeps the whole: it
  records a major decision nobody wrote down, keeps the index current, finds
  claims made stale by later work (in `CLAUDE.md`, the records and the
  README) and fixes or supersedes them, holds `CLAUDE.md` under a size
  budget, and writes a brief every few merges. It covers `main` since the
  last commit it briefed, so it can be run at any time and never covers the
  same ground twice.

**The migration** (this record's own PR, #103) split the then-current
`CLAUDE.md` into decision records plus a rules file: **move, don't rewrite**
(the narratives are measured history — numbers, runs, falsifications —
paraphrasing them is how facts get lost, so a record carries its original
text); **nothing silently dropped** (the PR carries a map from every section
of the old file to where it went — a rule kept in `CLAUDE.md`, or a record);
`CLAUDE.md` got a budget, stated at its own top, with the reason; and the
"Working on this repo" convention changed from "a PR documents its reasoning
in `CLAUDE.md`" to "a PR writes its decision record".

## Alternatives and why not

Not recorded — neither the issue nor the PR body names an alternative to
splitting rules from history that was considered and rejected.

## Measured

- `CLAUDE.md`: **173 817 characters before → 37 435 bytes after** (PR #103's
  own table; the issue's context cites ~158k/~174k as the size the problem
  was diagnosed at, ~10 minutes and ~38 minutes as the time cost on #76 and
  #96 respectively).
- `docs/decisions/`: 36 records + `README.md` (index) + `TEMPLATE.md`, **186k**
  characters total, after the migration.
- **Budget: 40 000 characters** (≈10k tokens) — "under a quarter" of the
  pre-migration size — chosen to fit the code map and the rules, not
  narrative, leaving roughly 2.5k of headroom (about fifteen rule lines)
  before the scribe has to move something out again. Enforced by
  `tests/test_claude_md_budget.py`; a second test checks that every `→ NNNN`
  pointer names a record that exists. **Falsified**: the budget test was
  shown to fail at a 30k budget.
- **No-loss check** (`scripts/check_moved.py`, kept in the repo, stdlib only):
  splits the old `CLAUDE.md` into blocks (heading, paragraph, bullet, table,
  fence), normalizes whitespace and list markers, and looks for each block
  verbatim in the new `CLAUDE.md` plus `docs/decisions/*.md`, then checks
  every `#NN` the old file cited. Result: `222 blocks in CLAUDE.md@8326b98:
  222 found verbatim, 0 condensed, 0 missing` and `39 issue references: 0
  lost`. **Falsified**: `0053-step-timestamps.md` was moved out of the tree
  on purpose; the check then reported 1 missing block and `#53` lost, exit 1;
  restoring the file made it pass again.
- `make check` was green at the time of the PR: lint, pyright, leak gate,
  **647 passed, 8 skipped**.

## Consequences

- The "Working on this repo" section of `CLAUDE.md` now says a PR writes its
  decision record, and `CLAUDE.md` gains a line only when a rule is created
  or changed — never narrative.
- Every record migrated in this PR carries a `**Source:** migrated verbatim
  from CLAUDE.md at 8326b98 (#102)` header line rather than being written
  fresh, and a small number of stale claims were found while moving them and
  deliberately left for the scribe's first run to fix or supersede (see
  `0000-foundations.md`, `0010-cross-process-ownership.md` and
  `0035-capability-artifacts-and-slide-deck.md` for what was found; the
  scribe's 2026-09-24 run resolved them).
- Open, not solved by this PR: `#100` (cross-process job events) and other
  issues cited without a record of their own are listed in
  `docs/decisions/README.md`'s "Issues cited without a record of their own"
  section rather than invented a number for.
- This record itself did not exist until the scribe's first run
  (2026-09-24) reconstructed it — PR #103 did not write a `0102-*.md` file
  for its own decision, only the process and layout it put in place for
  every decision after it.

## Supersedes / superseded by

None.
