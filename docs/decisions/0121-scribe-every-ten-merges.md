# 0121 — The scribe runs every ~10 merges, not every ~3

- **Issue:** none (asked directly) · **PR:** #121
- **Status:** accepted
- **Rule in `CLAUDE.md`:** "Run the `scribe` agent … every ~10 merges, or when this file nears its budget"

## Context

The scribe (0102) was to run every ~3 merges and after each batch of parallel
PRs. Each pass is a PR of its own, and `main` is protected with strict checks,
so each pass pays a full CI round (~2 min per Python version) plus, whenever
other PRs are in flight, a rebase and a re-run for each of them. On
2026-09-25 four PRs (#117–#120) landed in one session; at ~3 the scribe would
have interleaved at least one extra CI-gated PR into that sequence.

## Decision

Every ~10 merges, or when `CLAUDE.md` nears its 40k budget. The "after a batch
of parallel PRs" trigger is dropped: it re-introduced the frequency this
change removes.

## Alternatives and why not

- **Keep ~3.** The cost is paid on every pass; what it buys — records and
  briefs a few merges fresher — is not needed that often, because each PR
  already writes its own record and CLAUDE.md line (0102). The scribe
  catches what slipped through, which does not accumulate that fast.
- **Run it without a PR.** `main` is protected; everything lands through a PR.

## Measured

Not measured beyond the CI round per PR (~2 min, both versions in parallel)
and the rebase-and-rerun cost on strict checks described in the workflow.

## Consequences

Briefs cover longer ranges (~10 merges). The budget trigger is unchanged and
still enforced by `tests/test_claude_md_budget.py`, so a CLAUDE.md that grows
between passes fails CI rather than going unnoticed.

## Supersedes / superseded by

Partially supersedes the cadence stated with [0102](0102-decision-records.md).
