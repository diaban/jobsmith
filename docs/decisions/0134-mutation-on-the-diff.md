# 0134 — Mutation testing on the diff replaces falsifying by hand

- **Issue:** #134 · **PR:** #136 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "The protocol runs as commands" (`make mutate`)

**Context.** "Break the fix, watch the test fail, restore it" was done by hand: an edit, a run, a revert, for each test. It covered only the one break someone thought of, and it had already left files half-reverted.

**Decision.** `make mutate TESTS="tests/test_x.py" [BASE=main]` runs **cosmic-ray** on a scratch worktree of HEAD (cosmic-ray edits files in place). `cr-filter-git` keeps only the lines changed since `BASE`. Each mutant runs the targeted tests with `-x`. The command prints the counts and each line where a mutant survived. cosmic-ray was chosen over mutmut because it limits itself to the diff out of the box and takes an arbitrary test command.

| `evals/probe.py` since 4ec92ee, vs `tests/test_probe.py` | mutants | killed | survived |
|---|---|---|---|
| tests as merged in #135 | 37 | 8 | 29 |
| + 2 tests for `--compare` | 37 | 18 | 19 (5 lines) |

**Limits.**
- About 2 s per mutant: sequential, so it is fast only on a small diff.
- A mutant inside a type annotation always survives, because annotations are never evaluated. Read the survivors; do not chase 0.
