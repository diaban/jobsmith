# 0133 — The PR protocol runs as commands, not by hand

- **Issue:** #133 · **PR:** #135 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "The protocol runs as commands, not by hand" (Working on this repo)

**Context.** On 2026-09-26 most of the tokens and time went to steps Claude did by hand: probe scripts written again for each issue (4 that day), CI watched with polling loops, two PRs combined in a worktree made by hand, the index and `→ NNNN` checked by reading. Each step printed long output.

**Decision.** Each step becomes a command that prints a short result:
- `make probe`: versioned case sets per node; `main` and the branch run in parallel.
- `make combo PRS="a b"`: several open PRs merged onto `main`, then the tests and the structural tier.
- `make hooks`: a git pre-commit hook running ruff and `uv lock --check`.
- `make check`: lint, types, the leak gate and the tests in parallel, with `pytest -n auto`, also in CI.
- Repo auto-merge (`gh pr merge --auto --squash`), plus `tests/test_decision_records.py`.
Plain bash wherever Python or make would only add weight. Mutation testing, which replaces falsifying by hand, is #134.

| measured | before | after |
|---|---|---|
| full suite, local (16 cores) | ~86 s | ~17 s (2 runs, 713 passed) |
| probe before/after | one script per issue | one command, 10-line table |
