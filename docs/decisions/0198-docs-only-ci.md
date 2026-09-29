# 0198 — A docs-only pull request runs only the docs tests, under the same check names

- **Issue:** #198 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "A PR touching only `docs/**` and top-level `*.md` runs only the two docs tests" (Working on this repo, CI)

**Context.** Every PR paid the whole `check` job on both Pythons, a scribe pass included (0121). On #199, all Markdown, `check` took 65 s (3.11) and 69 s (3.12): the suite ~40 s, pyright 8 s, install 3 s, the Postgres container 7 s. Such a diff can break only `tests/test_claude_md_budget.py` and `tests/test_decision_records.py`; no other code or test reads `docs/` or a top-level `.md` (grep). `check (3.11)`, `check (3.12)` and `lockfile` are required, so the job must still run and report under those names.

**Decision.** A `Scope` step first in `check` diffs a PR's merge commit against its first parent (`fetch-depth: 2`, `git diff --no-renames --name-only HEAD^1 HEAD`). When every path is under `docs/` or a top-level `*.md`, the job builds a bare venv (pytest, pytest-asyncio for pyproject's `asyncio_mode`), runs the two tests with `--noconftest` (the conftest imports the project) and skips install, lint, pyright, the leak gate and the suite. The step fails toward the full path: a push, an empty diff, a failed `git diff`, a quoted or nested path. `--no-renames`, because a file moved into `docs/` is otherwise listed only where it arrived. `lockfile` is unchanged.

**Alternatives and why not.**
- `paths-ignore` on the workflow: the required checks are never reported, and the PR waits for them forever.
- A `changes` job ahead of `check`, which could also skip the Postgres service (an empty `image`): if `changes` fails, `check` is skipped, and GitHub counts a skipped required job as passed. It also adds a runner start in sequence to save 7 s.
- `dorny/paths-filter` or the PR files API: a third-party action or a token, for a ten-line `case`.
- Top-level `*.md` only, not `**/*.md`: nested Markdown could one day be a fixture or a prompt; `.claude/agents/scribe.md` runs the full path.

**Measured.** Locally: the scope script gives `true` on #199 (d1d032a) and #193 (3462465), `false` on #197 (42bb050), and `false` on a push, an empty diff, a code file moved into `docs/`, a nested `.md`, a top-level `Makefile`, a non-ASCII name (quoted by git) and a commit with no parent. The fast path in a venv without `langgraph`: 86 passed in 0.09 s. Falsified: a dangling `→ 9999` in `CLAUDE.md` fails 2 of its tests.

**On CI.** Full path, #200 (workflow + docs): `Scope` listed `.github/workflows/ci.yml` and said `false`; every step ran, `Docs tests` skipped; `check` 79 s (3.11), 73 s (3.12). Fast path, #201 (this record only): `Scope` said `true`; install, lint, pyright, leak gate and suite skipped, 88 passed; `check` 21 s (3.11), 18 s (3.12), against 65/69 s for #199. Of those, starting the Postgres container is 9–11 s: the part a `changes` job would save, left as it is (Alternatives).
