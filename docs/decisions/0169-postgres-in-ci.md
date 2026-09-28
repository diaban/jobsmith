# 0169 — CI runs the Postgres paths, and Postgres setup takes a tried advisory lock

- **Issue:** #169 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "Postgres included" (Working on this repo, CI) and "Postgres setup runs under a *tried* advisory lock" (persistence)

**Context.** CI installed `.[postgres]` with no server, so 8 tests gated on `$JOBSMITH_TEST_PG` were skipped: history at scale (0141), delivery per backend, LISTEN/NOTIFY events (0138). The core split's Postgres paths had never run (0161, "Measured"). The first local run on a new database (`postgres:16-alpine`, `-n auto`) failed 4 tests and errored 2 of 7: every worker's `setup()` reads the migration version, then applies what is missing with nothing held, and they collided (`UniqueViolation` on `checkpoint_migrations`). Not a test artefact: two processes' first start on a new database do the same. A CI service container is new on every run.

**Decision.** The `check` job gets a `postgres:16-alpine` service and `JOBSMITH_TEST_PG` on its test step, so the gated tests run on both Python versions and block the merge through the checks already required. `_open_postgres` runs both setups while holding a session advisory lock on a connection of its own (closing it releases the lock whatever happens), taken with `pg_try_advisory_lock` in a loop, never `pg_advisory_lock`.

**Alternatives and why not.**
- A separate CI job: one more required check to add to the branch protection, for the same tests on one Python version.
- Pre-creating the schema in a CI step: green CI, and the product race left in place.
- Retrying on `UniqueViolation`, as SQLite does (`_while_another_process_sets_up`): SQLite offers no lock, Postgres does; and two `CREATE INDEX CONCURRENTLY` on one table can leave an invalid index.
- A waited `pg_advisory_lock`: measured, it hangs forever. The migrations' `CREATE INDEX CONCURRENTLY` waits for every statement in flight, the other processes' blocked lock calls are such statements, and Postgres does not detect it.

**Measured.** `test_apps_opening_a_fresh_postgres_together_all_get_it` (5 apps on a new database): fails 5/5 without the lock, passes 10/10 with it. The full suite on a new container with `-n auto`: 836 passed, 3 runs out of 3; without the DSN, 828 passed and 8 skipped.

**Consequences.** Postgres is exercised on every PR. A local run needs a server: `docker run -d -e POSTGRES_PASSWORD=pg -p 55432:5432 postgres:16-alpine` and `JOBSMITH_TEST_PG=postgresql://postgres:pg@localhost:55432/postgres`.
