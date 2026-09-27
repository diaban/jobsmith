# 0138 — Job events cross processes on Postgres, by LISTEN/NOTIFY

- **Issue:** #138 · **PR:** #140 · **Status:** accepted (extends 0100)
- **Rule in `CLAUDE.md`:** "Events cross processes on a shared database"

**Context.** 0100 made `subscribe()` hear other processes on a SQLite file. On Postgres it still heard only its own.

**Decision.**
- `PostgresNotifyEvents`: every local publish also sends `pg_notify('jobsmith_jobs', job_id)`, fire-and-forget on one connection of its own.
- While someone is subscribed, one `LISTEN` connection loads each notified job and announces it if its `updated_at` moved. That is push, with no polling.
- Both connections are off the store's pool, so a long `LISTEN` never takes a pooled connection from a running job.
- `WatchedEvents` now holds what SQLite and Postgres share (subscription, "seen", no re-broadcast). A remote job is announced without `NOTIFY`ing again.

**Alternatives.** A trigger on LangGraph's `store` table: transactional, but it means DDL on a table we do not own. `LISTEN` on a pooled connection: it would hold one of the pool's 10 for as long as a UI is open.

**Measured.** `tests/test_events.py` runs the same cross-process cases on SQLite and on Postgres (`$JOBSMITH_TEST_PG`, here `postgres:16-alpine` in Docker): 7 passed; without the DSN, 3 skipped. `make mutate`: 93 mutants, 52 killed; the survivors are annotations, constants and the best-effort `except` paths.
