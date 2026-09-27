# 0141 — Complete queries filter in the store; human listings cut after ordering

- **Issue:** #141 · **PR:** #142 · **Status:** accepted
- **Rule in `CLAUDE.md`:** "`load_all` is complete" (Jobs layer)

**Context.** Found by an external review and reproduced here. `load_all(limit=…)` applied its limit to the whole index before any filter, and stores return rows oldest first. Past ~100 jobs, the newest disappeared from announcements (0006), `cancel_job`/`job_status`, events (0100), progress notices, listings and orphan recovery. The whole suite ran in memory with a handful of jobs, so nothing saw it.

**Decision.**
- `JobRepository.load_all(session_id=, status=, announced=, updated_since=)` is **complete**: the store applies the filters, and nothing is cut after them. It runs as one query, never pages, because Postgres orders a search by prefix only and paging by offset could skip or repeat rows.
- `JobManager.list_jobs` sorts newest first, **then** cuts. A caller that must see everything passes `limit=None`, never a bigger number.
- Announcements read `announced=False`; the in-flight notice reads by status.
- `SqliteWatchEvents` re-reads only what moved since its last look (`updated_since`).

| `tests/test_job_history.py` (250 jobs elsewhere, then mine) | main | branch |
|---|---|---|
| memory | fails | passes |
| SQLite | fails | passes |
| Postgres (`$JOBSMITH_TEST_PG`) | passes (by chance: unordered) | passes |

**Left out.** `resolve_job` (on the port, also over HTTP) matches a prefix among the 100 newest listed, which are the ids `jobsmith jobs` prints. A full id always resolves.
