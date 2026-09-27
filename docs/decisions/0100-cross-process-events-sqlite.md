# 0100 — Job events cross processes on a SQLite file, by watching `data_version`

- **Issue:** #100 · **PR:** #NN · **Status:** accepted (Postgres: #138)
- **Rule in `CLAUDE.md`:** "Events cross processes on a SQLite file"

**Context.** `subscribe()` (the TUI's repaint, `/events`) only fanned out what its own process persisted. A job run by `jobsmith chat` did not show up in a `jobsmith ui` on the same database until F5. `BaseStore` cannot answer "what changed anywhere" without rereading the whole index.

**Decision.** `SqliteWatchEvents` extends `InProcessEvents`:
- While at least one queue is subscribed, one task reads `PRAGMA data_version` every second, on its **own** read-only connection (never the saver's or the store's).
- Only when that value moves does it re-read the index summaries, and it publishes each job whose `updated_at` changed.
- Jobs that existed at subscription time are not announced.
- `build_app` picks it for a SQLite file. Memory keeps `InProcessEvents` and polls nothing.

**Alternatives.** Polling the index on every tick: O(jobs) per tick, forever. A file watcher on the WAL (inotify): platform-specific, and a new dependency.

**Measured.** `tests/test_events.py`: two apps on one file; the subscriber hears the other one's job (~1 s), not the jobs that already existed, and still hears it after another subscriber leaves. `make mutate`: 110 mutants on the diff, 62 → 68 killed once those last two tests were added. The survivors are annotations, constants (interval, limit) and the `sqlite3.Error` path.
