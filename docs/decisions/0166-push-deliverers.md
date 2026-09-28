# 0166 — A pushed return address is delivered off the persist path, retried until it lands

- **Issue:** #166 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none yet (the budget pass adds it, under Delivery: a `Pushed` kind is never called inline; saved, then pushed by its own task, backoff, stamped after; pending pushes resume at startup)

**Context.** `JobManager._persist_summary` asked the deliverer inline at every ending. That suits `none` and a pulled kind (0161), not a push (webhook, queue): a slow push would delay the persist, one that raised would break the settlement, and a failed one was never retried.

**Decision.** `engine/delivery.py` gains `Pushed`: `push(job)` returns once the receiver has the ending and raises when it could not. `_persist_summary` never calls it: it saves the ending, then starts a task (`_push_later`, one per `job_id#attempt`, in a fresh context so it carries nothing of the run). The task retries with a doubling delay capped at `max_retry`, for as long as the process lives. `delivered_at` is stamped only after a push landed, and only while the record is still that attempt: a job resumed meanwhile has its own push. At startup `recover_interrupted` pushes again every settled, unstamped job of a pushed kind. `Webhook(allowed)` is the first such kind: a JSON POST of the summary plus `job_id`, `Idempotency-Key: job_id:attempt`, any 2xx is delivered, and only to a URL under a prefix the deployment declared.

**Alternatives and why not.**
- An outbox table: the record already says what is pending (settled, `delivered_at` empty), and the repository is the one place that knows the schema.
- A bounded number of attempts per process: a daemon that runs for weeks would then leave an ending undelivered until it restarts.
- A webhook that accepts any URL: the address comes from whoever launched the job, so the daemon would POST wherever it is told.

**Measured.** `tests/test_delivery.py`, on memory, SQLite and Postgres: the ending is saved while the receiver is closed, then pushed after two refusals and stamped; a push left pending by a stopped manager lands from the next one's `recover_interrupted`. A webhook test, over `httpx.MockTransport`, checks the refused URL, the retried 500, the body and the key. Removing the startup sweep fails the restart test on all three backends. Removing the retry fails 5 tests.

**Consequences.** Two processes on one database may both push an ending, at startup or on a lease takeover: at least once, keyed by `(job_id, attempt)`. No push kind is registered by default: a deployment passes `deliverers=[Webhook([...])]`.
