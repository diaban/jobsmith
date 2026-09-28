"""Where a job's ending goes: its return address, and who delivers to it.

A job carries `reply_to`, plain JSON — `{"kind": …, …}` — never a callable,
so it survives a restart and crosses processes. When the job settles, the
deliverer registered for its kind is asked to deliver it; a True answer
stamps `delivered_at`. A resume clears the stamp: a job picked back up is
news again. The engine promises **at least once, keyed by job id**;
deduplicating is the receiver's business (docs/design/core-v1.md, "Delivery").

Two ways a kind can be delivered, both here:

- `Nobody` (`none`), the default: nobody waits, so a job is delivered as it
  settles, and nothing is ever pending;
- `Pulled(kind)`: the receiver pulls — `JobManager.pending_deliveries(address)`
  lists what settled and was not delivered, and `mark_delivered(job_id)` says
  it was, once the receiver has it. The engine gives such a kind its name and
  nothing else; what the address means is the registrant's.

A third way is pushed (#166): `Pushed` is a kind the engine hands the job to
itself — a webhook (`Webhook`), a queue — and never on the persist path. The
ending is saved first; the push runs in a task of its own and is retried with
a capped exponential backoff until it succeeds; only then is `delivered_at`
stamped. Until then the record says it is pending, so a process that stops
leaves it to the next start (`JobManager.recover_interrupted`) and nothing
is lost. Two processes may both push one ending: at least once, by job id.

Every address has a flat `key` the store can filter on (`pull:A1`): a nested
filter fails on SQLite, a dotted one finds nothing in memory (measured,
core-v1.md), and a flat string matches on every backend.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .models import Job

NONE = "none"


def nobody() -> dict[str, Any]:
    """The address of a job nobody waits on."""
    return {"kind": NONE}


class Deliverer(Protocol):
    kind: str

    def key(self, reply_to: dict[str, Any]) -> str:
        """The flat key the store indexes this address by; raises
        `ValueError` for an address this kind cannot read."""
        ...

    async def deliver(self, job: Job) -> bool:
        """Hand a settled job to its address; True when it is delivered now."""
        ...


class Nobody:
    """`none`: delivered as it settles, since nobody is waiting."""

    kind = NONE

    def key(self, reply_to: dict[str, Any]) -> str:
        return NONE

    async def deliver(self, job: Job) -> bool:
        return True


class Pulled:
    """A kind whose receiver pulls: `{"kind": kind, "id": …}`, pending until
    `mark_delivered`."""

    def __init__(self, kind: str):
        self.kind = kind

    def key(self, reply_to: dict[str, Any]) -> str:
        receiver = reply_to.get("id")
        if not isinstance(receiver, str) or not receiver:
            raise ValueError(f"a {self.kind!r} address needs an \"id\": {reply_to!r}")
        return f"{self.kind}:{receiver}"

    async def deliver(self, job: Job) -> bool:
        return False


class Pushed:
    """A kind the engine pushes to, off the persist path (see the module).

    `push(job)` hands one ending over and returns once the receiver has it;
    it raises when it could not, and is then called again after `first_retry`
    seconds, doubled each time up to `max_retry` — for as long as this
    process lives, and from the next start after that. What it is given is
    the record's summary with its id; `(job_id, attempt)` is what a receiver
    deduplicates on, since a resumed job ends again.
    """

    kind: str = "push"
    first_retry: float = 1.0
    max_retry: float = 60.0

    def key(self, reply_to: dict[str, Any]) -> str:
        raise NotImplementedError

    async def deliver(self, job: Job) -> bool:
        return False                        # never inline: `JobManager` pushes

    async def push(self, job: Job) -> None:
        raise NotImplementedError

    def delays(self):
        """The waits between attempts: doubling, then `max_retry` for ever."""
        delay = self.first_retry
        while True:
            yield delay
            delay = min(delay * 2, self.max_retry)


class Webhook(Pushed):
    """`{"kind": "webhook", "url": …}`: the ending is POSTed there as JSON, and
    any 2xx is delivered. Only to a URL under one of `allowed` — the prefixes
    this deployment declared it may reach — since the address comes from
    whoever launched the job. `client` is an `httpx.AsyncClient` (`.[api]`),
    made on first use when not given."""

    kind = "webhook"

    def __init__(self, allowed: Sequence[str], *, client: Any = None, timeout: float = 10.0):
        if not allowed:
            raise ValueError("a webhook deliverer needs the URL prefixes it may reach")
        self.allowed = tuple(allowed)
        self._client = client
        self.timeout = timeout

    def key(self, reply_to: dict[str, Any]) -> str:
        url = reply_to.get("url")
        if not isinstance(url, str) or not url.startswith(self.allowed):
            raise ValueError(f"a webhook address needs a \"url\" under {list(self.allowed)}: "
                             f"{reply_to!r}")
        return f"{self.kind}:{url}"

    async def push(self, job: Job) -> None:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.timeout)
        response = await self._client.post(
            job.reply_to["url"], json=job.summary() | {"job_id": job.job_id},
            headers={"Idempotency-Key": f"{job.job_id}:{job.attempt}"})
        response.raise_for_status()


__all__ = ["NONE", "Deliverer", "Nobody", "Pulled", "Pushed", "Webhook", "nobody"]
