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

Every address has a flat `key` the store can filter on (`pull:A1`): a nested
filter fails on SQLite, a dotted one finds nothing in memory (measured,
core-v1.md), and a flat string matches on every backend.
"""
from __future__ import annotations

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


__all__ = ["NONE", "Deliverer", "Nobody", "Pulled", "nobody"]
