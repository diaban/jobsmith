"""The contract a graph signs with the job engine (docs/design/core-v1.md).

A `GraphSpec` names a compiled LangGraph graph — compiled with the engine's
checkpointer, since resuming a job is re-entering its thread — and says how
what the graph returns becomes the job's result. That is the whole contract:
the input reaches the graph as the caller gave it, the facts it publishes are
recorded as they come (`facts.py`), and the rest is the graph's business.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class JobFailed(Exception):
    """Raised by a `GraphSpec.result` for a run that ended by DECLARING it failed.

    The job is FAILED with `reason` as its error, and keeps `result` — what the
    run returned is still the record of what it did. Nothing is left to run:
    the graph reached its end, so a resume refuses it.
    """

    def __init__(self, reason: str, *, result: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.result = result


def _as_is(output: Any) -> Any:
    return output


@dataclass(frozen=True)
class GraphSpec:
    name: str                                   # recorded on the job; a resume finds it by name
    graph: Any                                  # compiled with the engine's checkpointer
    # What `ainvoke` would have returned → the job's result (JSON). May raise
    # `JobFailed`. A graph whose output is its answer passes nothing.
    result: Callable[[Any], Any] = _as_is
