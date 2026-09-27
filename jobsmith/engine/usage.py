"""What a run spent, and which step spent it.

An LLM call is the only thing in this framework that costs money, and the
number was invisible: both real adapters already receive `usage` from their
SDKs and dropped it on the floor.

Three pieces, deliberately small:

    Usage           an immutable tally (tokens, calls, estimated cost)
    UsageLedger     per-scope accumulation for ONE run
    record_usage()  what an LLM adapter calls after each response

**Why a ledger and not a return value.** `LLMClient.chat` returns `str`, and
every capability, planner and generator is written against that. Widening it
to `(str, Usage)` would touch every call site in the project (and every
third-party capability) to thread a number that most of them do not care
about. Accounting is *ambient*: it belongs to the run, not to the call. So the
ledger travels in a `ContextVar` that the JobManager installs around the run,
adapters push into it, and the pieces that DO care (the capability's own
`meta`, the Job, the report) read it back.

**Attribution comes from the run, not from the caller.** A capability nobody
wrote for this feature — a third-party one, or a framework node like the
planner — must still be attributed correctly, so the scope is read from
LangGraph's runtime config rather than passed down: the root segment of the
checkpoint namespace IS the parent graph's node name (`planner`, or
`cap_research|research_notes:...` for a capability sub-graph). No node
signature changes, nothing to remember when writing a capability. If that
lookup ever fails, spend lands under `UNATTRIBUTED` — the total stays right
and only the breakdown degrades.

Not covered: the chat layer talks to LangChain models directly (the
deliberate two-stack split), so conversation tokens are not counted here.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

CAP_NODE_PREFIX = "cap_"          # parent-graph node name for a capability
UNATTRIBUTED = "unattributed"     # spend whose scope could not be determined


# ---------------------------------------------------------------- the tally

@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens and estimated cost of one or more LLM calls.

    `input_tokens` is *uncached* input: each adapter normalizes its provider's
    shape (Anthropic reports cache reads separately, OpenAI reports them as a
    subset of the prompt tokens) so the two are always disjoint here.

    `cost_usd` is None when no call in the tally used a model with a known
    price; when only some are priced it is the sum of what could be priced —
    an estimate, never a bill.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    calls: int = 0
    cost_usd: float | None = None
    models: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cached_input_tokens

    def __bool__(self) -> bool:
        return self.calls > 0

    def __add__(self, other: Usage) -> Usage:
        if not isinstance(other, Usage):
            return NotImplemented
        cost = (
            None if self.cost_usd is None and other.cost_usd is None
            else (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        )
        models = list(self.models)
        models += [m for m in other.models if m not in models]
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            calls=self.calls + other.calls,
            cost_usd=cost,
            models=tuple(models),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form — this is what lands in `meta` and in the store."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "models": list(self.models),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Usage:
        data = data or {}
        return cls(
            input_tokens=int(data.get("input_tokens") or 0),
            output_tokens=int(data.get("output_tokens") or 0),
            cached_input_tokens=int(data.get("cached_input_tokens") or 0),
            calls=int(data.get("calls") or 0),
            cost_usd=data.get("cost_usd"),
            models=tuple(data.get("models") or ()),
        )


# ---------------------------------------------------------------- prices

@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per MILLION tokens. `cached_input` defaults to a 90% discount."""

    input: float
    output: float
    cached_input: float | None = None

    def cost(self, usage: Usage) -> float:
        cached_rate = self.cached_input if self.cached_input is not None else self.input * 0.1
        return (
            usage.input_tokens * self.input
            + usage.cached_input_tokens * cached_rate
            + usage.output_tokens * self.output
        ) / 1_000_000


# A SNAPSHOT (checked 2026-09), not permanent truth: vendors reprice, and a
# stale table silently produces a wrong number. Override it without touching
# code, via $JOBSMITH_PRICES — either inline JSON or a path to a JSON file:
#
#   JOBSMITH_PRICES='{"gpt-5.1": {"input": 1.25, "output": 10.0}}'
#   JOBSMITH_PRICES=./prices.json
#
# Entries are matched by longest prefix, so a dated snapshot of a model
# (claude-opus-5-20260101) is priced by its family's row. A model with no row
# is reported in tokens with cost_usd = None: no price is better than a
# made-up one. OpenAI models are deliberately absent — nobody here has a
# reliable current figure for them; add yours through $JOBSMITH_PRICES.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4": ModelPrice(5.0, 25.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
    "claude-fable-5": ModelPrice(10.0, 50.0),
    # The keyless fake bills like Haiku, so a `--llm=fake` run (and CI) walks
    # the whole cost path instead of leaving it untested until a real key shows up.
    "fake-keyword-llm": ModelPrice(1.0, 5.0),
}

_price_overrides: dict[str, ModelPrice] | None = None


def _load_overrides() -> dict[str, ModelPrice]:
    """Parse $JOBSMITH_PRICES once (inline JSON, or a path to a JSON file)."""
    global _price_overrides
    if _price_overrides is not None:
        return _price_overrides
    raw = (os.environ.get("JOBSMITH_PRICES") or "").strip()
    prices: dict[str, ModelPrice] = {}
    if raw:
        try:
            if not raw.startswith("{"):
                with open(raw) as f:
                    raw = f.read()
            for model, entry in json.loads(raw).items():
                prices[model] = ModelPrice(
                    float(entry["input"]),
                    float(entry["output"]),
                    entry.get("cached_input"),
                )
        except (OSError, ValueError, KeyError, TypeError):
            prices = {}      # a broken table must not break a run
    _price_overrides = prices
    return prices


def reset_price_overrides() -> None:
    """Forget the parsed $JOBSMITH_PRICES (tests, and anything that sets it late)."""
    global _price_overrides
    _price_overrides = None


def price_for(model: str) -> ModelPrice | None:
    """Longest-prefix price lookup; overrides win over the built-in table."""
    table = {**DEFAULT_PRICES, **_load_overrides()}
    match = max((k for k in table if model.startswith(k)), key=len, default=None)
    return table[match] if match else None


def estimate_cost(model: str, usage: Usage) -> float | None:
    price = price_for(model)
    return price.cost(usage) if price else None


# ---------------------------------------------------------------- the ledger

@dataclass
class UsageLedger:
    """Per-scope accumulation for one run. Read it, don't guess from logs."""

    _by_scope: dict[str, Usage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, scope: str, usage: Usage) -> None:
        with self._lock:   # capability waves are parallel; sync nodes may be threaded
            self._by_scope[scope] = self._by_scope.get(scope, Usage()) + usage

    def get(self, scope: str) -> Usage:
        with self._lock:
            return self._by_scope.get(scope, Usage())

    def by_scope(self) -> dict[str, Usage]:
        with self._lock:
            return dict(self._by_scope)

    def total(self) -> Usage:
        with self._lock:
            total = Usage()
            for usage in self._by_scope.values():
                total = total + usage
            return total


_current_ledger: ContextVar[UsageLedger | None] = ContextVar(
    "jobsmith_usage_ledger", default=None
)


@contextmanager
def usage_ledger(ledger: UsageLedger | None = None) -> Iterator[UsageLedger]:
    """Install a ledger for everything that runs inside this block.

    A fresh ledger by default, so a nested run can never bill its parent.
    Tasks spawned inside inherit the *object* (contexts are copied on task
    creation, but the ledger mutates in place), which is exactly what makes
    parallel capability branches add up.
    """
    ledger = ledger if ledger is not None else UsageLedger()
    token = _current_ledger.set(ledger)
    try:
        yield ledger
    finally:
        _current_ledger.reset(token)


def current_ledger() -> UsageLedger | None:
    return _current_ledger.get()


def current_scope() -> str:
    """Which graph step is spending, read from LangGraph's runtime config.

    `checkpoint_ns` is `<node>:<uuid>` in the parent graph and
    `cap_<name>:<uuid>|<inner node>:<uuid>` inside a capability sub-graph, so
    its root segment names the responsible step. Outside a graph (a direct
    client call, a unit test) there is nothing to attribute to.
    """
    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:                     # not inside a graph node
        return UNATTRIBUTED
    namespace = (config.get("configurable") or {}).get("checkpoint_ns") or ""
    root = namespace.split("|", 1)[0].split(":", 1)[0]
    if not root:
        root = (config.get("metadata") or {}).get("langgraph_node") or ""
    if not root:
        return UNATTRIBUTED
    return root[len(CAP_NODE_PREFIX):] if root.startswith(CAP_NODE_PREFIX) else root


def record_usage(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
    scope: str | None = None,
) -> Usage:
    """Book one LLM call. Called by adapters; a no-op with no ledger installed.

    Returns the tally so a caller that wants the number for itself can have it
    without reaching into the ledger.
    """
    usage = Usage(
        input_tokens=max(int(input_tokens), 0),
        output_tokens=max(int(output_tokens), 0),
        cached_input_tokens=max(int(cached_input_tokens), 0),
        calls=1,
        models=(model,) if model else (),
    )
    usage = replace(usage, cost_usd=estimate_cost(model, usage)) if model else usage
    ledger = _current_ledger.get()
    if ledger is not None:
        ledger.add(scope or current_scope(), usage)
    return usage


__all__ = [
    "CAP_NODE_PREFIX",
    "DEFAULT_PRICES",
    "UNATTRIBUTED",
    "ModelPrice",
    "Usage",
    "UsageLedger",
    "current_ledger",
    "current_scope",
    "estimate_cost",
    "price_for",
    "record_usage",
    "reset_price_overrides",
    "usage_ledger",
]
