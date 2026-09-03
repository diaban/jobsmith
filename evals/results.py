"""Aggregating, storing and comparing a suite run.

A printed table answers "how is it now"; only a stored one answers "is it
better than before", which is the whole question a prompt change raises. So
every run is written to `evals/results/` as JSON, tagged with the agent, the
provider, the tier and the git revision, and the next run automatically diffs
itself against the most recent comparable one.

Comparability is on purpose narrow: a baseline is only picked up when the
agent, the provider and the tier match. Comparing a fake run against a Claude
run would produce a number that means nothing.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from .cases import EvalCase
from .harness import Observation
from .scoring import CHECK_NAMES, score, step_failure_rate

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class Tally:
    passed: int = 0
    applicable: int = 0

    @property
    def rate(self) -> float | None:
        return self.passed / self.applicable if self.applicable else None


@dataclass
class SuiteResult:
    #: UTC timestamp to the millisecond — two runs a second apart must not
    #: overwrite each other's record, and lexical order is chronological order
    run_id: str
    tier: str
    agent: str
    provider: str
    repeat: int
    registry: list[str]
    git_rev: str
    duration_s: float
    #: the case ids this run covered — a filtered run is not a baseline for a
    #: full one, so this is part of what "comparable" means
    cases: list[str] = field(default_factory=list)
    checks: dict[str, dict[str, int]] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    overall: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)

    # ---- derived ----

    @property
    def pass_rate(self) -> float | None:
        applicable = self.overall.get("applicable", 0)
        return self.overall["passed"] / applicable if applicable else None

    def check_rate(self, name: str) -> float | None:
        entry = self.checks.get(name)
        if not entry or not entry["applicable"]:
            return None
        return entry["passed"] / entry["applicable"]

    # ---- io ----

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SuiteResult:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def filename(self) -> str:
        return f"{self.run_id}-{self.tier}-{self.agent}-{self.provider}.json"


def summarize(
    cases: list[EvalCase],
    observations: list[Observation],
    *,
    tier: str,
    context: dict[str, Any],
    duration_s: float,
) -> SuiteResult:
    """Fold observations into the comparable record of one suite run."""
    by_id = {c.id: c for c in cases}
    tallies: dict[str, Tally] = {name: Tally() for name in CHECK_NAMES}
    failures: list[dict[str, Any]] = []

    for obs in observations:
        for check in score(by_id[obs.case_id], obs):
            tally = tallies.setdefault(check.name, Tally())
            if not check.applicable:
                continue
            tally.applicable += 1
            if check.passed:
                tally.passed += 1
            else:
                failures.append({
                    "case": obs.case_id,
                    "attempt": obs.attempt,
                    "check": check.name,
                    "detail": check.detail,
                })

    overall = Tally(
        passed=sum(t.passed for t in tallies.values()),
        applicable=sum(t.applicable for t in tallies.values()),
    )
    failure_rates = [r for r in (step_failure_rate(o) for o in observations) if r is not None]
    plan_sizes = [len(o.plan_steps) for o in observations if o.plan_steps]

    return SuiteResult(
        run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z",
        tier=tier,
        agent=str(context.get("agent", "?")),
        provider=str(context.get("provider", "?")),
        repeat=int(context.get("repeat", 1)),
        registry=list(context.get("registry", [])),
        git_rev=_git_rev(),
        duration_s=round(duration_s, 2),
        cases=sorted(c.id for c in cases),
        checks={name: asdict(t) for name, t in tallies.items()},
        metrics={
            "pass_rate": overall.rate,
            "step_failure_rate": fmean(failure_rates) if failure_rates else None,
            "mean_plan_steps": fmean(plan_sizes) if plan_sizes else None,
            "mean_run_seconds": fmean([o.duration_s for o in observations])
            if observations else None,
        },
        overall=asdict(overall),
        failures=failures,
        runs=[o.compact() for o in observations],
    )


# ---------------------------------------------------------------- storage

def write_result(result: SuiteResult, directory: Path | str = RESULTS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / result.filename
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_baseline(
    result: SuiteResult,
    directory: Path | str = RESULTS_DIR,
    *,
    exclude: Path | None = None,
) -> SuiteResult | None:
    """The most recent stored run comparable with this one.

    Comparable means: same tier, same agent, same provider, same cases. A fake
    run against a Claude run, or a `--case` slice against the full set, would
    produce a delta that means nothing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    pattern = f"*-{result.tier}-{result.agent}-{result.provider}.json"
    candidates = sorted(p for p in directory.glob(pattern) if p != exclude)
    for path in reversed(candidates):
        try:
            previous = SuiteResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if previous.cases and previous.cases != result.cases:
            continue          # a different slice of the golden set: not a baseline
        return previous
    return None


def load_result(path: Path | str) -> SuiteResult:
    return SuiteResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- rendering

def _pct(value: float | None) -> str:
    return "     —" if value is None else f"{value * 100:6.1f}%"


def _delta(now: float | None, before: float | None, *, points: bool = True) -> str:
    if now is None or before is None:
        return "      "
    diff = (now - before) * (100 if points else 1)
    text = f"{diff:+6.1f}" if points else f"{diff:+6.2f}"
    # A move too small to show is not a move: "-0.00" reads as a regression.
    return "     =" if float(text) == 0 else text


def render_summary(
    result: SuiteResult,
    baseline: SuiteResult | None = None,
    *,
    max_failures: int = 20,
) -> str:
    """The comparable, human-readable summary `make eval` prints."""
    lines = [
        f"eval · tier={result.tier} agent={result.agent} provider={result.provider} "
        f"repeat={result.repeat} rev={result.git_rev}",
        f"registry: {', '.join(result.registry) or '(none)'}",
        f"{len(result.runs)} run(s) in {result.duration_s}s",
        "",
    ]
    if baseline is not None:
        lines.append(f"baseline: {baseline.run_id} (rev {baseline.git_rev}) — Δ in points")
        lines.append("")

    header = f"{'check':<24}{'passed':>9}{'rate':>9}{'Δ':>7}"
    lines += [header, "-" * len(header)]
    for name in CHECK_NAMES:
        entry = result.checks.get(name, {"passed": 0, "applicable": 0})
        applicable = entry["applicable"]
        counts = f"{entry['passed']}/{applicable}" if applicable else "skipped"
        lines.append(
            f"{name:<24}{counts:>9}{_pct(result.check_rate(name)):>9}"
            f"{_delta(result.check_rate(name), baseline.check_rate(name) if baseline else None):>7}"
        )
    lines += ["-" * len(header)]
    total = f"{result.overall['passed']}/{result.overall['applicable']}"
    lines.append(
        f"{'overall':<24}{total:>9}{_pct(result.pass_rate):>9}"
        f"{_delta(result.pass_rate, baseline.pass_rate if baseline else None):>7}"
    )

    lines += ["", f"{'metric':<24}{'value':>9}{'Δ':>16}"]
    for key in ("step_failure_rate", "mean_plan_steps", "mean_run_seconds"):
        now = result.metrics.get(key)
        before = baseline.metrics.get(key) if baseline else None
        value = "        —" if now is None else f"{now:9.3f}"
        lines.append(f"{key:<24}{value}{_delta(now, before, points=False):>16}")

    if result.failures:
        lines += ["", f"failures ({len(result.failures)}):"]
        for failure in result.failures[:max_failures]:
            suffix = f" #{failure['attempt']}" if result.repeat > 1 else ""
            lines.append(
                f"  {failure['case']}{suffix}  {failure['check']}: {failure['detail']}"
            )
        if len(result.failures) > max_failures:
            lines.append(f"  … {len(result.failures) - max_failures} more")

    return "\n".join(lines)
