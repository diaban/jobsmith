"""Probe one node of the composed agent, N times per request.

The cheap instrument for a prompt change. A full-job eval pays for every node
of every run to observe one decision; this calls only the node that owns the
prompt, composed exactly as `build_app` composes it (same registry, same
formats, same profile), and tallies what it wrote. Run it on the branch and on
`main` and compare: `make probe NODE=… READ=… CASES=…` does both, in
parallel, from a scratch worktree of `main` (`scripts/probe-compare.sh`).
Versioned case sets live in `evals/probes/<node>.json`. It is what the budget
in CLAUDE.md is measured with.

    python -m evals.probe --node document_intent --read document_formats \\
        --cases cases.json -n 10 --out after.json --baseline before.json
    python -m evals.probe --node direct_answer --read draft_answer \\
        --grep "would you like|tell me" -q "what can you do? as a markdown file"

A case is `{"id", "query"}` or `{"id", "state"}` (a node reading more than the
query), with an optional `"expect"`: `"truthy"`, `"falsy"`, or a JSON value the
read key must equal. Half of a probe's cases should be controls — a fix
measured only where it should fire hides what it broke.

`.env` is loaded (unlike `python -m evals`): this is a real-model tool.
Refused above `--max-calls` (default 300); an LLM error is counted apart from
the tally, never as a miss.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

MAX_CALLS = 300


@dataclass
class Tally:
    """What one case's N calls wrote under the read key."""

    case_id: str
    counts: Counter = field(default_factory=Counter)
    passed: int = 0
    errors: int = 0
    expect: Any = None

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "n": self.n, "errors": self.errors,
                "passed": self.passed if self.expect is not None else None}


def load_cases(path: str | None, queries: list[str]) -> list[dict[str, Any]]:
    cases = json.loads(Path(path).read_text(encoding="utf-8")) if path else []
    cases += [{"id": f"q{i}", "query": q} for i, q in enumerate(queries, 1)]
    for case in cases:
        if "state" not in case and "query" not in case:
            raise ValueError(f"case {case.get('id')!r} has neither 'query' nor 'state'")
    return cases


def observe(output: dict[str, Any], read: str, grep: re.Pattern[str] | None) -> Any:
    """The value tallied for one call: the key itself, or whether it matched."""
    value = (output or {}).get(read)
    if grep is not None:
        return bool(grep.search(value if isinstance(value, str) else json.dumps(value)))
    return value


def meets(value: Any, expect: Any) -> bool:
    if expect == "truthy":
        return bool(value)
    if expect == "falsy":
        return not value
    return value == expect


def _label(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 60 else text[:57] + "..."


async def probe(
    node: str, cases: list[dict[str, Any]], *, read: str, n: int = 10,
    grep: str | None = None, provider: str | None = None, agent: str | None = None,
    concurrency: int = 8, max_calls: int = MAX_CALLS,
) -> list[Tally]:
    """Call `node` n times per case and tally what it wrote under `read`."""
    if len(cases) * n > max_calls:
        raise ValueError(f"{len(cases)} cases x {n} = {len(cases) * n} calls, "
                         f"over the budget of {max_calls} (--max-calls)")
    from jobsmith.app.agent import build_app
    from jobsmith.app.providers import KeywordChatModel, load_dotenv, make_llm, pick_provider

    if provider != "fake":
        load_dotenv()           # never under the fake: it would leak keys into the process
    pattern = re.compile(grep, re.IGNORECASE) if grep else None
    with TemporaryDirectory(prefix="jobsmith-probe-") as scratch:
        app = await build_app(agent=agent, llm=make_llm(provider or pick_provider()),
                              chat_model=KeywordChatModel(), db="memory",
                              reports_dir=scratch)
        try:
            nodes = app.manager.graph.nodes
            if node not in nodes:
                raise ValueError(f"no node {node!r}; this agent has: "
                                 f"{', '.join(sorted(k for k in nodes if not k.startswith('__')))}")
            runnable = nodes[node].bound
            semaphore = asyncio.Semaphore(max(1, concurrency))
            tallies = [Tally(c["id"], expect=c.get("expect")) for c in cases]

            async def one(case: dict[str, Any], tally: Tally) -> None:
                state = dict(case.get("state") or {"query": case["query"]})
                async with semaphore:
                    try:
                        output = await runnable.ainvoke(state)
                    except Exception:
                        tally.errors += 1
                        return
                if any(e.get("recoverable") is False for e in (output or {}).get("errors", [])):
                    tally.errors += 1       # the node caught its own LLM failure
                    return
                value = observe(output, read, pattern)
                tally.counts[_label(value)] += 1
                if tally.expect is not None and meets(value, tally.expect):
                    tally.passed += 1

            await asyncio.gather(*[one(c, t) for c, t in zip(cases, tallies, strict=True) for _ in range(n)])
        finally:
            await app.aclose()
    return tallies


def render(after: dict[str, Any], before: dict[str, Any] | None = None) -> str:
    """One line per case from two `--out` summaries: `before -> after`, top values."""
    lines = []
    for case_id, now in after.items():
        was = (before or {}).get(case_id)
        score = f"{now['passed']}/{now['n']}" if now.get("passed") is not None else "-"
        if was and was.get("passed") is not None:
            score = f"{was['passed']}/{was['n']} -> {score}"
        top = sorted(now["counts"].items(), key=lambda kv: -kv[1])[:3]
        seen = ", ".join(f"{k} x{v}" for k, v in top)
        errors = f"  ({now['errors']} errors)" if now.get("errors") else ""
        lines.append(f"{case_id:<28} {score:>12}  {seen}{errors}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.probe",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--node", help="graph node to call (e.g. document_intent)")
    parser.add_argument("--read", help="state key the node writes, to tally")
    parser.add_argument("--grep", help="tally whether the read key matches this regex instead")
    parser.add_argument("--cases", help="JSON file of cases")
    parser.add_argument("-q", "--query", action="append", default=[], help="an ad-hoc case")
    parser.add_argument("-n", type=int, default=10, help="calls per case (default 10)")
    parser.add_argument("--llm", choices=["anthropic", "openai", "fake"])
    parser.add_argument("--agent")
    parser.add_argument("-j", "--concurrency", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--out", help="write the per-case summary here (JSON)")
    parser.add_argument("--baseline", help="a previous --out, shown as 'before -> after'")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="only print two --out summaries side by side (no call made)")
    args = parser.parse_args(argv)
    if args.compare:
        before, after = (json.loads(Path(f).read_text(encoding="utf-8")) for f in args.compare)
        print(render(after, before))
        return 0
    if not (args.node and args.read):
        parser.error("--node and --read are required unless --compare is given")
    try:
        cases = load_cases(args.cases, args.query)
        tallies = asyncio.run(probe(
            args.node, cases, read=args.read, n=args.n, grep=args.grep, provider=args.llm,
            agent=args.agent, concurrency=args.concurrency, max_calls=args.max_calls))
    except ValueError as e:
        print(f"probe: {e}", file=sys.stderr)
        return 2
    summaries = {t.case_id: t.summary() for t in tallies}
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    print(render(summaries, baseline))
    if args.out:
        Path(args.out).write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
