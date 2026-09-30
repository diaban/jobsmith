"""`python -m evals.compare` — the DAG agent against the ReAct baseline, same tasks.

    python -m evals.compare                       # provider auto-detected, like `make eval-llm`
    python -m evals.compare --llm openai --repeat 3
    python -m evals.compare --case width_every_site

Step −1 of docs/design/compiler-v1.md (#208): every compiler step is measured
against the `react` baseline (0206) on one task set, or "compiled is better"
stays a claim. Per run: did it answer, did the answer cover what the case says
it must, model calls, tokens, cost when the model is priced, wall time.

**Run-time width.** The cases are ones where the number of things to read is
only known after retrieval — the shape the compiler's `map` exists for. Their
material is a local fixture, written into a scratch docs root at run time and
handed to BOTH agents as the same `DocumentSource`: the same documents,
reproducibly, with no web and no key for retrieval. The DAG is the default
agent over that source; the baseline is `react` over the same.

**Success is coverage, not taste.** A case names the terms its answer must
mention (every site, the one value asked for); a run succeeds on the first try
when it answered and mentions all of them. It is deliberately crude and
deterministic: a judgement of quality would be a model grading models, and the
first question — did it read everything it had to — is answerable without one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jobsmith.agents.default import DefaultResources, LocalFiles
from jobsmith.app.agent import build_app
from jobsmith.app.providers import KeywordChatModel, load_dotenv, make_chat_model, make_llm

from .harness import resolve_provider
from .results import RESULTS_DIR

DAG, BASELINE = "default", "react"
AGENTS = (DAG, BASELINE)


# ------------------------------------------------------------------ material

#: Seven sites a width case has to find one by one, and two notes on other
#: subjects the search must not mistake for sites. Domain-neutral on purpose
#: (`make leak-check` scans this package); short, so each file is one chunk
#: and "the answer names Farrow" means "a passage about Farrow was read".
SITES: dict[str, dict[str, str]] = {
    "Alder": {"power": "solar", "storage": "40 kWh", "uplink": "8 Mbps", "staff": "3"},
    "Brackett": {"power": "grid", "storage": "none", "uplink": "100 Mbps", "staff": "12"},
    "Corran": {"power": "solar", "storage": "25 kWh", "uplink": "2 Mbps", "staff": "1"},
    "Dunmore": {"power": "diesel", "storage": "10 kWh", "uplink": "12 Mbps", "staff": "5"},
    "Eskdale": {"power": "solar", "storage": "60 kWh", "uplink": "20 Mbps", "staff": "4"},
    "Farrow": {"power": "grid", "storage": "15 kWh", "uplink": "50 Mbps", "staff": "8"},
    "Glenmoor": {"power": "solar", "storage": "30 kWh", "uplink": "4 Mbps", "staff": "2"},
}
DISTRACTORS: dict[str, str] = {
    "procurement-calendar.md": (
        "# Procurement calendar\n\nOrders for the next quarter close on the 14th. Laptops "
        "are renewed every four years; monitors every six. Late orders roll over."),
    "writing-guide.md": (
        "# Writing guide\n\nPrefer short sentences. Put the answer first. Name the source "
        "of every figure. Avoid acronyms a new reader would not know."),
}


def site_note(name: str, facts: dict[str, str]) -> str:
    storage = ("no battery storage" if facts["storage"] == "none"
               else f"{facts['storage']} of battery storage")
    return (f"# Field site: {name}\n\n"
            f"The {name} field site runs on {facts['power']} power with {storage}. "
            f"Its uplink is {facts['uplink']}. {facts['staff']} staff work there on a "
            f"normal day. This note is the site's own record.\n")


def write_material(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, facts in SITES.items():
        (root / f"site-{name.lower()}.md").write_text(site_note(name, facts), encoding="utf-8")
    for filename, text in DISTRACTORS.items():
        (root / filename).write_text(text, encoding="utf-8")
    return root


# ------------------------------------------------------------------ cases

@dataclass(frozen=True)
class CompareCase:
    id: str
    query: str
    #: terms the answer must mention, case-insensitively, as whole words
    must_mention: tuple[str, ...]
    #: how many things the task has to read — known to the case, never to the agents
    width: int
    note: str = ""


SOLAR = tuple(name for name, facts in SITES.items() if facts["power"] == "solar")

CASES: tuple[CompareCase, ...] = (
    CompareCase(
        id="width_every_site",
        query=("The notes describe several field sites. Compare every one of them on "
               "uplink bandwidth and power source, and say which site is the most "
               "constrained."),
        must_mention=tuple(SITES), width=len(SITES),
        note="every site, however many the search turns up"),
    CompareCase(
        id="width_solar_sites",
        query=("Which field sites described in the notes run on solar power? Give each "
               "one's battery storage."),
        must_mention=SOLAR, width=len(SOLAR),
        note="a filter: the width is what the notes say, not what the request says"),
    CompareCase(
        id="one_site_fact",
        query="What uplink bandwidth does the Dunmore field site have?",
        must_mention=("Dunmore", SITES["Dunmore"]["uplink"].split()[0]), width=1,
        note="control: one document, no fan-out to gain"),
)


def cases_for(only: tuple[str, ...] = ()) -> list[CompareCase]:
    return [case for case in CASES if not only or case.id in only]


def missing_terms(answer: str, terms: tuple[str, ...]) -> list[str]:
    """The terms `answer` does not mention, as whole words, ignoring case."""
    return [term for term in terms
            if not re.search(rf"(?<!\w){re.escape(term)}(?!\w)", answer, re.IGNORECASE)]


# ------------------------------------------------------------------ runs

@dataclass
class Run:
    agent: str
    case: str
    attempt: int
    answered: bool = False
    missing: list[str] = field(default_factory=list)
    calls: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    duration_s: float = 0.0
    job_id: str = ""
    status: str = ""
    error: str | None = None       # the harness broke, not the agent

    @property
    def success(self) -> bool:
        return self.answered and not self.missing and self.error is None


def _usage(run: Run, usage: dict[str, Any]) -> None:
    run.calls = int(usage.get("calls") or 0)
    run.tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    run.cost_usd = usage.get("cost_usd")


async def run_dag(app: Any, case: CompareCase, attempt: int) -> Run:
    run = Run(DAG, case.id, attempt)
    started = time.perf_counter()
    try:
        job = await app.manager.create_job(case.query, {})
        run.job_id = job.job_id
        job = await app.manager.run_job(job.job_id)
        answer = job.final_answer or ""
        run.status = job.status.value
        run.answered = job.terminal_kind == "answer" and bool(answer.strip())
        run.missing = missing_terms(answer, case.must_mention)
        _usage(run, job.usage)
    except Exception as e:                  # a broken harness must not look like a bad agent
        run.error = f"{type(e).__name__}: {e}"
    run.duration_s = time.perf_counter() - started
    return run


async def run_baseline(app: Any, case: CompareCase, attempt: int) -> Run:
    run = Run(BASELINE, case.id, attempt)
    started = time.perf_counter()
    try:
        job = await app.engine.create_job(
            {"messages": [{"role": "user", "content": case.query}]}, label=case.query)
        run.job_id = job.job_id
        job = await app.engine.run_job(job.job_id)
        answer = job.result if isinstance(job.result, str) else ""
        run.status = job.status.value
        run.answered = job.status.value == "done" and bool(answer.strip())
        run.missing = missing_terms(answer, case.must_mention)
        _usage(run, job.usage)
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
    run.duration_s = time.perf_counter() - started
    return run


async def compare(cases: list[CompareCase], *, provider: str, repeat: int = 1,
                  concurrency: int = 1, chat_model: Any = None,
                  agents: tuple[str, ...] = AGENTS) -> list[Run]:
    """Every case × `repeat` through each agent, on one shared fixture."""
    llm = make_llm(provider)
    model = chat_model or make_chat_model(provider)
    runs: list[Run] = []
    with TemporaryDirectory(prefix="jobsmith-compare-") as scratch:
        docs = write_material(Path(scratch) / "docs")
        resources = DefaultResources(documents=LocalFiles(docs), documents_root=str(docs))
        semaphore = asyncio.Semaphore(max(1, concurrency))
        for agent in agents:
            app = await build_app(
                agent=agent, llm=llm, db="memory", reports_dir=str(Path(scratch) / agent),
                resources=resources,
                # The DAG's chat is never opened; the baseline's graph IS its model.
                chat_model=model if agent == BASELINE else KeywordChatModel())
            try:
                runs += await _through(app, run_baseline if agent == BASELINE else run_dag,
                                       cases, repeat, semaphore)
            finally:
                await app.aclose()
    return runs


async def _through(app: Any, one: Any, cases: list[CompareCase], repeat: int,
                   semaphore: asyncio.Semaphore) -> list[Run]:
    async def bounded(case: CompareCase, attempt: int) -> Run:
        async with semaphore:
            return await one(app, case, attempt)

    return list(await asyncio.gather(*[
        bounded(case, attempt)
        for attempt in range(1, max(1, repeat) + 1) for case in cases]))


# ------------------------------------------------------------------ summary

def summarize(runs: list[Run]) -> dict[str, dict[str, dict[str, Any]]]:
    """`{case: {agent: {success, answered, calls, tokens, cost_usd, seconds, n}}}`,
    rates over the attempts, the rest averaged; cost only when every run was priced."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for case in dict.fromkeys(r.case for r in runs):
        table[case] = {}
        for agent in dict.fromkeys(r.agent for r in runs):
            mine = [r for r in runs if r.case == case and r.agent == agent]
            if not mine:
                continue
            n = len(mine)
            costs = [r.cost_usd for r in mine]
            table[case][agent] = {
                "n": n,
                "success": sum(r.success for r in mine) / n,
                "answered": sum(r.answered for r in mine) / n,
                "calls": sum(r.calls for r in mine) / n,
                "tokens": sum(r.tokens for r in mine) / n,
                "cost_usd": (sum(c for c in costs if c is not None) / n
                             if all(c is not None for c in costs) else None),
                "seconds": sum(r.duration_s for r in mine) / n,
                "errors": sum(r.error is not None for r in mine),
            }
    return table


def render(table: dict[str, dict[str, dict[str, Any]]]) -> str:
    head = f"{'case':<20} {'agent':<8} {'success':>7} {'calls':>6} {'tokens':>8} {'cost $':>8} {'secs':>6}"
    lines = [head, "-" * len(head)]
    for case, agents in table.items():
        for agent, row in agents.items():
            cost = "—" if row["cost_usd"] is None else f"{row['cost_usd']:.4f}"
            errors = f"  ({row['errors']} harness error)" if row["errors"] else ""
            lines.append(f"{case:<20} {agent:<8} {row['success']:>7.0%} {row['calls']:>6.1f} "
                         f"{row['tokens']:>8.0f} {cost:>8} {row['seconds']:>6.1f}{errors}")
    return "\n".join(lines)


def write_record(runs: list[Run], table: dict, *, provider: str,
                 directory: Path | str = RESULTS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"compare-{stamp}-{provider}.json"
    path.write_text(json.dumps({"provider": provider, "summary": table,
                                "runs": [asdict(r) for r in runs]}, indent=2) + "\n",
                    encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.compare",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--llm", choices=["anthropic", "openai", "fake"],
                        help="provider for both agents (default: auto-detected from the keys)")
    parser.add_argument("--case", action="append", default=[], metavar="ID")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per case and agent — one run of a model is noise")
    parser.add_argument("-j", "--concurrency", type=int, default=1)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--no-write", action="store_true")
    return parser


async def _main(args: argparse.Namespace) -> int:
    # The keys live in .env; it only fills what the environment has not set.
    # Not for the fakes: a keyless run must stay keyless.
    if args.llm != "fake":
        load_dotenv()
    provider = resolve_provider(args.llm)
    cases = cases_for(tuple(args.case))
    if not cases:
        print("no case matches", file=sys.stderr)
        return 2
    runs = await compare(cases, provider=provider, repeat=args.repeat,
                         concurrency=args.concurrency)
    table = summarize(runs)
    print(f"provider: {provider} · repeat: {args.repeat} · material: "
          f"{len(SITES)} site notes + {len(DISTRACTORS)} others")
    print(render(table))
    if not args.no_write:
        print(f"\nwritten: {write_record(runs, table, provider=provider, directory=args.results_dir)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
