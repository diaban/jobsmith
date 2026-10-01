"""Probe: can a provider's strict mode hold the compiler's IR? (#219, step 1a)

`docs/design/compiler-v1.md` rests on one hypothesis: the planner can be
constrained by a JSON schema generated from the registry. This measures it,
before any product code depends on it (record 0219):

1. **Acceptance.** The step-1 IR schema — the default registry's ops plus the
   1d kit (`analyze`, `extract` with its `fields` form, `synthesize`), each
   op's `args`, `$…` references, `map`, `when`, `output` — is written as
   Pydantic models and submitted in three forms: as Pydantic emits it,
   through `strict()` (the draft of the one normaliser 1c will ship), and
   `degraded` (0219's fallback: a keyword the provider refuses moves into the
   description). What a provider refuses is its 400, recorded verbatim.
2. **Validity on the first call**, on one request set, each constrained
   variant in the first form its provider accepts: `full` (a step shape per
   op, discriminated by `op`), `shared` (one step shape, `args` typed by shape,
   the op/args pairing left to the analysis), `fallback` (`args` a JSON
   string) and `plain` (today's `json_object`). A program is valid when it
   parses against the IR models AND passes the structural checks
   `findings()` runs (analysis checks 1 and 3: unique ids, references that
   resolve, `$item` only in a `map`, `after`/`output` that name steps, no
   cycle). No repair: first call only.

`--static` also reports what Anthropic's SDK `transform_schema` would move
out of the grammar. `.env` is loaded; refused above `--max-calls` (300).

    python -m evals.strict_probe --provider openai -n 4 --out probe.json
    python -m evals.strict_probe --provider anthropic -n 4   # needs a key
    python -m evals.strict_probe --provider openai --variants shared,plain
    python -m evals.strict_probe --static                    # no call
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

MAX_CALLS = 300

ID = r"^[a-z][a-z0-9_]*$"
REF = r"^\$(input|item|[a-z][a-z0-9_]*)(\.[a-z0-9_]+)*$"
Id = Annotated[str, Field(pattern=ID)]
Ref = Annotated[str, Field(pattern=REF)]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# -------- the ops' arguments (the step-1 kit over the default registry) --------

class Query(_Closed):
    query: str


class Files(_Closed):
    paths: list[str] | Ref           # a value or a reference: `$input.source_files`


class Jobs(_Closed):
    jobs: list[str] | Ref


class Material(_Closed):
    material: list[Ref]


class Instructed(_Closed):
    material: list[Ref]
    instruction: str


class Items(_Closed):
    items: list[Ref]
    instruction: str


class FieldSpec(_Closed):
    name: Id
    type: Literal["string", "integer", "number", "boolean", "enum", "string[]", "integer[]"]
    values: Annotated[list[str], Field(max_length=12)] | None = None


class Extract(_Closed):
    material: list[Ref]
    fields: Annotated[list[FieldSpec], Field(max_length=8)] | None = None
    schema_name: Literal["RepoFacts"] | None = None


# (op, args model, description): the descriptions of the registry's ops are the
# gist of their CapabilitySpec's, the kit's are the note's table.
OPS: tuple[tuple[str, type[BaseModel], str], ...] = (
    ("prior_jobs", Jobs, "read what earlier jobs of this session produced"),
    ("read_files", Files, "read, in full, the files the request names"),
    ("documents", Query, "search the user's local documents; items: [{id, source, title, text}]"),
    ("web_search", Query, "search the public web; items: [{id, source, title, text}]"),
    ("research", Instructed, "break the request into aspects and write research notes"),
    ("analysis", Material, "key findings, tensions and implications of the material"),
    ("critique", Material, "claims the material does not support, gaps, disagreements"),
    ("slide_deck", Material, "a slide deck file, only when slides are asked for; last"),
    ("analyze", Instructed, "one model call over the material; output {text}"),
    ("extract", Extract, "an object of the given fields (or a named schema) per material"),
    ("synthesize", Items, "reduce several results (a map's) into one text; output {text}"),
)
OP_NAMES = tuple(name for name, _, _ in OPS)


class MinOk(_Closed):
    min_ok: int = Field(ge=1)


class MapSpec(_Closed):
    over: Ref
    max_items: int = Field(ge=1, le=50)
    concurrency: int = Field(ge=1, le=10)
    on_error: Literal["all", "partial"] | MinOk


class When(_Closed):
    ref: Ref
    in_: list[str] = Field(alias="in")


def _step(op: str, args: Any) -> type[BaseModel]:
    return create_model(
        f"Step_{op}", __base__=_Closed,
        id=(Id, ...), op=(Literal[op], ...), args=(args, ...),  # type: ignore[valid-type]
        after=(list[Id], []), map=(MapSpec | None, None),
        when=(When | None, None))


FullStep = Annotated[Union[tuple(_step(n, a) for n, a, _ in OPS)], Field(discriminator="op")]  # noqa: UP007 — a runtime tuple


class Program(_Closed):
    version: Literal[1]
    steps: Annotated[list[FullStep], Field(min_length=1)]  # type: ignore[valid-type]
    output: list[Id] | None = None


_ARGS = tuple(dict.fromkeys(model for _, model, _ in OPS))  # the 7 distinct shapes


class SharedStep(_Closed):
    """One step shape for every op: `args` is typed by shape, and which shape
    goes with which op is left to the analysis (Anthropic refuses the
    discriminated union as a grammar too large, 0219)."""
    id: Id
    op: Literal[OP_NAMES]  # type: ignore[valid-type]
    args: Union[_ARGS]  # type: ignore[valid-type]  # noqa: UP007 — a runtime tuple
    after: list[Id] = []
    map: MapSpec | None = None
    when: When | None = None


class SharedProgram(_Closed):
    version: Literal[1]
    steps: Annotated[list[SharedStep], Field(min_length=1)]
    output: list[Id] | None = None


class LooseStep(_Closed):
    """The fallback: structure constrained, arguments a JSON-encoded string."""
    id: Id
    op: Literal[OP_NAMES]  # type: ignore[valid-type]
    args_json: str
    after: list[Id] = []
    map: MapSpec | None = None
    when: When | None = None


class LooseProgram(_Closed):
    version: Literal[1]
    steps: Annotated[list[LooseStep], Field(min_length=1)]
    output: list[Id] | None = None


# -------- the normaliser 1c will ship, in draft --------

def strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic's JSON Schema made strict-mode ready: every object closed with
    every property required (an optional one is already `anyOf […, null]`),
    `oneOf` → `anyOf` without `discriminator`, `const` → a one-value `enum`,
    no `default` or `title`."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items()
               if k not in ("default", "title", "discriminator")}
        if "oneOf" in out:
            out["anyOf"] = out.pop("oneOf")
        if "const" in out:
            out["enum"] = [out.pop("const")]
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out
    return walk(copy.deepcopy(schema))


MODELS: dict[str, type[BaseModel]] = {
    "full": Program, "shared": SharedProgram, "fallback": LooseProgram, "plain": Program}


def schema_of(variant: str) -> dict[str, Any]:
    return MODELS[variant].model_json_schema()


def schema_stats(schema: dict[str, Any]) -> dict[str, Any]:
    counts: Counter = Counter()

    def walk(node: Any, depth: int) -> int:
        deepest = depth
        if isinstance(node, dict):
            for key in ("anyOf", "oneOf", "enum", "const", "pattern", "maxItems",
                        "minItems", "minimum", "maximum", "$ref"):
                if key in node:
                    counts[key] += 1
            counts["properties"] += len(node.get("properties", {}))
            for value in node.values():
                deepest = max(deepest, walk(value, depth + 1))
        elif isinstance(node, list):
            for value in node:
                deepest = max(deepest, walk(value, depth))
        return deepest
    depth = walk(schema, 0)
    stats: dict[str, Any] = dict(counts)
    stats["depth"] = depth
    stats["defs"] = len(schema.get("$defs", {}))
    stats["chars"] = len(json.dumps(schema))
    return stats


def anthropic_relocated(schema: dict[str, Any]) -> dict[str, Any]:
    """What the Anthropic SDK's `transform_schema` keeps out of the grammar:
    each keyword it appends to a description instead ("{pattern: …}")."""
    from anthropic.lib._parse._transform import transform_schema

    moved: Counter = Counter()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            text = node.get("description")
            if isinstance(text, str) and "{" in text:
                for key in re.findall(r"[{,] ?([a-zA-Z]+):", text.split("\n\n")[-1]):
                    moved[key] += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(transform_schema(schema))
    return dict(moved)


# -------- validity of one answer --------

def refs_in(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.startswith("$") else []
    if isinstance(value, list):
        return [r for v in value for r in refs_in(v)]
    if isinstance(value, dict):
        return [r for v in value.values() for r in refs_in(v)]
    return []


def findings(program: dict[str, Any]) -> list[str]:
    """Analysis checks 1 and 3 (plus `output`'s half of 6): codes, not prose."""
    codes: list[str] = []
    steps = program["steps"]
    ids = [s["id"] for s in steps]
    if len(set(ids)) != len(ids):
        codes.append("duplicate_id")
    known = set(ids)
    edges: dict[str, set[str]] = defaultdict(set)
    for s in steps:
        mapped = s.get("map") is not None
        refs = refs_in(s["args"]) + refs_in(s.get("map") or {}) + refs_in(s.get("when") or {})
        for ref in refs:
            head = ref[1:].split(".", 1)[0]
            if head == "item":
                if not mapped:
                    codes.append("item_outside_map")
            elif head == "input":
                continue
            elif head not in known:
                codes.append("unknown_ref")
            else:
                edges[s["id"]].add(head)
        for after in s.get("after") or []:
            if after not in known:
                codes.append("unknown_after")
            edges[s["id"]].add(after)
    for out in program.get("output") or []:
        if out not in known:
            codes.append("unknown_output")
    if _cyclic(ids, edges):
        codes.append("cycle")
    return codes


def _cyclic(ids: list[str], edges: dict[str, set[str]]) -> bool:
    indegree = {i: len(edges[i] & set(ids)) for i in ids}
    ready = [i for i, d in indegree.items() if d == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for other in ids:
            if node in edges[other]:
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
    return seen < len(set(ids))


def judge(variant: str, text: str) -> tuple[str, dict[str, Any] | None]:
    """`valid`, or the first reason it is not: `json`, `schema`, `args`, a finding."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return "json", None
    try:
        program = MODELS[variant].model_validate(raw).model_dump(by_alias=True)
    except ValidationError as exc:
        where = str(exc.errors()[0]["loc"])
        return ("args" if "args" in where else "schema"), raw
    args_of = {name: model for name, model, _ in OPS}
    if variant == "fallback":
        try:
            for step in program["steps"]:
                step["args"] = args_of[step["op"]].model_validate_json(
                    step.pop("args_json")).model_dump()
        except ValidationError:
            return "args", raw
    if variant == "shared":  # the pairing the schema left to the analysis
        try:
            for step in program["steps"]:
                args_of[step["op"]].model_validate(step["args"])
        except ValidationError:
            return "op_args", raw
    codes = findings(program)
    return (codes[0] if codes else "valid"), program


# -------- the prompt, the requests, the calls --------

def _args_line(model: type[BaseModel]) -> str:
    """`material: [ref], instruction: string`: the types every variant is told,
    so an unconstrained one is not judged on types it never saw."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def render(node: dict[str, Any]) -> str:
        if "$ref" in node:
            target = defs[node["$ref"].rsplit("/", 1)[-1]]
            return "{" + ", ".join(f"{k}: {render(v)}"
                                   for k, v in target["properties"].items()) + "}"
        if "anyOf" in node:
            return " | ".join(render(v) for v in node["anyOf"])
        if "enum" in node or "const" in node:
            return "|".join(json.dumps(v) for v in node.get("enum", [node.get("const")]))
        if node.get("type") == "array":
            return f"[{render(node['items'])}]"
        return "ref" if node.get("pattern") == REF else node.get("type", "any")
    return ", ".join(f"{k}: {render(v)}" for k, v in schema["properties"].items())


PROMPT = """You are the planner of an assistant agent. Write a PROGRAM that answers
the user's request with the operations below, as one JSON object.

Operations (name(args): what it does):
{ops}

A program: {{"version": 1, "steps": [...], "output": [<step ids>] or null}}.
A step: {{"id": "<snake_case, unique>", "op": "<operation>", {args_hint},
"after": [<ids to wait for, no data passed>], "map": null or {{...}}, "when": null}}.
- An argument may be a reference instead of a value, as a whole string:
  "$<id>" (a step's result), "$<id>.<field>" (one of its fields),
  "$input.<key>" (the job's inputs, e.g. "$input.source_files"),
  and inside a map "$item" or "$item.<field>". No string interpolation.
- Dependencies come from references and "after"; a step may only reference
  steps of the same program; the program must be acyclic.
- "map": run the step once per element of a list another step produced:
  {{"over": "$<id>.items", "max_items": <1-50>, "concurrency": <1-10>,
  "on_error": "all" | "partial" | {{"min_ok": <n>}}}}; "$item" is the element.
- "output": the steps whose results the answer is written from; null = all.
- extract takes "fields" (at most 8 of {{"name", "type", "values"}}, type one of
  string, integer, number, boolean, enum, string[], integer[]; "values" only
  for enum) or "schema_name"; set the one you do not use to null.
- Use only the steps the request needs. Return ONLY the JSON object."""


def system_prompt(variant: str) -> str:
    ops = "\n".join(f"- {name}({_args_line(model)}): {text}" for name, model, text in OPS)
    hint = ('"args_json": "<the args object, JSON-encoded as a string>"'
            if variant == "fallback" else '"args": {<the operation\'s arguments>}')
    return PROMPT.format(ops=ops, args_hint=hint)


REQUESTS: tuple[tuple[str, str], ...] = (
    ("sites_compare", "The notes describe several field sites. Compare every one of them "
     "on uplink bandwidth and power source, and say which site is the most constrained."),
    ("sites_filter", "Which field sites described in the notes run on solar power? "
     "Give each one's battery storage."),
    ("repos_compare", "Find the GitHub repositories named jobsmith and compare them on "
     "purpose and activity."),
    ("extract_table", "From the documents about our vendors, list each vendor's name, "
     "yearly price and whether the contract auto-renews."),
    ("research_critique", "Research how teams roll out feature flags safely, analyse the "
     "trade-offs and point out where the sources disagree."),
    ("deck", "Make a short slide deck comparing TCP and UDP for real-time games."),
    ("control_fact", "What uplink bandwidth does the Dunmore field site have?"),
    ("control_web", "What is the latest stable release of PostgreSQL?"),
    ("control_file", "Summarise the file I gave you (source_files is set)."),
    ("control_prior", "Turn the result of my previous job into three bullet points."),
)
VARIANTS = ("full", "shared", "fallback", "plain")
CONSTRAINED = VARIANTS[:-1]


def response_format(provider: str, variant: str, schema: dict[str, Any]) -> dict[str, Any]:
    if variant == "plain":
        return ({"response_format": {"type": "json_object"}} if provider == "openai" else {})
    if provider == "openai":
        return {"response_format": {"type": "json_schema", "json_schema": {
            "name": "program", "strict": True, "schema": schema}}}
    return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}


class Caller:
    def __init__(self, provider: str):
        self.provider = provider
        if provider == "openai":
            from openai import AsyncOpenAI
            self.client: Any = AsyncOpenAI()
            self.model = os.environ.get("OPENAI_MODEL", "gpt-5.1")
        else:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic()
            self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")

    async def __call__(self, system: str, user: str, fmt: dict[str, Any]) -> tuple[str, int]:
        if self.provider == "openai":
            r = await self.client.chat.completions.create(
                model=self.model, max_completion_tokens=16000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}], **fmt)
            return r.choices[0].message.content or "", r.usage.completion_tokens
        r = await self.client.messages.create(
            model=self.model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}], **fmt)
        text = "".join(b.text for b in r.content if b.type == "text")
        from jobsmith.dag.clients import AnthropicLLMClient  # what the product parses
        return AnthropicLLMClient._strip_fences(text), r.usage.output_tokens


async def accepts(call: Caller, schema: dict[str, Any]) -> str:
    """`accepted`, or the provider's refusal, verbatim (one call)."""
    try:
        await call("Return a program with one step.", "What is 2 + 2?",
                   response_format(call.provider, "full", schema))
        return "accepted"
    except Exception as exc:  # noqa: BLE001 — the refusal IS the measurement
        return f"{type(exc).__name__}: {str(exc)[:600]}"


REFUSED = ("minimum", "maximum", "maxItems")  # by Anthropic, measured (0219)


def degraded(schema: dict[str, Any]) -> dict[str, Any]:
    """0219's fallback: a keyword the provider refuses moves into the
    description (the analysis enforces it); `pattern` stays, it is accepted."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {k: walk(v) for k, v in node.items() if k not in REFUSED}
        moved = {k: node[k] for k in REFUSED if k in node}
        if moved:
            out["description"] = (out.get("description", "") + f" {moved}").strip()
        return out
    return walk(schema)


FORMS = {"pydantic": lambda v: schema_of(v), "strict": lambda v: strict(schema_of(v)),
         "degraded": lambda v: degraded(strict(schema_of(v)))}


async def measure(call: Caller, n: int, concurrency: int, forms: dict[str, str],
                  variants: tuple[str, ...] = VARIANTS) -> dict[str, Any]:
    gate = asyncio.Semaphore(concurrency)
    tally: dict[str, Counter] = {v: Counter() for v in VARIANTS}
    per_case: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    cost: dict[str, list[float]] = {v: [0.0, 0.0] for v in VARIANTS}  # seconds, tokens
    samples: dict[str, str] = {}

    async def one(variant: str, case: str, query: str) -> None:
        form = FORMS[forms.get(variant, "strict")](variant)
        fmt = response_format(call.provider, variant, form)
        async with gate:
            start = time.monotonic()
            try:
                text, tokens = await call(system_prompt(variant), query, fmt)
            except Exception as exc:  # noqa: BLE001 — counted apart, never a miss
                tally[variant]["error"] += 1
                samples.setdefault(f"{variant}/error", str(exc)[:300])
                return
            cost[variant][0] += time.monotonic() - start
            cost[variant][1] += tokens
        verdict, program = judge(variant, text)
        tally[variant][verdict] += 1
        per_case[case][variant][verdict] += 1
        samples.setdefault(f"{variant}/{case}/{verdict}", text[:1500])

    await asyncio.gather(*(one(v, c, q) for v in variants if forms.get(v) != "refused"
                           for c, q in REQUESTS for _ in range(n)))
    return {"tally": {v: dict(c) for v, c in tally.items()},
            "per_case": {c: {v: dict(t) for v, t in vs.items()} for c, vs in per_case.items()},
            "mean_seconds": {v: round(s / max(sum(tally[v].values()), 1), 1)
                             for v, (s, _) in cost.items()},
            "mean_output_tokens": {v: round(t / max(sum(tally[v].values()), 1))
                                   for v, (_, t) in cost.items()},
            "forms": forms, "samples": samples}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("openai", "anthropic"))
    parser.add_argument("--static", action="store_true", help="schema facts only, no call")
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help="the variants to measure, comma-separated (default: all)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    variants = tuple(v for v in VARIANTS if v in args.variants.split(","))
    constrained = tuple(v for v in CONSTRAINED if v in variants)

    report: dict[str, Any] = {"static": {}}
    for variant in CONSTRAINED:
        raw, normalised = schema_of(variant), strict(schema_of(variant))
        report["static"][variant] = {
            "pydantic": schema_stats(raw), "strict": schema_stats(normalised),
            "anthropic_sdk_relocates": anthropic_relocated(normalised)}
    if not args.static:
        if args.provider is None:
            parser.error("--provider is required unless --static")
        calls = len(constrained) * len(FORMS) + len(variants) * len(REQUESTS) * args.n
        if calls > args.max_calls:
            parser.error(f"{calls} calls > --max-calls {args.max_calls}")
        from jobsmith.app.providers import load_dotenv
        load_dotenv()
        call = Caller(args.provider)
        report["model"] = call.model

        async def run() -> None:
            report["acceptance"] = {
                f"{variant}/{form}": await accepts(call, build(variant))
                for variant in constrained for form, build in FORMS.items()}
            ok = report["acceptance"]
            forms = {v: next((f for f in ("strict", "degraded")
                              if ok[f"{v}/{f}"] == "accepted"), "refused")
                     for v in constrained}
            report["validity"] = await measure(call, args.n, args.concurrency, forms, variants)
        asyncio.run(run())
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    summary = {k: v for k, v in report.items() if k != "validity"}
    if "validity" in report:
        summary["validity"] = {k: v for k, v in report["validity"].items() if k != "samples"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
