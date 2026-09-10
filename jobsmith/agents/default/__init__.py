"""The default agent.

`read_files` / `documents` → `research` → `analysis` → `critique`, and
`slide_deck` when the request wants a presentation. The first step is what
keeps a job from being the model talking to itself; the rest reason over
whatever it found. The two grounding steps answer different questions —
`read_files` opens the document the request NAMED, `documents` searches the
configured material for a topic — which is why the planner is offered both
rather than one step with a mode.

Four steps appear **only when something backs them** — `read_files` with a
readable root (see `readable_roots`), `documents` with `--docs PATH` /
`$JOBSMITH_DOCS`, `web_search` with `$TAVILY_API_KEY`, `slide_deck` with a
`DeckRenderer` (the extra `.[pptx]`). A capability the agent cannot serve
should not be in the registry at all: the planner would otherwise plan a step
that always fails. With none of them, the agent still works, LLM-only,
needing nothing but a model key.
"""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from ...core.capability import Capability
from ..base import AgentContext, AgentDefinition
from .analysis import AnalysisCapability
from .critique import CritiqueCapability
from .documents import DocumentsCapability, WebSearchCapability
from .profile import DEFAULT_APP_PROFILE
from .read_files import ReadFilesCapability
from .research import ResearchCapability
from .slides import Deck, DeckRenderer, Slide, SlideDeckCapability
from .sources import Document, DocumentReader, DocumentSource, LocalFileReader, LocalFiles
from .web import TavilySource


@dataclass(frozen=True)
class DefaultResources:
    """One adapter per port this agent declares — none is also valid.

    Two adapters, one port: local files and the web are the same contract to
    the capability that consumes them, which is what the port was shaped for.
    `decks` is a second port (`slides.DeckRenderer`) with one adapter behind
    it — a presentation file is not a document source, and saying so keeps the
    contract shaped by its consumer.
    """

    documents: DocumentSource | None = None
    web: DocumentSource | None = None
    decks: DeckRenderer | None = None
    #: the directory `documents` was pointed at, when it IS a directory. Kept
    #: beside the adapter rather than read back out of it: a `DocumentSource`
    #: promises `search` and nothing else, and `read_files` needs the root as
    #: a fact, not as an implementation detail prised out of the port.
    documents_root: str | None = None


def pick_docs(spec: str | None = None) -> str | None:
    """Where to read documents from: argument > --docs= > $JOBSMITH_DOCS."""
    if spec:
        return spec
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--docs=")), None)
    return flag or os.environ.get("JOBSMITH_DOCS") or None


def _local_docs_root() -> str | None:
    """The directory `--docs` / `$JOBSMITH_DOCS` points at, if it is one."""
    spec = pick_docs()
    if not spec:
        return None
    root = Path(spec).expanduser()
    if not root.is_dir():
        print(f"[documents: {root} is not a directory — source disabled]", file=sys.stderr)
        return None
    print(f"[documents: local files under {root}]", file=sys.stderr)
    return str(root)


async def _open_web(stack: AsyncExitStack) -> DocumentSource | None:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None
    try:
        import httpx
    except ImportError:
        print("[web_search: httpx missing — install .[web]; source disabled]", file=sys.stderr)
        return None
    # The first adapter that genuinely has to be closed: the client is entered
    # on the app's stack, so its connection pool is released with the app —
    # whether it shut down cleanly or startup raised.
    client = await stack.enter_async_context(httpx.AsyncClient())
    print("[web_search: Tavily]", file=sys.stderr)
    return TavilySource(api_key, client)


async def _open_decks() -> DeckRenderer | None:
    """The renderer behind `slide_deck`, when a library can produce one.

    Nothing to open and nothing to close — `python-pptx` is pure Python — so
    this is an availability check, not a connection. It reads like `_open_web`
    because it answers the same question: is there anything behind this port?
    Silence when the extra is absent is deliberate: unlike a missing
    `$TAVILY_API_KEY` next to an installed httpx, nobody asked for decks here.
    """
    try:
        from .pptx_deck import PptxRenderer
    except ImportError:
        return None
    print("[slide_deck: .pptx via python-pptx]", file=sys.stderr)
    return PptxRenderer()


async def open_default_resources(stack: AsyncExitStack) -> DefaultResources:
    docs_root = _local_docs_root()
    return DefaultResources(
        documents=LocalFiles(docs_root) if docs_root else None,
        web=await _open_web(stack),
        decks=await _open_decks(),
        documents_root=docs_root,
    )


def readable_roots(ctx: AgentContext, resources: DefaultResources) -> tuple[str, ...]:
    """Where `read_files` may open a file, and nowhere else.

    Two entries, and both are already exposed by this deployment — this agent
    opens no new door:

    - what the composition root declared (`ctx.readable_roots`, i.e. the
      directory jobs write their deliverables and annexes into). That is the
      whole point of the step: the report a job wrote twenty minutes ago is
      exactly the file the next request names, so the write-then-read loop is
      covered deliberately rather than by accident.
    - the `--docs` directory, when there is one. A capability can already
      search it and quote any passage of any file in it, so refusing to open
      one *by name* would protect nothing and surprise everyone.

    Order matters only for a relative reference that could match in both:
    the job artifacts win, because that is the tree this product's own paths
    point into.
    """
    roots = [root for root in ctx.readable_roots if root]
    if resources.documents_root:
        roots.append(resources.documents_root)
    return tuple(dict.fromkeys(roots))


def default_capabilities(ctx: AgentContext) -> list[Capability]:
    llm = ctx.llm
    resources: DefaultResources = ctx.resources or DefaultResources()
    capabilities: list[Capability] = []
    # First, because a document the request named is the most specific
    # material there is — and, like every other conditional step, present only
    # when something backs it: no readable root, no capability. `requires_inputs`
    # is the second gate, dropping it from any plan for a request that named
    # no file at all.
    if roots := readable_roots(ctx, resources):
        capabilities.append(ReadFilesCapability(LocalFileReader(roots)))
    if resources.documents is not None:
        capabilities.append(DocumentsCapability(llm, resources.documents))
    if resources.web is not None:
        capabilities.append(WebSearchCapability(llm, resources.web))
    capabilities += [ResearchCapability(llm), AnalysisCapability(llm), CritiqueCapability(llm)]
    # Last, so it is offered to the planner after the steps whose material it
    # presents. Both conditions are the same rule: a renderer is what turns a
    # deck into a file, a store is where that file goes, and a capability
    # nothing can serve stays out of the registry.
    if resources.decks is not None and ctx.artifacts is not None:
        capabilities.append(SlideDeckCapability(llm, ctx.artifacts, resources.decks))
    return capabilities


DEFAULT_AGENT = AgentDefinition(
    name="default",
    description=(
        "General-purpose analyst: documents, web search, research, analysis, "
        "critique, and a slide deck when the request wants one."
    ),
    capabilities=default_capabilities,
    profile=DEFAULT_APP_PROFILE,
    open_resources=open_default_resources,
)

__all__ = [
    "DEFAULT_AGENT",
    "AnalysisCapability",
    "CritiqueCapability",
    "Deck",
    "DeckRenderer",
    "DefaultResources",
    "Document",
    "DocumentReader",
    "DocumentSource",
    "DocumentsCapability",
    "Slide",
    "SlideDeckCapability",
    "TavilySource",
    "WebSearchCapability",
    "LocalFileReader",
    "LocalFiles",
    "ReadFilesCapability",
    "ResearchCapability",
    "readable_roots",
    "default_capabilities",
    "open_default_resources",
    "pick_docs",
]
