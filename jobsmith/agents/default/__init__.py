"""The default agent.

`documents` → `research` → `analysis` → `critique`, and `slide_deck` when the
request wants a presentation. The first step is what keeps a job from being
the model talking to itself; the rest reason over whatever it found.

Three steps appear **only when something backs them** — `documents` with
`--docs PATH` / `$JOBSMITH_DOCS`, `web_search` with `$TAVILY_API_KEY`,
`slide_deck` with a `DeckRenderer` (the extra `.[pptx]`). A capability the
agent cannot serve should not be in the registry at all: the planner would
otherwise plan a step that always fails. With none of them, the agent still
works, LLM-only, needing nothing but a model key.
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
from .research import ResearchCapability
from .slides import Deck, DeckRenderer, Slide, SlideDeckCapability
from .sources import Document, DocumentSource, LocalFiles
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


def pick_docs(spec: str | None = None) -> str | None:
    """Where to read documents from: argument > --docs= > $JOBSMITH_DOCS."""
    if spec:
        return spec
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--docs=")), None)
    return flag or os.environ.get("JOBSMITH_DOCS") or None


async def _open_local_files() -> DocumentSource | None:
    spec = pick_docs()
    if not spec:
        return None
    root = Path(spec).expanduser()
    if not root.is_dir():
        print(f"[documents: {root} is not a directory — source disabled]", file=sys.stderr)
        return None
    print(f"[documents: local files under {root}]", file=sys.stderr)
    return LocalFiles(root)


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
    return DefaultResources(
        documents=await _open_local_files(),
        web=await _open_web(stack),
        decks=await _open_decks(),
    )


def default_capabilities(ctx: AgentContext) -> list[Capability]:
    llm = ctx.llm
    resources: DefaultResources = ctx.resources or DefaultResources()
    capabilities: list[Capability] = []
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
    "DocumentSource",
    "DocumentsCapability",
    "Slide",
    "SlideDeckCapability",
    "TavilySource",
    "WebSearchCapability",
    "LocalFiles",
    "ResearchCapability",
    "default_capabilities",
    "open_default_resources",
    "pick_docs",
]
