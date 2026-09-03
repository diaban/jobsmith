"""The default agent.

`documents` → `research` → `analysis` → `critique`. The first step is what
keeps a job from being the model talking to itself; the rest reason over
whatever it found.

Both retrieval steps appear **only when something backs them** — `documents`
with `--docs PATH` / `$JOBSMITH_DOCS`, `web_search` with `$TAVILY_API_KEY`. A
capability the agent cannot serve should not be in the registry at all: the
planner would otherwise plan a step that always fails. With neither, the agent
still works, LLM-only, needing nothing but a model key.
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
from .sources import Document, DocumentSource, LocalFiles
from .web import TavilySource


@dataclass(frozen=True)
class DefaultResources:
    """One adapter per port this agent declares — none is also valid.

    Two adapters, one port: local files and the web are the same contract to
    the capability that consumes them, which is what the port was shaped for.
    """

    documents: DocumentSource | None = None
    web: DocumentSource | None = None


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


async def open_default_resources(stack: AsyncExitStack) -> DefaultResources:
    return DefaultResources(documents=await _open_local_files(), web=await _open_web(stack))


def default_capabilities(ctx: AgentContext) -> list[Capability]:
    llm = ctx.llm
    resources: DefaultResources = ctx.resources or DefaultResources()
    capabilities: list[Capability] = []
    if resources.documents is not None:
        capabilities.append(DocumentsCapability(llm, resources.documents))
    if resources.web is not None:
        capabilities.append(WebSearchCapability(llm, resources.web))
    capabilities += [ResearchCapability(llm), AnalysisCapability(llm), CritiqueCapability(llm)]
    return capabilities


DEFAULT_AGENT = AgentDefinition(
    name="default",
    description=(
        "General-purpose analyst: documents, web search, research, analysis "
        "and critique."
    ),
    capabilities=default_capabilities,
    profile=DEFAULT_APP_PROFILE,
    open_resources=open_default_resources,
)

__all__ = [
    "DEFAULT_AGENT",
    "AnalysisCapability",
    "CritiqueCapability",
    "DefaultResources",
    "Document",
    "DocumentSource",
    "DocumentsCapability",
    "TavilySource",
    "WebSearchCapability",
    "LocalFiles",
    "ResearchCapability",
    "default_capabilities",
    "open_default_resources",
    "pick_docs",
]
