"""The default agent.

`documents` → `research` → `analysis` → `critique`. The first step is what
keeps a job from being the model talking to itself; the rest reason over
whatever it found.

`documents` appears **only when a source is configured** (`--docs PATH`, or
`$JOBSMITH_DOCS`). A capability the agent cannot serve should not be in the
registry at all: the planner would otherwise plan a step that always fails.
Without it the agent still works, LLM-only, needing nothing but a key.
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
from .documents import DocumentsCapability
from .profile import DEFAULT_APP_PROFILE
from .research import ResearchCapability
from .sources import Document, DocumentSource, LocalFiles


@dataclass(frozen=True)
class DefaultResources:
    """One adapter per port this agent declares — none is also valid."""

    documents: DocumentSource | None = None


def pick_docs(spec: str | None = None) -> str | None:
    """Where to read documents from: argument > --docs= > $JOBSMITH_DOCS."""
    if spec:
        return spec
    flag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--docs=")), None)
    return flag or os.environ.get("JOBSMITH_DOCS") or None


async def open_default_resources(stack: AsyncExitStack) -> DefaultResources:
    # Nothing to close here yet — LocalFiles holds no handle. A vector-store
    # or HTTP adapter would be entered on `stack` at this point instead.
    spec = pick_docs()
    if not spec:
        return DefaultResources()
    root = Path(spec).expanduser()
    if not root.is_dir():
        print(f"[documents: {root} is not a directory — source disabled]", file=sys.stderr)
        return DefaultResources()
    print(f"[documents: local files under {root}]", file=sys.stderr)
    return DefaultResources(documents=LocalFiles(root))


def default_capabilities(ctx: AgentContext) -> list[Capability]:
    llm = ctx.llm
    resources: DefaultResources = ctx.resources or DefaultResources()
    capabilities: list[Capability] = []
    if resources.documents is not None:
        capabilities.append(DocumentsCapability(llm, resources.documents))
    capabilities += [ResearchCapability(llm), AnalysisCapability(llm), CritiqueCapability(llm)]
    return capabilities


DEFAULT_AGENT = AgentDefinition(
    name="default",
    description="General-purpose analyst: documents, research, analysis and critique.",
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
    "LocalFiles",
    "ResearchCapability",
    "default_capabilities",
    "open_default_resources",
    "pick_docs",
]
