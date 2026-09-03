"""The default agent: LLM-only, no external backend required.

research → analysis → critique gives the planner real DAG decisions and the
jobs real substance with nothing but an API key (or the fakes). This is what
`jobsmith chat` runs unless another agent is asked for.
"""
from ...core.capability import Capability
from ...core.deps import LLMClient
from ..base import AgentDefinition
from .analysis import AnalysisCapability
from .critique import CritiqueCapability
from .profile import DEFAULT_APP_PROFILE
from .research import ResearchCapability


def default_capabilities(llm: LLMClient) -> list[Capability]:
    return [ResearchCapability(llm), AnalysisCapability(llm), CritiqueCapability(llm)]


DEFAULT_AGENT = AgentDefinition(
    name="default",
    description="General-purpose analyst: research, analysis and critique, LLM-only.",
    capabilities=default_capabilities,
    profile=DEFAULT_APP_PROFILE,
)


__all__ = [
    "DEFAULT_AGENT",
    "AnalysisCapability",
    "CritiqueCapability",
    "ResearchCapability",
    "default_capabilities",
]
