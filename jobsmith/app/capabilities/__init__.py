"""Default capability pack: LLM-only, no external backend required.

research → analysis → critique gives the planner real DAG decisions and the
jobs real substance with nothing but an API key (or the fakes).
"""
from ...core.capability import Capability
from ...core.deps import LLMClient
from .analysis import AnalysisCapability
from .critique import CritiqueCapability
from .research import ResearchCapability


def default_capabilities(llm: LLMClient) -> list[Capability]:
    return [ResearchCapability(llm), AnalysisCapability(llm), CritiqueCapability(llm)]


__all__ = [
    "AnalysisCapability",
    "CritiqueCapability",
    "ResearchCapability",
    "default_capabilities",
]
