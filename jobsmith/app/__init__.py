"""The global agent — jobsmith as a product, not just a framework.

This package is the default composition root: provider selection, a
domain-neutral capability pack (LLM-only), the generic chat REPL, and the
`python -m jobsmith` entrypoints. Domain examples (see jobsmith/examples/)
reuse these pieces and only supply their own capabilities/profile.
"""
from .agent import AgentApp, build_app
from .capabilities import default_capabilities

__all__ = ["AgentApp", "build_app", "default_capabilities"]
