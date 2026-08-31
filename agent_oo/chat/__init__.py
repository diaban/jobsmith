"""Conversational front layer: chat by default, background jobs on complexity.

The chat agent is a LangGraph prebuilt ReAct agent whose tools are the
JobManager use-cases — launching a job is just a tool call, gated by a
human-in-the-loop interrupt. This layer deliberately uses LangChain models
(tool-format handling per provider) while the job engine underneath keeps
the framework's minimal LLMClient protocol.
"""
from .session import DEFAULT_CHAT_SYSTEM_PROMPT, ChatSession
from .tools import make_job_tools

__all__ = ["ChatSession", "DEFAULT_CHAT_SYSTEM_PROMPT", "make_job_tools"]
