"""Conversational front layer: chat by default, background jobs on complexity.

The chat agent is a LangGraph prebuilt ReAct agent whose tools are the
JobManager use-cases — running a task is just a tool call. It runs in the
turn and is promoted to the background by the clock, never by a prediction
(#83). This layer deliberately uses LangChain models (tool-format handling
per provider) while the job engine underneath keeps the framework's minimal
LLMClient protocol.
"""
from .runner import (
    ChatEvent,
    ChatRunner,
    JobPlanned,
    JobStarted,
    Message,
    Proposal,
    Token,
    ToolFinished,
    ToolStarted,
)
from .session import DEFAULT_CHAT_SYSTEM_PROMPT, ChatSession
from .tools import make_job_tools

__all__ = [
    "ChatEvent",
    "ChatRunner",
    "ChatSession",
    "DEFAULT_CHAT_SYSTEM_PROMPT",
    "JobPlanned",
    "JobStarted",
    "Message",
    "Proposal",
    "Token",
    "ToolFinished",
    "ToolStarted",
    "make_job_tools",
]
