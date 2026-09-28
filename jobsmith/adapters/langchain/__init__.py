"""The job ↔ conversation contract, for a LangChain agent (docs/design/core-v1.md,
step 9): a conversation that launches jobs on the engine is told of each of
their endings once, in its thread. Imports the engine and LangChain, nothing
of the bench."""
from .delivery import DELIVERED_CHANNEL, DeliveredState, JobDeliveryMiddleware, inject, told

__all__ = ["DELIVERED_CHANNEL", "DeliveredState", "JobDeliveryMiddleware", "inject", "told"]
