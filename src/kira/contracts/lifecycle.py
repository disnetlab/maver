"""Lifecycle states for a persisted incident session and its distilled records.

A record's lifecycle state is what retrieval weights, so that an invalidated lesson
cannot silently steer a later incident. The state describes the standing of the
lesson, not the operational status of the run that produced it; the two are stored
separately on purpose, because a run that completed is not thereby a run that was
right.
"""

from enum import StrEnum

__all__ = ["LifecycleState", "RunStatus"]


class LifecycleState(StrEnum):
    """How much confidence a stored record has earned.

    ``PENDING`` on creation, ``TENTATIVE_SUCCESS`` after a clean run, then
    ``HUMAN_VERIFIED`` or ``INVALIDATED`` once later evidence arrives.

    Note on ``HUMAN_VERIFIED``: the prototype declared this state and weighted it in
    ranking, but no code path ever wrote it, so the only way to reach it was a manual
    database update. Kira must either wire it to a real operator action, which the
    approval surface makes natural, or drop it and its ranking weight. That decision
    belongs to the memory and approval work, not here; the state is declared now
    because it is named in the published description of the system.
    """

    PENDING = "pending"
    TENTATIVE_SUCCESS = "tentative_success"
    HUMAN_VERIFIED = "human_verified"
    INVALIDATED = "invalidated"
    HALTED = "halted"
    """A role stopped the pipeline deliberately, with a stated reason."""


class RunStatus(StrEnum):
    """Where an incident run is operationally, independent of its lesson's standing."""

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
