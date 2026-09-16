"""Lifecycle states for a persisted incident session and its distilled records.

A record's lifecycle state describes the standing of the lesson, not the operational
status of the run that produced it. The two are stored separately on purpose, because a
run that completed is not thereby a run that was right.

**Retrieval does not suppress a record for its state.** A record of an attempt that
failed tells a later incident what was already tried and what happened, which is exactly
the kind of thing an operator wants to know and the kind of thing a model will otherwise
retry. Excluding those records was measured and lost ground, so an incorrect precedent
is demoted rather than dropped: it ranks below an equally matching correct one and still
comes back. :mod:`maver.memory.lifecycle` carries the weights and the measurement.

That is a design position rather than an accident, and it is worth stating because
hiding the failures reads as the obvious default. The state exists so that a reader of a
record knows how much its claim was ever confirmed, not so that retrieval can bury it.
"""

from enum import StrEnum

__all__ = ["LifecycleState", "RunStatus"]


class LifecycleState(StrEnum):
    """How far a stored record's claim was ever confirmed.

    ``PENDING`` on creation, ``TENTATIVE_SUCCESS`` after a clean run, then
    ``HUMAN_VERIFIED`` or ``INVALIDATED`` once later evidence arrives.

    ``INVALIDATED`` reads as a verdict of wrongness and is not one. It marks a record
    whose claim did not hold up or could not be confirmed, closer to inconclusive than
    to false, and it is not a signal to bury the record. See the module docstring: a
    failed attempt is informative precedent.

    Note on ``HUMAN_VERIFIED``: the prototype declared this state and weighted it in
    ranking, but no code path ever wrote it, so the only way to reach it was a manual
    database update. MAVER must either wire it to a real operator action, which the
    approval surface makes natural, or drop it. That decision belongs to the memory and
    approval work, not here; the state is declared now because it is named in the
    published description of the system.
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
