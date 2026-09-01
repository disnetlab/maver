"""What a record's standing may become, and what that standing is worth when retrieved.

Two things live here because they are one fact read twice. The transition graph says
which verdicts a record can still receive; the weight table says how much a record in
each state counts once a later incident retrieves it.

The prototype had no transition guard at all: whatever a caller passed was written. That
is survivable where the field is decorative, and it is not survivable here, because a
record's standing is a research claim and an illegal write nobody rejects is an illegal
write nobody can find afterwards either.

Nothing here decides *which* state a finished run earns. That decision is
:func:`kira.orchestration.outcomes.resolve_outcome`, and it stays there, next to the
persistence that acts on it.
"""

from collections.abc import Mapping
from types import MappingProxyType

from kira.contracts.lifecycle import LifecycleState

__all__ = [
    "DEFAULT_LIFECYCLE_WEIGHT",
    "LIFECYCLE_WEIGHTS",
    "IllegalTransition",
    "check_transition",
    "resolve_weights",
    "weight_for",
]


class IllegalTransition(Exception):
    """A change of lifecycle state the graph does not allow.

    Carries both states because whoever reports the refusal — an HTTP conflict, a CLI
    message, an audit entry — has to be able to say what the record's standing actually
    is, not merely that the request was rejected.
    """

    def __init__(self, current: LifecycleState, target: LifecycleState) -> None:
        """Record the refused move.

        Args:
            current: The state the record holds.
            target: The state that was asked for.
        """
        super().__init__(f"a record in {current} cannot become {target}")
        self.current = current
        self.target = target


_LEGAL: Mapping[LifecycleState, frozenset[LifecycleState]] = MappingProxyType(
    {
        # Pending never reaches human_verified directly. A record only exists after
        # distillation, and distillation only runs on a completed session, so while a
        # session is pending there is nothing for an operator to have verified.
        LifecycleState.PENDING: frozenset(
            {
                LifecycleState.TENTATIVE_SUCCESS,
                LifecycleState.INVALIDATED,
                LifecycleState.HALTED,
            }
        ),
        LifecycleState.TENTATIVE_SUCCESS: frozenset(
            {LifecycleState.HUMAN_VERIFIED, LifecycleState.INVALIDATED}
        ),
        LifecycleState.HALTED: frozenset(
            {LifecycleState.HUMAN_VERIFIED, LifecycleState.INVALIDATED}
        ),
        # An operator can rehabilitate a record the pipeline wrote off, which is the
        # asymmetry the asynchronous-feedback framing requires: the automatic verdict is
        # provisional in both directions.
        LifecycleState.INVALIDATED: frozenset({LifecycleState.HUMAN_VERIFIED}),
        # And a human verdict is not permanent either.
        LifecycleState.HUMAN_VERIFIED: frozenset({LifecycleState.INVALIDATED}),
    }
)


def check_transition(current: LifecycleState, target: LifecycleState) -> bool:
    """Refuse a change of state the graph does not allow.

    Args:
        current: The state the record holds now.
        target: The state a caller wants to write.

    Returns:
        True when there is a transition to write. False when the record already holds
        the target state, which is a no-op rather than an error so that a job retried
        after a crash is safe to run again.

    Raises:
        IllegalTransition: When the graph does not allow the move. Callers record the
            refusal rather than dropping it, because a rejected verdict is as much a
            part of a record's history as an accepted one.
    """
    if current == target:
        return False
    if target not in _LEGAL[current]:
        raise IllegalTransition(current, target)
    return True


# Invalidated is demoted, not excluded, and that is the one value here that was measured
# rather than inherited. The offline campaign tested excluding invalidated records
# outright and found it a net loss: coverage fell 88% -> 72% and 90% -> 68%, and 5.7pp
# and 9.1pp fewer queries were served a true precedent first. The reason is that
# invalidation records the prior session's *outcome*, not the relevance of what it
# observed. An invalidated memory can still be the only same-family precedent in the
# corpus, and its observations can be perfectly accurate, because the session was marked
# invalidated when the remediation failed or was superseded, not when it saw the cluster
# wrong. Down-weighting keeps the coverage while demoting the rank; exclusion throws the
# coverage away.
#
# 0.3 is the value the E6 and oracle arms ran. The paper's main sweep did not set it at
# all, so invalidated records ranked there exactly like successful ones; an arm
# reproducing that campaign has to pin 1.0 through `memory.lifecycle_weights` rather
# than inherit it.
LIFECYCLE_WEIGHTS: Mapping[LifecycleState, float] = MappingProxyType(
    {
        LifecycleState.HUMAN_VERIFIED: 1.25,
        LifecycleState.TENTATIVE_SUCCESS: 1.0,
        LifecycleState.PENDING: 0.9,
        LifecycleState.HALTED: 0.9,
        LifecycleState.INVALIDATED: 0.3,
    }
)

DEFAULT_LIFECYCLE_WEIGHT = 0.9
"""What a state the table does not name is worth, and what an orphan record is worth.

Not zero. A record whose state cannot be read is a record whose standing is unknown, and
unknown standing is not a reason to withhold a precedent that matches.
"""


def resolve_weights(override: Mapping[str, float] | None = None) -> Mapping[LifecycleState, float]:
    """Apply a pinned weight table over the shipped one.

    Args:
        override: State name to weight, in the shape ``memory.lifecycle_weights``
            carries. None or empty leaves the shipped table alone.

    Returns:
        A weight for every state the shipped table names, with the override applied.

    Raises:
        ValueError: When a key names no lifecycle state. Ignoring it would leave a
            frozen arm believing it had pinned a ranking it had not, which is the exact
            failure the field exists to prevent.
    """
    if not override:
        return LIFECYCLE_WEIGHTS
    weights = dict(LIFECYCLE_WEIGHTS)
    for name, weight in override.items():
        try:
            state = LifecycleState(name)
        except ValueError as exc:
            known = ", ".join(member.value for member in LifecycleState)
            raise ValueError(
                f"unknown lifecycle state {name!r} in memory.lifecycle_weights; "
                f"expected one of {known}"
            ) from exc
        weights[state] = weight
    return MappingProxyType(weights)


def weight_for(
    state: str,
    weights: Mapping[LifecycleState, float] = LIFECYCLE_WEIGHTS,
) -> float:
    """Look up what a record in this state is worth to a ranker.

    Args:
        state: The state as stored, which is a bare string on the record and on an old
            enough record may name nothing this version knows about.
        weights: The table to read, from :func:`resolve_weights`.

    Returns:
        The weight, or :data:`DEFAULT_LIFECYCLE_WEIGHT` for a state the table does not
        name.
    """
    try:
        known = LifecycleState(state)
    except ValueError:
        return DEFAULT_LIFECYCLE_WEIGHT
    return weights.get(known, DEFAULT_LIFECYCLE_WEIGHT)
