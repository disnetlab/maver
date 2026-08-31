"""Lifecycle states and the packet handed to an offline learning pass."""

from kira.contracts import LearningPacket, LifecycleState, RunStatus


def test_the_published_lifecycle_states_all_exist() -> None:
    """A record starts pending, earns tentative success, then is verified or invalidated."""
    assert [s.value for s in LifecycleState] == [
        "pending",
        "tentative_success",
        "human_verified",
        "invalidated",
        "halted",
    ]


def test_lesson_standing_and_run_status_are_separate_vocabularies() -> None:
    """A run that completed is not thereby a run that was right.

    They share the literal "pending" and nothing else: operational progress belongs to
    one and the standing of the lesson to the other.
    """
    lifecycle = {s.value for s in LifecycleState}
    runs = {s.value for s in RunStatus}
    assert lifecycle & runs == {"pending"}
    assert "awaiting_human" in runs and "awaiting_human" not in lifecycle
    assert "tentative_success" in lifecycle and "tentative_success" not in runs


def test_a_packet_needs_only_the_session_identity_and_its_alert() -> None:
    packet = LearningPacket(
        session_id="s-1", lifecycle_state=LifecycleState.PENDING, alert={"name": "HighErrorRate"}
    )
    assert packet.root_cause is None
    assert packet.external_feedback is None


def test_the_packet_can_carry_the_verification_verdict() -> None:
    """The prototype's packet omitted it, so learning judged an outcome without it."""
    assert "verify_result" in LearningPacket.model_fields
    packet = LearningPacket(
        session_id="s-1",
        lifecycle_state=LifecycleState.TENTATIVE_SUCCESS,
        alert={},
        verify_result={"recovered": True, "verdict": "recovered"},
    )
    assert packet.verify_result is not None
