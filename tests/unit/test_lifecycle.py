"""The transition graph, and what a state is worth once a later incident retrieves it."""

from collections.abc import Sequence

import pytest

from maver.contracts.lifecycle import LifecycleState as S
from maver.memory.lifecycle import (
    DEFAULT_LIFECYCLE_WEIGHT,
    IllegalTransition,
    check_transition,
    resolve_weights,
    weight_for,
)
from maver.memory.ports import SearchScope, SessionMemory
from maver.memory.ranking import Bm25Ranker
from maver.memory.retrieval import Retriever


@pytest.mark.parametrize(
    ("current", "target", "writes"),
    [
        (S.PENDING, S.TENTATIVE_SUCCESS, True),
        (S.PENDING, S.INVALIDATED, True),
        (S.PENDING, S.HALTED, True),
        (S.TENTATIVE_SUCCESS, S.HUMAN_VERIFIED, True),
        (S.HALTED, S.HUMAN_VERIFIED, True),
        # Both directions between the two verdicts. An operator can rehabilitate a
        # record the pipeline wrote off, and can withdraw a verdict already given.
        (S.INVALIDATED, S.HUMAN_VERIFIED, True),
        (S.HUMAN_VERIFIED, S.INVALIDATED, True),
        # Writing the state a record already holds is a no-op, so a job retried after a
        # crash is safe to run again.
        (S.TENTATIVE_SUCCESS, S.TENTATIVE_SUCCESS, False),
        (S.HUMAN_VERIFIED, S.HUMAN_VERIFIED, False),
    ],
)
def test_the_legal_transitions(current: S, target: S, writes: bool) -> None:
    assert check_transition(current, target) is writes


@pytest.mark.parametrize(
    ("current", "target"),
    [
        # Nothing exists to verify while a session is pending: a record is written by
        # distillation, and distillation only runs on a completed session.
        (S.PENDING, S.HUMAN_VERIFIED),
        (S.TENTATIVE_SUCCESS, S.PENDING),
        (S.INVALIDATED, S.TENTATIVE_SUCCESS),
    ],
)
def test_an_illegal_transition_raises_rather_than_being_written(current: S, target: S) -> None:
    """The prototype wrote whatever it was given, which made a wrong standing invisible."""
    with pytest.raises(IllegalTransition) as refused:
        check_transition(current, target)
    assert refused.value.current is current
    assert refused.value.target is target


def test_what_a_state_is_worth_to_the_ranker() -> None:
    """0.3 is the demotion the offline campaign validated against outright exclusion.

    A state the table does not name is worth the default rather than nothing: unknown
    standing is not a reason to withhold a precedent that matches.
    """
    assert weight_for(S.INVALIDATED) == 0.3
    assert weight_for(S.TENTATIVE_SUCCESS) == 1.0
    assert weight_for("distilled") == DEFAULT_LIFECYCLE_WEIGHT


def test_an_arm_can_pin_the_ranking_a_past_campaign_used() -> None:
    """The main sweep never set the weight, so invalidated records ranked at 1.0 there.

    Reproducing that campaign means saying so in the arm rather than inheriting whatever
    the shipped table happens to hold.
    """
    weights = resolve_weights({"invalidated": 1.0})
    assert weight_for(S.INVALIDATED, weights) == 1.0
    assert weight_for(S.PENDING, weights) == 0.9, "an override replaces an entry, not the table"


class Records:
    """A text backend returning everything, so the retriever does the ranking."""

    def __init__(self, memories: list[SessionMemory]) -> None:
        self.memories = memories

    def candidates(
        self,
        text: str,
        *,
        limit: int = 500,
        scope: SearchScope = "evidence",
        exclude: Sequence[str] = (),
    ) -> list[SessionMemory]:
        return list(self.memories)


def a_record(session_id: str, evidence: str) -> SessionMemory:
    return SessionMemory(
        session_id=session_id,
        evidence_text=evidence,
        lesson_text="",
        incident_summary="",
        lifecycle_state=S.TENTATIVE_SUCCESS.value,
        fault_category="",
        alert_name="",
        namespace="",
        affected_resource="",
    )


def test_a_demoted_record_still_comes_back_below_an_equal_one() -> None:
    """Demotion is not exclusion, which is the whole of the decision.

    The two matching records carry identical text, so the lexical score cannot separate
    them, and both the pool order and the tie-break on session id favour the demoted
    one. Only the lifecycle weight can produce this ordering, and the demoted record is
    still in the results. Its stored state says tentative_success: state is resolved as
    the candidate is ranked, so an operator's verdict lands on the next retrieval
    without anything being reindexed.
    """
    matching = "checkout deployment crashloopbackoff oomkilled memory limit"
    corpus = [
        a_record("demoted", matching),
        a_record("equal", matching),
        a_record("u1", "payment latency spike upstream timeout"),
        a_record("u2", "cart name resolution failure coredns"),
        a_record("u3", "search index rebuild disk pressure"),
    ]
    retriever = Retriever(
        text=Records(corpus),
        k=2,
        ranker=Bm25Ranker(min_score_per_token=0.0),
        states=lambda ids: {"demoted": S.INVALIDATED.value},
    )

    hits = retriever.search("checkout crashloopbackoff oomkilled")

    assert [hit.session_id for hit in hits] == ["equal", "demoted"]
    assert hits[1].score == pytest.approx((hits[0].score or 0.0) * 0.3)
    assert hits[1].metadata["state"] == S.INVALIDATED.value
