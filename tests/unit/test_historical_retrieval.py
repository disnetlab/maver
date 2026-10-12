"""Synthetic policy checks, without claiming historical database/vector parity."""

from dataclasses import replace

import pytest
from memory_contract import a_record

from maver.contracts.lifecycle import LifecycleState
from maver.memory import bm25
from maver.memory.ports import SearchScope, SessionMemory
from maver.memory.ranking import Bm25Ranker
from maver.memory.retrieval import Retriever
from maver.settings.groups import MemorySettings


class Pool:
    def __init__(self, records: list[SessionMemory]) -> None:
        self.records = records
        self.calls: list[tuple[int, SearchScope, tuple[str, ...]]] = []

    def candidates(self, text, *, limit=500, scope="evidence", exclude=()):
        self.calls.append((limit, scope, tuple(exclude)))
        return [record for record in self.records if record.session_id not in exclude][:limit]


def record(sid: str, **fields) -> SessionMemory:
    defaults = {"evaluation_label": "unknown", "lifecycle_state": "tentative_success"}
    defaults.update(fields)
    return a_record(sid, evidence="pod failure", **defaults)


def historical(pool: Pool, **kwargs) -> Retriever:
    return Retriever(
        text=pool,
        retrieval_policy="historical",
        ranker=Bm25Ranker(min_score_per_token=0),
        **kwargs,
    )


def ids(hits) -> list[str]:
    return [hit.session_id for hit in hits]


def test_current_remains_weighted_raw_bm25_and_ignores_query_metadata() -> None:
    records = [record("b", lifecycle_state="invalidated"), record("a")]
    pool = Pool(records)
    current = Retriever(text=pool, k=2, ranker=Bm25Ranker(min_score_per_token=0))
    raw = bm25.rank([r.evidence_text for r in records], "pod failure")
    scores = {records[i].session_id: score * (0.3 if i == 0 else 1) for i, score in raw}
    hits = current.search("pod failure", query_metadata={"namespace": records[0].namespace})
    assert ids(hits) == sorted(scores, key=lambda sid: (-scores[sid], sid))
    assert {h.session_id: h.score for h in hits} == pytest.approx(scores)
    assert current.retrieval_policy == MemorySettings().retrieval_policy == "current"
    current.search("pod failure", exclude=["a"])
    assert pool.calls[-1] == (500, "evidence", ("a",))


def test_historical_ties_use_session_ids_then_reciprocal_rank_and_fresh_states() -> None:
    pool = Pool([record("c"), record("b"), record("a")])
    got = historical(pool, states=lambda _: {"a": "invalidated"}).search("pod failure")
    assert ids(got) == ["b", "c", "a"]
    assert [h.score for h in got] == pytest.approx([0.5, 1 / 3, 0.3])
    assert got[-1].metadata["state"] == "invalidated"
    assert set(got[-1].metadata) == {
        "state",
        "fault_category",
        "alert_name",
        "namespace",
        "affected_resource",
    }
    assert LifecycleState.INVALIDATED in historical(pool).weights
    assert historical(pool).weights[LifecycleState.INVALIDATED] == 0.3


def test_exact_identifier_boost_is_additive_after_weighting_and_capped() -> None:
    matched = record(
        "d",
        alert_name=" Alert ",
        namespace=" shop ",
        fault_category=" fault ",
        affected_resource=" shop/Pod/x ",
    )
    retriever = historical(Pool([record("c"), matched, record("b"), record("a")]))
    metadata = {
        "alert_name": "Alert",
        "namespace": "shop",
        "fault_category": "fault",
        "affected_resource": "shop/Pod/x",
        "evaluation_label": "unknown",
        "service": "x",
    }
    hits = retriever.search("pod failure", query_metadata=metadata)
    assert ids(hits) == ["a", "b", "d", "c"]
    assert hits[2].score == pytest.approx(0.25 + 0.15)
    hits = retriever.search("pod failure", query_metadata={"alert_name": "alert"})
    assert ids(hits) == ["a", "b", "c", "d"]
    assert hits[-1].score == 0.25
    hits = retriever.search("pod failure", query_metadata={"namespace": " shop "})
    assert hits[-1].score == pytest.approx(0.30)


def test_role_lanes_prefer_evaluated_and_corrected_over_unverified() -> None:
    records = [
        record("a", lifecycle_state="human_verified"),  # unknown grade overrides state
        replace(record("b"), evaluation_label="failure"),
        replace(record("c"), evaluation_label="success"),
        record("d", authoritative_correction=True, evaluation_diagnosis_success=False),
        record("e", evaluation_diagnosis_success=False, corrected_cause="model guess"),
    ]
    assert ids(historical(Pool(records), k=3).search("pod failure")) == ["c", "d", "b"]
    # A correction string and a false verdict do not establish external authority.
    records[3].authoritative_correction = False
    assert ids(historical(Pool(records), k=3).search("pod failure")) == ["c", "b", "a"]


def test_lifecycle_fallback_and_evaluated_remainder_fill_short_positive_lane() -> None:
    records = [
        replace(record("a", lifecycle_state="human_verified"), evaluation_label=""),
        replace(record("b", lifecycle_state="halted"), evaluation_label=""),
        replace(record("c", lifecycle_state="invalidated"), evaluation_label=""),
        record("d"),
    ]
    assert ids(historical(Pool(records), k=3).search("pod failure")) == ["a", "b", "c"]


def test_exclusion_happens_after_lane_selection_and_inflates_requested_depth() -> None:
    records = [
        replace(record("a"), evaluation_label="failure"),
        *[replace(record(sid), evaluation_label="success") for sid in ("b", "c", "d")],
        record("e"),
    ]
    pool = Pool(records)
    retriever = historical(pool, k=3)
    assert ids(retriever.search("pod failure")) == ["b", "c", "a"]
    hits = retriever.search("pod failure", exclude=["a", "a"])
    assert ids(hits) == ["b", "c", "d"]
    assert [hit.score for hit in hits] == pytest.approx([1 / 2, 1 / 3, 1 / 4])
    assert pool.calls[-1] == (500, "evidence", ())


def test_bm25_scores_full_pool_then_truncates_raw_lane_before_role_selection() -> None:
    records = [record(f"s{i:03}") for i in range(500)]
    records[55].evaluation_label = "success"
    pool = Pool(records)
    retriever = historical(pool, k=3)
    assert ids(retriever.search("pod failure")) == ["s000", "s001", "s002"]
    # Seven requested records widen the text lane to 70, admitting the low-ranked
    # positive. Exclusions still participate in scoring and in the lane quota.
    hits = retriever.search("pod failure", exclude=[f"s{i:03}" for i in range(4)])
    assert ids(hits) == ["s055", "s004", "s005"]
    assert hits[0].score == pytest.approx(1 / 56)
    assert pool.calls[-1][0] == 500
    retriever.search("pod failure", exclude=[f"x{i}" for i in range(60)])
    assert pool.calls[-1][0] == 630


def test_floor_uses_raw_magnitude_and_original_ascii_runs_not_scoring_tokens() -> None:
    # Unicode Kelvin signs lowercase to ASCII k for BM25 but are not original ASCII
    # query runs. Punctuation separates the prototype's denominator into two runs.
    query = "\u212a\u212a\u212a-pod.failure"
    pool = Pool([record("a")])
    raw = abs(bm25.rank(["pod failure"], query)[0][1])
    assert len(bm25.tokenize(query)) == 3
    retriever = Retriever(
        text=pool, retrieval_policy="historical", ranker=Bm25Ranker(min_score_per_token=raw / 2.5)
    )
    assert ids(retriever.search(query)) == ["a"]
    retriever = replace(retriever, ranker=Bm25Ranker(min_score_per_token=raw))
    assert retriever.search("pod failure", query_metadata={"namespace": "different"}) == []
    assert ids(
        retriever.search("pod failure", query_metadata={"namespace": pool.records[0].namespace})
    ) == ["a"]


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_never_fetches_or_leaks_excluded_records(k: int) -> None:
    pool = Pool([record("a")])
    assert historical(pool, k=k).search("pod", exclude=["a"]) == []
    assert pool.calls == []


def test_unsupported_historical_ranker_is_rejected_at_settings_and_direct_seam() -> None:
    with pytest.raises(ValueError, match=r"requires memory.ranker='bm25'"):
        MemorySettings(retrieval_policy="historical", ranker="dense")
    with pytest.raises(ValueError, match="requires Bm25Ranker"):
        Retriever(text=Pool([]), retrieval_policy="historical", ranker=object())
