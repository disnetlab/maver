import json
import pathlib
from collections.abc import Sequence

import pytest

from kira.memory import bm25, ranking
from kira.memory.ports import SearchScope, SessionMemory
from kira.memory.ranking import Bm25Ranker
from kira.memory.retrieval import Retriever
from kira.settings.groups import MemorySettings

FIXTURE = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_bm25.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parity
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["query"][:30])
def test_the_scorer_matches_the_prototype_exactly(case: dict) -> None:
    """The ranker is the measured mechanism, so its arithmetic is a held-constant input.

    The fixture is the prototype's own `rank` over the same documents, which is itself
    pinned against `rank_bm25.BM25Okapi`. A drift in the formula, the tokenisation, the
    negative-IDF floor or the tie-break shows up here.
    """
    ranked = bm25.rank(FIXTURE["docs"], case["query"])
    assert [i for i, _ in ranked] == [i for i, _ in case["ranked"]]
    for (_, got), (_, want) in zip(ranked, case["ranked"], strict=True):
        assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parity
@pytest.mark.parametrize(("text", "expected"), list(FIXTURE["tokenize"].items()))
def test_tokenisation_matches_the_prototype(text: str, expected: list[str]) -> None:
    assert bm25.tokenize(text) == expected


class FakeText:
    """A text backend that returns everything, so the retriever does the ranking."""

    def __init__(self, memories: list[SessionMemory]) -> None:
        self.memories = memories
        self.asked_for: list[str] = []

    def candidates(
        self,
        text: str,
        *,
        limit: int = 500,
        scope: SearchScope = "evidence",
        exclude: tuple[str, ...] = (),
    ) -> list[SessionMemory]:
        self.asked_for.append(text)
        return [m for m in self.memories if m.session_id not in exclude][:limit]

    def open_record(self, session_id: str) -> SessionMemory | None:
        return next((m for m in self.memories if m.session_id == session_id), None)

    def upsert_summary(self, memory: SessionMemory) -> None: ...

    def delete_summary(self, session_id: str) -> None: ...

    def count(self) -> int:
        return len(self.memories)


def memory(session_id: str, evidence: str) -> SessionMemory:
    return SessionMemory(
        session_id=session_id,
        evidence_text=evidence,
        lesson_text=f"lesson for {session_id}",
        incident_summary="",
        lifecycle_state="tentative_success",
        fault_category="",
        alert_name="",
        namespace="",
        affected_resource="",
    )


CORPUS = [memory(f"s{i}", doc) for i, doc in enumerate(FIXTURE["docs"])]


def test_retrieval_can_answer_that_nothing_is_comparable() -> None:
    """Ranked retrieval returns its top-k whether or not anything is relevant.

    On this corpus roughly 88% of what a role read at k=5 was irrelevant, and a role
    given five unrelated incidents treats them as evidence. The floor is what buys the
    ability to return nothing, which is a real answer.
    """
    retriever = Retriever(text=FakeText(CORPUS), ranker=Bm25Ranker(min_score_per_token=0.5))
    assert retriever.search("nothing whatsoever resembling these records") == []


def test_a_real_match_still_comes_back_above_the_floor() -> None:
    """The floor must not be so high that it rejects everything."""
    retriever = Retriever(text=FakeText(CORPUS), ranker=Bm25Ranker(min_score_per_token=0.5))
    hits = retriever.search("zero endpoints selector checkout")
    assert hits, "a directly matching record should clear the floor"
    assert hits[0].session_id == "s1"
    assert hits[0].score is not None


def test_a_zero_floor_disables_the_floor_rather_than_setting_it_at_zero() -> None:
    """A BM25 score is not bounded below, so zero is not the bottom of the scale.

    Okapi floors a negative IDF at a fraction of the mean IDF, which makes every score
    in a small or homogeneous pool negative — and one strongly matching precedent in a
    young corpus produces exactly that pool. Treating zero as a threshold would answer
    "nothing is comparable" for the case the mechanism exists to serve, and would give
    the arms that pin 0.0 to reproduce the prototype's dose a floor instead of none.
    """
    only_record = [memory("s0", "checkout pod OOMKilled memory limit exceeded")]
    query = "checkout pod OOMKilled memory limit exceeded"
    assert bm25.rank([only_record[0].evidence_text], query)[0][1] < 0

    disabled = Retriever(text=FakeText(only_record), ranker=Bm25Ranker(min_score_per_token=0.0))
    assert [hit.session_id for hit in disabled.search(query)] == ["s0"]

    enabled = Retriever(text=FakeText(only_record), ranker=Bm25Ranker(min_score_per_token=0.5))
    assert enabled.search(query) == []


def test_the_harness_exclusion_withholds_a_run_s_own_record() -> None:
    """Leave-one-out is supplied per cell by the evaluation harness, not inferred."""
    retriever = Retriever(text=FakeText(CORPUS), ranker=Bm25Ranker(min_score_per_token=0.0))
    hits = retriever.search("zero endpoints selector checkout", exclude=("s1",))
    assert "s1" not in [h.session_id for h in hits]


def test_an_empty_query_retrieves_nothing_rather_than_everything() -> None:
    backend = FakeText(CORPUS)
    assert Retriever(text=backend).search("") == []
    assert backend.asked_for == [], "an empty query should not reach the backend"


# --- the seam ---------------------------------------------------------------------


def test_another_ranker_reaches_retrieval_without_a_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring is the one step of retrieval that is a research question, and lexical
    ranking being the ceiling is a claim about this corpus rather than about retrieval.

    So a different ranker registers by name and everything around it — the lifecycle
    weight, the tie-break, the cut to k — is unchanged.
    """
    monkeypatch.setattr(ranking, "_REGISTRY", dict(ranking.registered_rankers()))

    @ranking.ranker("reverse")
    def _build(settings: MemorySettings) -> ranking.Ranker:
        class Reverse:
            def rank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
                return [(i, float(len(documents) - i)) for i in range(len(documents))]

        return Reverse()

    built = ranking.build_ranker(MemorySettings(ranker="reverse"))
    hits = Retriever(text=FakeText(CORPUS), ranker=built).search("anything at all")
    assert [hit.session_id for hit in hits] == [memory.session_id for memory in CORPUS]


def test_a_ranker_nothing_registered_is_a_startup_failure() -> None:
    """Not a fall back to the shipped one: a run reporting a ranker it did not retrieve
    with is a result that does not mean what it says."""
    with pytest.raises(ValueError, match="unknown ranker 'dense'; registered: bm25"):
        ranking.build_ranker(MemorySettings(ranker="dense"))


def test_the_relevance_decision_belongs_to_whatever_produced_the_score() -> None:
    """A threshold in the caller cannot serve BM25's unbounded scores, a cosine in
    [-1, 1] and a cross-encoder's logit at once, and one that tried would silently mean
    something different for each. A ranker returning fewer results than it was given
    documents is how it says the rest are not comparable."""
    admits_nothing = Bm25Ranker(min_score_per_token=1e9)
    assert admits_nothing.rank("payment service latency", ["payment service latency"]) == []
    assert Retriever(text=FakeText(CORPUS), ranker=admits_nothing).search("payment") == []
