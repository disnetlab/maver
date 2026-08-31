import json
import pathlib

import pytest

from kira.memory import bm25
from kira.memory.ports import SearchHit, SearchScope, SessionMemory
from kira.memory.retrieval import Retriever, build_query

FIXTURE = json.loads(
    (pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_bm25.json").read_text(
        encoding="utf-8"
    )
)


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

    def scan_summaries(self, pattern, *, n_results=5, scope="evidence") -> list[SearchHit]:
        return []

    def get_session(self, session_id: str) -> SessionMemory | None:
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
    retriever = Retriever(text=FakeText(CORPUS), min_score_per_token=0.5)
    assert retriever.search("nothing whatsoever resembling these records") == []


def test_a_real_match_still_comes_back_above_the_floor() -> None:
    """The floor must not be so high that it rejects everything."""
    retriever = Retriever(text=FakeText(CORPUS), min_score_per_token=0.5)
    hits = retriever.search("zero endpoints selector checkout")
    assert hits, "a directly matching record should clear the floor"
    assert hits[0].session_id == "s1"
    assert hits[0].score is not None


def test_the_harness_exclusion_withholds_a_run_s_own_record() -> None:
    """Leave-one-out is supplied per cell by the evaluation harness, not inferred."""
    retriever = Retriever(text=FakeText(CORPUS), min_score_per_token=0.0)
    hits = retriever.search("zero endpoints selector checkout", exclude=("s1",))
    assert "s1" not in [h.session_id for h in hits]


def test_the_query_is_short_and_drops_boilerplate() -> None:
    """84.6% of the deployed query's body was lines carrying no incident-specific token.

    They cannot discriminate between records but do dominate a bag-of-words query, so
    the long query retrieved roughly six times fewer true precedents than a short one.
    """
    query = build_query(
        ["alert=PodCrashLooping", "namespace=shop"],
        [
            "frontend restarting",
            "  ",
            "no data",
            "unavailable",
            *[f"observation {i}" for i in range(20)],
        ],
    )
    lines = query.splitlines()
    assert lines[0] == "Current incident observations:"
    assert "- no data" not in lines
    assert "- unavailable" not in lines
    # two identifiers plus the eight-line observation budget
    assert len(lines) == 1 + 2 + 8


def test_an_empty_query_retrieves_nothing_rather_than_everything() -> None:
    backend = FakeText(CORPUS)
    assert Retriever(text=backend).search("") == []
    assert backend.asked_for == [], "an empty query should not reach the backend"
