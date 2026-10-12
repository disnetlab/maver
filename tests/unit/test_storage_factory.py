"""Opening the configured stores, once, at startup."""

import logging
import pathlib
import sys

import pytest

from maver.memory.factory import build_storage
from maver.memory.ports import SessionMemory
from maver.memory.ranking import Bm25Ranker
from maver.settings.groups import MemorySettings, ServerSettings
from maver.settings.root import Settings


def a_record(session_id: str, evidence: str) -> SessionMemory:
    return SessionMemory(
        session_id=session_id,
        evidence_text=evidence,
        lesson_text="",
        incident_summary="",
        lifecycle_state="tentative_success",
        fault_category="",
        alert_name="",
        namespace="",
        affected_resource="",
    )


def settings_for(tmp_path: pathlib.Path, **memory: object) -> Settings:
    return Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(**memory),  # type: ignore[arg-type]
    )


def test_the_embedded_database_lives_under_this_deployment_s_workspace(
    tmp_path: pathlib.Path,
) -> None:
    """Two deployments sharing a machine must not share a corpus: the workspace is
    what separates them, so the database has to follow it rather than a fixed path."""
    stack = build_storage(settings_for(tmp_path))
    try:
        session = stack.sessions.create_session({"alertname": "HighErrorRate"})
        assert stack.sessions.get_session(session.session_id) is not None
        assert (tmp_path / "memory" / "default.sqlite").is_file()
    finally:
        stack.close()


def test_the_retrieval_floor_reaches_the_retriever(tmp_path: pathlib.Path) -> None:
    """The floor decides how many precedents a run is served, and that dose is what an
    arm's memory result is divided by. Left at the constructor default, a campaign that
    ran without a floor had no way to say so and could not be reproduced."""
    stack = build_storage(settings_for(tmp_path, min_score_per_token=0.0))
    try:
        assert isinstance(stack.retriever.ranker, Bm25Ranker)
        assert stack.retriever.ranker.min_score_per_token == 0.0
    finally:
        stack.close()


def test_historical_policy_reaches_the_retriever(tmp_path: pathlib.Path) -> None:
    stack = build_storage(settings_for(tmp_path, retrieval_policy="historical"))
    try:
        assert stack.retriever.retrieval_policy == "historical"
        stack.index.upsert_summary(a_record("s", "pod failure"))
        assert stack.retriever.search("pod failure")[0].score == 1.0
    finally:
        stack.close()


def test_the_candidate_pool_reaches_the_backend_and_an_outgrown_one_says_so(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A backend fills the pool in write order, so a corpus past the ceiling loses its
    newest records from every retrieval — the ones a deployment has most recently
    learned. Left at a constructor default nothing could reach, an adopter crossed the
    ceiling and the store went quiet with no error and no way to raise it."""
    query = "checkout deployment OOMKilled memory limit exceeded"
    corpus = [a_record(f"s{i}", "checkout deployment unrelated symptom") for i in range(3)]
    corpus.append(a_record("newest", query))

    def search_with(pool: int) -> list[str]:
        stack = build_storage(settings_for(tmp_path, candidate_pool=pool, min_score_per_token=0.0))
        try:
            for record in corpus:
                stack.index.upsert_summary(record)
            return [hit.session_id for hit in stack.retriever.search(query)]
        finally:
            stack.close()

    with caplog.at_level(logging.WARNING, logger="maver.memory.retrieval"):
        assert "newest" not in search_with(3)
    assert "raise memory.candidate_pool" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="maver.memory.retrieval"):
        assert search_with(10)[0] == "newest"
    assert "raise memory.candidate_pool" not in caplog.text


def test_an_unavailable_backend_fails_at_startup_and_says_what_to_install(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment that cannot persist has nothing useful to do, so it should not
    discover that on its first incident.

    The driver is hidden rather than assumed absent: this suite runs with the extra
    installed, and the failure being pinned is the one an install without it gets.
    """
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.delitem(sys.modules, "maver.memory.backends.postgres", raising=False)

    with pytest.raises(ValueError, match=r"maver\[postgres\]"):
        build_storage(settings_for(tmp_path, backend="postgres", postgres_dsn="postgresql://x/y"))
