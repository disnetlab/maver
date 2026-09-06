"""The embedded backend against the shared contract.

Nothing SQLite-specific is asserted here on purpose. The contract is the test budget;
a second backend is done when it passes the same file.
"""

import contextlib
import dataclasses
import json
import pathlib
import sqlite3
from collections.abc import Iterator

import pytest
from memory_contract import MemoryStoreContract, OperationalStore, a_record

from kira.memory.backends.sqlite import SqliteRecordIndex, SqliteStore
from kira.memory.ports import TextSearch


class TestSqliteBackend(MemoryStoreContract):
    @pytest.fixture
    def sqlite_store(self) -> Iterator[SqliteStore]:
        store = SqliteStore(":memory:")
        yield store
        store.close()

    @pytest.fixture
    def store(self, sqlite_store: SqliteStore) -> OperationalStore:
        return sqlite_store

    @pytest.fixture
    def records(self, sqlite_store: SqliteStore) -> TextSearch:
        return sqlite_store.records


def test_a_file_database_is_created_and_survives_reopening(tmp_path: pathlib.Path) -> None:
    """The one thing the contract cannot cover, because it runs against ":memory:"."""
    path = tmp_path / "nested" / "kira.db"

    store = SqliteStore(path)
    session = store.create_session({"name": "PodCrashLooping"})
    store.records.upsert_summary(a_record(session.session_id, evidence="oomkilled"))
    store.close()

    assert path.exists()
    reopened = SqliteStore(path)
    try:
        read = reopened.get_session(session.session_id)
        assert read is not None
        assert read.alert == {"name": "PodCrashLooping"}
        assert reopened.records.count() == 1
    finally:
        reopened.close()


# The FTS5 table as it was before the shared normaliser: one tokeniser, no stemming, no
# stopwords, and the raw text indexed directly. Written out here rather than imported so
# that changing the current schema cannot quietly change what "the old shape" means.
_OLD_SCHEMA = """
CREATE VIRTUAL TABLE records_fts USING fts5(
    session_id UNINDEXED,
    evidence_text,
    lesson_text,
    payload_json UNINDEXED
);
"""


def test_a_database_written_before_the_normaliser_is_reindexed_on_open(
    tmp_path: pathlib.Path,
) -> None:
    """An existing corpus has to answer the new query, not be silently left behind."""
    written = a_record(
        "old", evidence="the checkout pods were crashing", lesson="raised the memory limit"
    )
    payload = {
        field.name: getattr(written, field.name)
        for field in dataclasses.fields(written)
        if field.name not in ("session_id", "evidence_text", "lesson_text")
    }
    path = tmp_path / "old.db"
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.executescript(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO records_fts VALUES (?, ?, ?, ?)",
            (written.session_id, written.evidence_text, written.lesson_text, json.dumps(payload)),
        )
        conn.commit()

    index = SqliteRecordIndex(path)
    try:
        assert [m.session_id for m in index.candidates("pod crash")] == ["old"]
        assert [m.session_id for m in index.candidates("raising limits", scope="lesson")] == ["old"]
        assert index.candidates("the and of") == []

        assert index.open_record("old") == written, "the record itself is carried through"
    finally:
        index.close()

    # Idempotent: a second open finds the new shape and rebuilds nothing.
    reopened = SqliteRecordIndex(path)
    try:
        assert reopened.count() == 1
        assert [m.session_id for m in reopened.candidates("pod crash")] == ["old"]
    finally:
        reopened.close()
