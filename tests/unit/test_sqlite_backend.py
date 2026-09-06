"""The embedded backend against the shared contract.

Nothing SQLite-specific is asserted here on purpose. The contract is the test budget;
a second backend is done when it passes the same file.
"""

import contextlib
import dataclasses
import json
import logging
import pathlib
import sqlite3
from collections.abc import Callable, Iterator

import pytest
from memory_contract import MemoryStoreContract, OperationalStore, a_record

from kira.memory.backends.schema import STORE_SCHEMA_VERSION
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

    @pytest.fixture
    def database(self, tmp_path: pathlib.Path) -> pathlib.Path:
        return tmp_path / "kira.db"

    @pytest.fixture
    def reopen(self, database: pathlib.Path) -> Callable[[], SqliteStore]:
        return lambda: SqliteStore(database)

    @pytest.fixture
    def stamp_version(self, database: pathlib.Path) -> Callable[[int], None]:
        def stamp(version: int) -> None:
            with contextlib.closing(sqlite3.connect(database)) as conn:
                conn.execute("UPDATE schema_version SET version = ?", (version,))
                conn.commit()

        return stamp


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
        "old",
        evidence="the checkout pods were crashing, RunContainerError/CrashLoopBackOff",
        lesson="raised the memory limit",
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
        assert index.schema_version == STORE_SCHEMA_VERSION
        assert [m.session_id for m in index.candidates("pod crash")] == ["old"]
        assert [m.session_id for m in index.candidates("raising limits", scope="lesson")] == ["old"]
        assert index.candidates("the and of") == []
        assert [m.session_id for m in index.candidates("crashloopbackoff")] == ["old"]

        assert index.open_record("old") == written, "the record itself is carried through"
    finally:
        index.close()

    # Idempotent: a second open finds the new shape and rebuilds nothing.
    reopened = SqliteRecordIndex(path)
    try:
        reopened.upsert_summary(a_record("new", evidence="coredns resolution failures"))
        assert reopened.count() == 2
        assert [m.session_id for m in reopened.candidates("pod crash")] == ["old"]
        assert [m.session_id for m in reopened.candidates("resolve failure")] == ["new"]
    finally:
        reopened.close()

    assert _stamped_versions(path) == [STORE_SCHEMA_VERSION], "one row, and it is current"
    fresh = tmp_path / "fresh.db"
    with contextlib.closing(SqliteRecordIndex(fresh)):
        pass
    assert _index_columns(path) == _index_columns(fresh), (
        "a migrated index is the current layout, not a shape of its own"
    )


def test_an_unstamped_database_in_the_current_shape_is_stamped_and_left_alone(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Not every database without a version is an old one.

    One written by the build that migrated the index but did not yet stamp anything is
    already current, and reading its shape has to say so. Rebuilding it would be correct
    and would still be a cost taken for nothing.
    """
    path = tmp_path / "unstamped.db"
    with contextlib.closing(SqliteStore(path)) as store:
        store.records.upsert_summary(a_record("s1", evidence="the checkout pods were crashing"))
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP TABLE schema_version")
        conn.commit()

    with (
        caplog.at_level(logging.INFO, logger="kira.memory.backends.schema"),
        contextlib.closing(SqliteStore(path)) as reopened,
    ):
        assert reopened.schema_version == STORE_SCHEMA_VERSION
        assert [m.session_id for m in reopened.records.candidates("pod crash")] == ["s1"]
    assert not [record for record in caplog.records if "migrating" in record.getMessage()]
    assert _stamped_versions(path) == [STORE_SCHEMA_VERSION]


def _stamped_versions(path: pathlib.Path) -> list[int]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return [row[0] for row in conn.execute("SELECT version FROM schema_version")]


def _index_columns(path: pathlib.Path) -> list[str]:
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return [row[1] for row in conn.execute("PRAGMA table_info(records_fts)")]
