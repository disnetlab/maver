"""The embedded backend against the shared contract.

Nothing SQLite-specific is asserted here on purpose. The contract is the test budget;
a second backend is done when it passes the same file.
"""

import pathlib
from collections.abc import Iterator

import pytest
from memory_contract import MemoryStoreContract, OperationalStore, a_record

from kira.memory.backends.sqlite import SqliteStore
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
