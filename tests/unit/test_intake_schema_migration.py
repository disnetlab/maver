"""An existing version-4 session remains usable after all new store tables arrive."""

import sqlite3
from contextlib import closing
from pathlib import Path

from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.memory.backends.schema import STORE_SCHEMA_VERSION
from maver.memory.backends.sqlite import SqliteStore
from maver.orchestration.outcome_feedback import submit_feedback


def test_stamped_v4_store_migrates_to_combined_layout(tmp_path: Path) -> None:
    path = tmp_path / "v4.db"
    with closing(SqliteStore(path)) as store:
        old = store.create_session({"alertname": "existing"})
        store.update_artefacts(old.session_id, root_cause={"summary": "original diagnosis"})
        store.set_run_status(old.session_id, RunStatus.COMPLETED)
        store.set_state(old.session_id, LifecycleState.TENTATIVE_SUCCESS)
    with closing(sqlite3.connect(path)) as conn:
        for table in ("intake_receipts", "intake_quarantine", "outcome_feedback"):
            conn.execute(f"DROP TABLE {table}")
        conn.execute("UPDATE schema_version SET version = 4")
        conn.commit()

    with closing(SqliteStore(path)) as store:
        assert store.schema_version == STORE_SCHEMA_VERSION
        preserved = store.get_session(old.session_id)
        assert preserved is not None
        assert preserved.alert == {"alertname": "existing"}
        assert preserved.root_cause == {"summary": "original diagnosis"}
        admitted, created = store.open_idempotent_session(
            "http", "request-1", "hash-a", {"alertname": "new"}, {}
        )
        assert created
        assert store.get_intake_receipt("http", "request-1", "hash-a") == admitted
        poison = store.quarantine("kafka:group:alerts:0", "7", "bm90IEpTT04=", "invalid JSON")
        assert store.get_quarantine(poison.record_id) == poison
        feedback = submit_feedback(
            store,
            session_id=old.session_id,
            verdict="confirmed",
            decided_by="on-call",
            reason="observed recovery",
            evidence="healthy pods",
        )
        assert store.feedback_for(old.session_id) == [feedback]
    with closing(SqliteStore(path)) as reopened:
        assert reopened.get_session(old.session_id).root_cause == {"summary": "original diagnosis"}
        assert reopened.get_intake_receipt("http", "request-1", "hash-a") is not None
        assert reopened.get_quarantine(poison.record_id) is not None
        assert reopened.feedback_for(old.session_id) == [feedback]
