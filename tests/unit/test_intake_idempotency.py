"""Durable delivery identity takes precedence over same-alert reconciliation gating."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from maver.api.deps import Service
from maver.api.incidents import IncidentRequest, _request_hash, accept_incident
from maver.contracts.lifecycle import RunStatus
from maver.memory.backends.sqlite import SqliteStore


@pytest.mark.asyncio
async def test_exact_key_replay_returns_parked_session_while_new_intake_waits(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "sessions.db")
    body = IncidentRequest(alert={"alertname": "A", "namespace": "shop"})
    session, created = store.open_idempotent_session(
        "http",
        "delivery-1",
        _request_hash(body),
        body.alert,
        {"selection": {"name": "removed-pipeline"}},
    )
    assert created
    store.set_run_status(session.session_id, RunStatus.NEEDS_RECONCILIATION)
    service = Service(
        deployment=SimpleNamespace(
            storage=SimpleNamespace(sessions=store),
            settings=SimpleNamespace(customisation=None),
        ),
        schedule_resume=lambda _: None,
    )
    try:
        replay = await accept_incident(body, service, intake_id=("http", "delivery-1"))
        assert replay["session_id"] == session.session_id
        assert replay["started_at"] == session.started_at
        for intake_id in (("http", "delivery-2"), None):
            with pytest.raises(HTTPException) as refused:
                await accept_incident(body, service, intake_id=intake_id)
            assert refused.value.status_code == 409
            assert refused.value.headers == {"Retry-After": "30"}
        changed = IncidentRequest(alert={"alertname": "B"})
        with pytest.raises(HTTPException) as conflict:
            await accept_incident(changed, service, intake_id=("http", "delivery-1"))
        assert conflict.value.status_code == 409
        assert store.running_sessions() == []
    finally:
        store.close()
