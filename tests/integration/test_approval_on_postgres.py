"""The whole approval cycle against the backend the campaigns run on.

The approval mechanism is load-bearing for what the system claims about supervised
autonomy, and every part of it that persists — the parked run status, the checkpoint,
the request row, the decision — goes through a store. The unit suite exercises all of it
on SQLite. That is the arrangement that already hid one defect: a value SQLite accepted
and PostgreSQL refused, in a code path no test ran against PostgreSQL.

A checkpoint is stored as JSON and read back as a validated model with ``extra="forbid"``,
so it is exactly the kind of thing that survives a round trip on one backend and not the
other, and it is only ever read on a path that runs minutes or hours after the write.

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=maver postgres:17-alpine
    POSTGRES_TEST_DSN=postgresql://postgres:maver@127.0.0.1:55432/postgres \\
        uv run pytest -m integration tests/integration/test_approval_on_postgres.py
"""

import json
import os
import pathlib
import uuid
from typing import Any

import pytest

from maver.audit.journal import ListSink
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.middleware.authority import ApprovalPending
from maver.middleware.human_approval import authorities
from maver.orchestration.runner import Deployment, resume_after_decision, run_incident
from maver.settings.authority import AuthoritySettings
from maver.settings.groups import MemorySettings, ServerSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings

pytestmark = pytest.mark.integration

DSN = os.environ.get("POSTGRES_TEST_DSN", "")
PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}
PATCH = {"cmd": "kubectl patch svc user -p '{}'"}

ROOT_CAUSE: dict[str, Any] = {
    "fault_type": "misconfiguration",
    "affected_resource": "shop/Service/user",
    "confidence": 0.8,
    "evidence": ["endpoints empty for svc/user"],
    "summary": "the user Service targets a port nothing listens on",
}
PLAN: dict[str, Any] = {
    "steps": [
        {
            "action": "patch",
            "target": "svc/user",
            "params": {"targetPort": 8080},
            "justification": "repoint the service",
        }
    ],
    "expected_outcome": "svc/user has endpoints",
    "confidence": 0.8,
}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched; endpoints appeared"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}

COMMANDS: list[list[str]] = []
"""Every argv the fake cluster ran. Reads are expected — the diagnosing role snapshots
the namespace — so a test asserting a write did not happen must say so specifically."""


def writes() -> list[list[str]]:
    """The mutating commands only."""
    return [argv for argv in COMMANDS if argv and argv[0] == "patch"]


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        COMMANDS.append(list(argv))
        return "service/user patched"


class FakeModel:
    """Diagnoses, plans, writes once — which parks — then reports."""

    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        if caller == "rca":
            name, arguments = "submit_root_cause", ROOT_CAUSE
        elif caller == "planning":
            name, arguments = "submit_plan", PLAN
        elif self.callers.count("execution") == 1:
            name, arguments = "exec_kubectl_cmd_safely", PATCH
        else:
            name, arguments = "submit_execution_result", EXECUTION
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{len(self.callers)}",
                    name=name,
                    arguments=arguments,
                    raw_arguments=json.dumps(arguments),
                ),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            duration_ms=1,
            attempts=1,
        )


@pytest.mark.skipif(not DSN, reason="set POSTGRES_TEST_DSN to run this")
async def test_an_incident_parks_and_resumes_on_postgres(tmp_path: pathlib.Path) -> None:
    schema = f"maver_hitl_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(
            backend="postgres",
            postgres_dsn=DSN,
            postgres_schema=schema,
            enabled=False,
            learning_enabled=False,
        ),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(mode="prompt", timeout_s=1800),
    )
    storage = build_storage(settings)
    COMMANDS.clear()
    try:
        deploy = Deployment(
            settings=settings,
            storage=storage,
            completion=FakeModel(),
            profile=PROFILE,
            transport_for=lambda session_id: FakeTransport(),
            authorities=authorities(
                settings.authority,
                storage.approvals,
                lambda request_id, expires_at: None,
            ),
            sink=ListSink(),
        )
        session = storage.sessions.create_session(ALERT)

        with pytest.raises(ApprovalPending):
            await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

        parked = storage.sessions.get_session(session.session_id)
        assert parked is not None
        assert parked.run_status == "awaiting_human"
        assert parked.awaiting_request_id is not None
        # Written as JSON and read back through a model that forbids unknown keys. A
        # field the store round-trips differently shows up here and nowhere else.
        assert parked.checkpoint is not None
        assert parked.checkpoint["stage"] == "execution"
        assert parked.checkpoint["mode"] == "multi"
        assert writes() == []

        request_id = parked.awaiting_request_id
        storage.approvals.decide(
            request_id, status="approved", decided_by="an operator", reason="ok"
        )

        assert await resume_after_decision(request_id, deploy) == "success"

        assert writes() == [["patch", "svc", "user", "-p", "{}"]]
        finished = storage.sessions.get_session(session.session_id)
        assert finished is not None
        assert finished.run_status == "completed"
        assert finished.execution_result is not None
        # Two attempts on one incident: the one that stopped for a decision, and the one
        # that carried on. A row left open would be indistinguishable from a hang.
        runs = storage.runs.runs_for(session.session_id)
        assert [r.agent_name for r in runs] == ["rca", "planning", "execution", "execution"]
        assert [r.status for r in runs][-2:] == ["awaiting_human", "succeeded"]
    finally:
        storage.close()
