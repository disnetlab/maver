"""One whole incident against the backend the campaigns run on.

The unit suite covers the ports; the contract suite covers the backend against those
ports. Neither ran an incident, and the gap between them was wide enough to hold a
defect that made every incident on this backend fail at its first role: the runner
handed the store an empty start time, one backend accepted it and the other refused,
and the refusal released the environment with nothing submitted — which grades as an
agent that never answered.

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=kira postgres:17-alpine
    POSTGRES_TEST_DSN=postgresql://postgres:kira@127.0.0.1:55432/postgres \\
        uv run pytest -m integration tests/integration/test_incident_on_postgres.py
"""

import json
import os
import pathlib
import uuid
from typing import Any

import pytest

from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.orchestration.runner import Deployment, run_incident
from kira.settings.groups import MemorySettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

pytestmark = pytest.mark.integration

DSN = os.environ.get("POSTGRES_TEST_DSN", "")
PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}

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
SUBMISSIONS = {
    "rca": ("submit_root_cause", ROOT_CAUSE),
    "planning": ("submit_plan", PLAN),
    "execution": ("submit_execution_result", EXECUTION),
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "targetPort: 8081"


class FakeModel:
    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        name, arguments = SUBMISSIONS[caller]
        if caller == "execution" and self.callers.count("execution") == 1:
            name, arguments = "exec_kubectl_cmd_safely", {"cmd": "kubectl get svc user"}
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
async def test_an_incident_runs_end_to_end_on_postgres(tmp_path: pathlib.Path) -> None:
    schema = f"kira_e2e_{uuid.uuid4().hex[:8]}"
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
    )
    storage = build_storage(settings)
    try:
        model = FakeModel()
        deploy = Deployment(
            settings=settings,
            storage=storage,
            completion=model,
            profile=PROFILE,
            transport_for=lambda session_id: FakeTransport(),
            sink=ListSink(),
        )
        session = storage.sessions.create_session(ALERT)

        final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

        assert final["execution_result"] is not None
        stored = storage.sessions.get_session(session.session_id)
        assert stored is not None
        assert stored.run_status == "completed"
        assert stored.root_cause is not None and stored.plan is not None
        runs = storage.runs.runs_for(session.session_id)
        assert [r.agent_name for r in runs] == ["rca", "planning", "execution"]
        # The defect this file exists for: a run row has to carry a real start time,
        # and it is what the ordering above depends on.
        assert all(r.started_at for r in runs)
    finally:
        storage.close()
