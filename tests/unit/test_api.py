"""The HTTP surface, and the contract an outside harness already depends on.

The intake shape here is held: an evaluation harness outside this repository posts to
it and polls it, and it was written against the prototype. So these tests pin the wire,
not the implementation — the exact keys, the old stage word, and what a poller sees when
a run crashes rather than finishes.
"""

import asyncio
import json
import pathlib
from typing import Any

import httpx
import pytest

from kira.api.app import create_app
from kira.api.deps import Service
from kira.api.incidents import fingerprint, run_incident_task
from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.middleware.human_approval import HumanApproval
from kira.orchestration.runner import Deployment, resume_after_decision
from kira.settings.authority import AuthoritySettings
from kira.settings.groups import MemorySettings, PipelineSettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

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
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched; endpoints non-empty"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}
PATCH = {"cmd": "kubectl patch svc user -p '{}'"}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "service/user patched"


class ParkingModel:
    """Diagnoses, plans, writes once — which is where a human is asked — then submits."""

    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        wrote = self.callers.count("execution") >= 1
        self.callers.append(caller)
        name, arguments = {
            "rca": ("submit_root_cause", ROOT_CAUSE),
            "planning": ("submit_plan", PLAN),
            "execution": (
                ("submit_execution_result", EXECUTION)
                if wrote
                else ("exec_kubectl_cmd_safely", PATCH)
            ),
        }[caller]
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
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


def deployment(tmp_path: pathlib.Path, *, mode: str = "auto") -> Deployment:
    """Build a deployment over a temporary workspace and a fake cluster."""
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(mode=mode, timeout_s=1800),  # type: ignore[arg-type]
    )
    storage = build_storage(settings)
    return Deployment(
        settings=settings,
        storage=storage,
        completion=ParkingModel(),
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        authority=HumanApproval(
            store=storage.approvals,
            settings=settings.authority,
            arm_timeout=lambda request_id, expires_at: None,
        ),
        sink=ListSink(),
    )


def client(app: Any) -> httpx.AsyncClient:
    """Speak to the application in this event loop, so its tasks are ours to await."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://kira")


async def settle(service: Service) -> None:
    """Wait for every incident this process started."""
    while pending := {task for task in service.tasks if not task.done()}:
        await asyncio.gather(*pending, return_exceptions=True)


async def test_intake_answers_with_the_shape_the_harness_reads(tmp_path: pathlib.Path) -> None:
    """Held contract. The four keys and their values are what an outside driver parses,
    and a run it can poll has to look started from the moment it is accepted."""
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        response = await http.post("/api/v1/incidents", json={"alert": ALERT})
        assert response.status_code == 202
        body = response.json()
        assert set(body) == {"session_id", "status", "target_stage", "started_at"}
        assert body["status"] == "accepted"
        assert body["target_stage"] == "full"
        assert body["started_at"]

        polled = await http.get(f"/api/v1/incidents/{body['session_id']}")
        assert polled.status_code == 200
        assert polled.json()["run_status"] == "running"

        await settle(app.state.service)
    deploy.storage.close()


async def test_the_rca_wire_value_still_means_the_diagnosis_only_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prototype named the stage after the role; the pipeline names it after what it
    delivers. The wire keeps the old word rather than asking the harness to move."""
    stages: list[str] = []

    async def record(alert: dict[str, Any], session_id: str, deploy: Deployment, **kw: Any) -> Any:
        stages.append(kw["target_stage"])

    monkeypatch.setattr("kira.api.incidents.run_incident", record)
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        response = await http.post(
            "/api/v1/incidents", json={"alert": ALERT, "target_stage": "rca"}
        )
        await settle(app.state.service)

    assert response.json()["target_stage"] == "rca"
    assert stages == ["diagnose"]
    deploy.storage.close()


async def test_a_crashed_incident_is_left_terminal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polling adapters key on run_status. A task that has unwound but left the session
    running strands the caller until its own timeout, over a failure already decided."""

    async def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the transport is gone")

    monkeypatch.setattr("kira.api.incidents.run_incident", crash)
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        session_id = (await http.post("/api/v1/incidents", json={"alert": ALERT})).json()[
            "session_id"
        ]
        await settle(app.state.service)
        polled = (await http.get(f"/api/v1/incidents/{session_id}")).json()

    assert polled["run_status"] == "failed"
    assert polled["status"] == "invalidated"
    assert polled["error"] == "RuntimeError: the transport is gone"
    assert app.state.service.runs.get(session_id) is None
    deploy.storage.close()


async def test_a_cancelled_incident_re_raises_and_still_unregisters(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swallowing the cancellation would break the cooperative stop a drain depends on,
    and the registry has to be released either way."""

    async def cancelled(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr("kira.api.incidents.run_incident", cancelled)
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    service: Service = app.state.service
    session = deploy.storage.sessions.create_session(ALERT)
    service.runs.register(session.session_id)

    with pytest.raises(asyncio.CancelledError):
        await run_incident_task(
            session_id=session.session_id, alert=ALERT, target_stage="full", service=service
        )

    assert service.runs.get(session.session_id) is None
    stopped = deploy.storage.sessions.get_session(session.session_id)
    assert stopped is not None and stopped.state == "pending", "a stop is not a crash"
    deploy.storage.close()


async def test_an_unknown_or_already_decided_request_is_a_404(tmp_path: pathlib.Path) -> None:
    """A duplicate decision is not a second decision. The store decides by compare-and-set
    and the route reports the same refusal for both cases the prototype did."""
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        assert (await http.get("/api/v1/hitl/nope")).status_code == 404
        decide = await http.post("/api/v1/hitl/nope/decide", json={"approve": True})
        assert decide.status_code == 404

        session_id = (await http.post("/api/v1/incidents", json={"alert": ALERT})).json()[
            "session_id"
        ]
        await settle(app.state.service)
        request_id = (await http.get(f"/api/v1/incidents/{session_id}")).json()[
            "awaiting_request_id"
        ]

        first = await http.post(f"/api/v1/hitl/{request_id}/decide", json={"approve": True})
        assert first.status_code == 200
        second = await http.post(f"/api/v1/hitl/{request_id}/decide", json={"approve": True})
        assert second.status_code == 404
    deploy.storage.close()


async def test_a_write_pauses_and_a_decision_over_http_continues_it(
    tmp_path: pathlib.Path,
) -> None:
    """The whole surface, once: an incident parks on a person, the request is reviewable
    over HTTP, and the decision continues the run it belongs to rather than a new one."""
    deploy = deployment(tmp_path)
    resumed: list[asyncio.Task[str]] = []
    app = create_app(
        deployment=deploy,
        schedule_resume=lambda request_id: resumed.append(
            asyncio.create_task(resume_after_decision(request_id, deploy))
        ),
    )

    async with client(app) as http:
        session_id = (await http.post("/api/v1/incidents", json={"alert": ALERT})).json()[
            "session_id"
        ]
        await settle(app.state.service)

        listed = (await http.get("/api/v1/hitl")).json()["requests"]
        assert len(listed) == 1
        request = listed[0]
        assert request["stage"] == "execution"
        assert request["payload"]["arguments"] == PATCH, "the whole call, not a preview of it"

        decided = await http.post(
            f"/api/v1/hitl/{request['request_id']}/decide",
            json={"approve": True, "reason": "checked the target", "decided_by": "an operator"},
        )
        assert decided.json() == {
            "decided": True,
            "approved": True,
            "request_id": request["request_id"],
        }

        await asyncio.gather(*resumed)
        polled = (await http.get(f"/api/v1/incidents/{session_id}")).json()

    assert polled["run_status"] == "completed"
    assert polled["execution_result"] is not None
    assert polled["awaiting_request_id"] is None
    deploy.storage.close()


# --- intake bounds ------------------------------------------------------------------


async def test_the_same_alert_twice_is_one_incident(tmp_path: pathlib.Path) -> None:
    """An alerting source retries. Two agents remediating the same fault against each
    other is a worse outcome than a repost being told what is already on it."""
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        first = await http.post("/api/v1/incidents", json={"alert": ALERT})
        second = await http.post("/api/v1/incidents", json={"alert": ALERT})

        assert first.status_code == second.status_code == 202
        assert first.json()["session_id"] == second.json()["session_id"]
        await settle(app.state.service)
    deploy.storage.close()


async def test_a_harness_key_cannot_make_one_alert_look_like_two(
    tmp_path: pathlib.Path,
) -> None:
    """The fingerprint is over what the model can see, so an evaluation harness's own
    bookkeeping key does not defeat deduplication exactly where it reposts."""
    assert fingerprint(ALERT) == fingerprint({**ALERT, "_problem_id": "leak-the-answer"})


async def test_intake_refuses_rather_than_queues_when_it_is_full(
    tmp_path: pathlib.Path,
) -> None:
    """A queue would be invisible to the caller and would slow every run already in
    flight to make room. A private model endpoint serves few concurrent sequences, so
    the bound is on accepting work."""
    deploy = deployment(tmp_path)
    object.__setattr__(deploy.settings.server, "max_concurrent_incidents", 1)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        accepted = await http.post("/api/v1/incidents", json={"alert": ALERT})
        refused = await http.post(
            "/api/v1/incidents", json={"alert": {**ALERT, "alertname": "Another"}}
        )

        assert accepted.status_code == 202
        assert refused.status_code == 429
        assert refused.headers["Retry-After"] == "60"
        await settle(app.state.service)
    deploy.storage.close()
