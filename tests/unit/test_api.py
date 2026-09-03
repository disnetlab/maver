"""The HTTP surface, and the contract an outside harness already depends on.

The intake shape here is held: an evaluation harness outside this repository posts to
it and polls it, and it was written against the prototype. So these tests pin the wire,
not the implementation — the exact keys, the old stage word, and what a poller sees when
a run crashes rather than finishes.
"""

import asyncio
import dataclasses
import json
import pathlib
from typing import Any

import httpx
import pytest
import yaml

from kira.api.app import _drain, create_app
from kira.api.deps import Service
from kira.api.incidents import fingerprint, run_incident_task
from kira.audit.journal import ListSink
from kira.contracts.lifecycle import RunStatus
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.middleware.human_approval import authorities
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


def deployment(tmp_path: pathlib.Path, *, mode: str = "prompt") -> Deployment:
    """Build a deployment over a temporary workspace and a fake cluster.

    Prompting, because every test here is about the pause. The fixture said `auto` and
    parked anyway, because it wired the human authority in regardless of the mode — so
    the mode it declared and the behaviour it produced were unrelated.
    """
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
        authorities=authorities(
            settings.authority,
            storage.approvals,
            lambda request_id, expires_at: None,
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


async def test_the_bound_holds_when_the_requests_arrive_together(
    tmp_path: pathlib.Path,
) -> None:
    """Asking whether there is room and taking the room have to be one step.

    Opening a session and marking it running are two database writes, and every request
    in a burst yields at both — so each passed a capacity check no other had yet
    invalidated. A process configured for two carried seven, which is the arrival
    pattern the bound exists for: an alert storm, or a harness firing several problems
    at once, against an endpoint that wedges rather than queues when oversubscribed.
    """
    deploy = deployment(tmp_path)
    object.__setattr__(deploy.settings.server, "max_concurrent_incidents", 2)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        replies = await asyncio.gather(
            *(
                http.post("/api/v1/incidents", json={"alert": {**ALERT, "service": f"s{i}"}})
                for i in range(6)
            )
        )
        assert sorted(reply.status_code for reply in replies) == [202, 202, 429, 429, 429, 429]
        await settle(app.state.service)
    deploy.storage.close()


async def test_a_draining_process_closes_the_door_before_it_waits(
    tmp_path: pathlib.Path,
) -> None:
    """Draining is two steps in that order, and only the second was here.

    A process that keeps accepting while it drains opens an incident it cannot finish:
    the session says running, the process goes, and nothing resumes it — the outcome the
    drain exists to prevent, arriving through the front door. `/readyz` going 503 is what
    removes the pod from a Service's endpoint list so the requests stop arriving at all,
    while `/healthz` stays 200 so the kubelet does not restart it mid-drain.
    """
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        assert (await http.get("/readyz")).status_code == 200

        app.state.service.runs.begin_draining()

        refused = await http.post("/api/v1/incidents", json={"alert": ALERT})
        assert refused.status_code == 503
        assert refused.headers["Retry-After"] == "30"
        assert (await http.get("/readyz")).status_code == 503
        assert (await http.get("/healthz")).status_code == 200
        await settle(app.state.service)
    deploy.storage.close()


async def test_an_incident_that_outlasts_the_drain_is_recorded_as_over(
    tmp_path: pathlib.Path,
) -> None:
    """The drain waited and then walked away, leaving the session saying `running` on a
    process that no longer exists. Nothing resumes an incident — only an approval — so
    that record never changed again, and `cancelled` was a terminal status both backends
    already recognised and nothing ever wrote.
    """

    class Forever:
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    deploy = dataclasses.replace(deployment(tmp_path, mode="auto"), completion=Forever())
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        created = await http.post("/api/v1/incidents", json={"alert": ALERT})
        session_id = created.json()["session_id"]
        for _ in range(100):  # until the diagnosing role has opened its run row
            if deploy.storage.runs.runs_for(session_id):
                break
            await asyncio.sleep(0.01)
        assert deploy.storage.sessions.get_session(session_id).run_status == "running"

        app.state.service.runs.begin_draining()
        await _drain(app.state.service, 0.0)

    record = deploy.storage.sessions.get_session(session_id)
    assert record.run_status == RunStatus.CANCELLED.value
    assert [run.status for run in deploy.storage.runs.runs_for(session_id)] == ["cancelled"]
    deploy.storage.close()


def test_the_grace_period_outlasts_the_drain_it_has_to_contain() -> None:
    """A drain cut off part-way through is killed while writing the terminal state that
    is the record it exists to produce. The manifest's grace period has to cover the
    drain plus the time to write it."""
    manifest = yaml.safe_load(pathlib.Path("deploy/k8s/base/deployment.yaml").read_text())
    grace = manifest["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
    assert Settings().server.drain_timeout_s + 10 <= grace, (
        "server.drain_timeout_s + 10 must fit inside terminationGracePeriodSeconds"
    )


class _NeedsTheLoop:
    """A session store whose reads need the event loop to run something for them.

    On the loop's own thread the answer can never come, so a route that reads inline
    fails here on this clock instead of hanging the suite.
    """

    STALLED = 2.0

    def __init__(self, sessions: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._sessions = sessions
        self._loop = loop

    async def _anything(self) -> None:
        """Work for the loop to do, whose completion is the proof that it is free."""

    def get_session(self, session_id: str) -> Any:
        pending = asyncio.run_coroutine_threadsafe(self._anything(), self._loop)
        try:
            pending.result(timeout=self.STALLED)
        except TimeoutError:
            raise AssertionError(
                "the poll route read the session store on the event loop, which stalled"
            ) from None
        return self._sessions.get_session(session_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sessions, name)


async def test_polling_an_incident_does_not_stall_the_incident(tmp_path: pathlib.Path) -> None:
    """There is one event loop in this process, not one for the API and one for the runs.

    An incident is a task created inside the intake handler, so it runs on the loop that
    serves requests. A store read made inline by the poll route therefore does not cost
    the API alone: it stops every incident this process is handling, and a harness polls
    one of them every few seconds for hours.
    """
    deploy = deployment(tmp_path)
    deploy = dataclasses.replace(
        deploy,
        storage=dataclasses.replace(
            deploy.storage,
            sessions=_NeedsTheLoop(deploy.storage.sessions, asyncio.get_running_loop()),
        ),
    )
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        accepted = await http.post("/api/v1/incidents", json={"alert": ALERT})
        assert accepted.status_code == 202
        polled = await http.get(f"/api/v1/incidents/{accepted.json()['session_id']}")
        assert polled.status_code == 200
        await settle(app.state.service)
    deploy.storage.close()
