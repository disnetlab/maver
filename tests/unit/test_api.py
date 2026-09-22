"""The HTTP surface, and the contract an outside harness already depends on.

The intake shape here is held: an evaluation harness outside this repository posts to
it and polls it, and it was written against the prototype. So these tests pin the wire,
not the implementation — the exact keys, the old stage word, and what a poller sees when
a run crashes rather than finishes.
"""

import asyncio
import dataclasses
import importlib
import json
import pathlib
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import yaml

from maver.api.app import _drain, create_app, validate_deployment
from maver.api.deps import Service
from maver.api.incidents import fingerprint, run_incident_task
from maver.api.recover import recover_active
from maver.audit.journal import ListSink
from maver.contracts.lifecycle import RunStatus
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.middleware.human_approval import authorities
from maver.orchestration import scheduler
from maver.orchestration.jobs import resume_approval
from maver.orchestration.runner import Deployment, resume_after_decision, run_incident
from maver.roles import registry as role_registry
from maver.roles.registry import registered_roles
from maver.settings.authority import AuthoritySettings
from maver.settings.groups import MemorySettings, PipelineSettings, ServerSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools import allowlists
from maver.tools import policy as tool_policy
from maver.tools import registry as tool_registry
from maver.tools.registry import registered

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
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
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
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://maver")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "queue_stop"])
async def test_lifespan_releases_resources_even_when_initialization_or_cleanup_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module = importlib.import_module("maver.api.app")
    dep = deployment(tmp_path)
    closed = Mock(wraps=dep.storage.close)
    dep = dataclasses.replace(dep, storage=dataclasses.replace(dep.storage, close=closed))
    monkeypatch.setattr(module, "build_deployment", lambda settings: dep)
    monkeypatch.setattr(module, "load_settings", lambda: dep.settings)
    stopped, unbound = Mock(), Mock()
    monkeypatch.setattr(module, "shutdown_scheduler", stopped)
    monkeypatch.setattr(module, "unbind_deployment", unbound)
    monkeypatch.setattr(module, "bind_deployment", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "schedule_approval_sweep", lambda *args: None)

    def fail(*args):
        raise RuntimeError("injected failure")

    if failure == "startup":
        monkeypatch.setattr(module, "configure_tracing", fail)
    else:
        object.__setattr__(dep.settings.intake.kafka, "enabled", True)
        monkeypatch.setattr(module, "configure_tracing", lambda *args: None)

        class BrokenQueue:
            def __init__(self, service):
                pass

            async def start(self):
                pass

            async def stop(self):
                fail()

        monkeypatch.setattr(module, "QueueConsumer", BrokenQueue)
    app = module.create_app()
    with pytest.raises(RuntimeError, match="injected failure"):
        async with app.router.lifespan_context(app):
            assert failure == "queue_stop"
    assert app.state.service.runs.draining
    stopped.assert_called_once()
    unbound.assert_called_once()
    closed.assert_called_once()


async def settle(service: Service) -> None:
    """Wait for every incident this process started."""
    while pending := {task for task in service.runs.tasks() if not task.done()}:
        await asyncio.gather(*pending, return_exceptions=True)


async def test_opt_in_restart_resumes_an_interrupted_diagnosis(tmp_path: pathlib.Path) -> None:
    """A stored pre-role snapshot resumes under one transcript and the original session."""
    deploy = deployment(tmp_path, mode="auto")
    settings = deploy.settings.model_copy(
        update={
            "server": deploy.settings.server.model_copy(
                update={"resume_interrupted_incidents": True}
            )
        }
    )

    class CrashOnce:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = ParkingModel()

        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated process loss during diagnosis")
            return await self.delegate.complete(request, caller=caller)

    model = CrashOnce()
    deploy = dataclasses.replace(deploy, settings=settings, completion=model)
    session = deploy.storage.sessions.create_session(ALERT)
    deploy.storage.sessions.set_run_status(session.session_id, RunStatus.RUNNING)
    with pytest.raises(RuntimeError, match="simulated process loss"):
        await run_incident(ALERT, session.session_id, deploy, target_stage="diagnose")
    saved = deploy.storage.sessions.get_session(session.session_id)
    assert saved is not None and saved.checkpoint is not None
    assert saved.checkpoint["stage"] == "rca"

    service = Service(deployment=deploy, schedule_resume=lambda request_id: None)
    await recover_active(service)
    await settle(service)
    finished = deploy.storage.sessions.get_session(session.session_id)
    assert finished is not None and finished.run_status == RunStatus.COMPLETED
    assert model.calls == 2


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

    monkeypatch.setattr("maver.api.incidents.run_incident", record)
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

    monkeypatch.setattr("maver.api.incidents.run_incident", crash)
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
    assert app.state.service.runs.live() == []
    deploy.storage.close()


async def test_a_cancelled_incident_re_raises_and_still_unregisters(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swallowing the cancellation would break the cooperative stop a drain depends on,
    and the registry has to be released either way."""

    async def cancelled(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr("maver.api.incidents.run_incident", cancelled)
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    service: Service = app.state.service
    session = deploy.storage.sessions.create_session(ALERT)
    service.runs.register(session.session_id)

    with pytest.raises(asyncio.CancelledError):
        await run_incident_task(
            session_id=session.session_id, alert=ALERT, target_stage="full", service=service
        )

    assert service.runs.live() == []
    stopped = deploy.storage.sessions.get_session(session.session_id)
    assert stopped is not None and stopped.state == "pending", "a stop is not a crash"
    deploy.storage.close()


async def test_an_operator_can_stop_a_running_incident(tmp_path: pathlib.Path) -> None:
    """The runbook's answer to a run that is spiralling, and it reached nothing.

    Deleting the pod is the alternative, and it skips every ``finally`` the run is
    holding: the terminal record, and an environment adapter's release of the
    application being graded.
    """

    class Forever:
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    deploy = dataclasses.replace(deployment(tmp_path, mode="auto"), completion=Forever())
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        session_id = (await http.post("/api/v1/incidents", json={"alert": ALERT})).json()[
            "session_id"
        ]
        for _ in range(100):  # until the diagnosing role has opened its run row
            if deploy.storage.runs.runs_for(session_id):
                break
            await asyncio.sleep(0.01)

        stopped = await http.post(f"/api/v1/incidents/{session_id}/cancel")
        assert stopped.status_code == 202
        assert stopped.json() == {"session_id": session_id, "status": "cancelling"}
        await settle(app.state.service)

        assert (await http.post("/api/v1/incidents/nope/cancel")).status_code == 404
        again = await http.post(f"/api/v1/incidents/{session_id}/cancel")
        assert again.status_code == 409, "a run that has already stopped is not running"

    assert deploy.storage.sessions.get_session(session_id).run_status == RunStatus.CANCELLED.value
    assert app.state.service.runs.live() == []
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


async def test_the_deduplication_holds_when_the_reposts_arrive_together(
    tmp_path: pathlib.Path,
) -> None:
    """Asking whether an alert is already being handled and recording that it now is
    have to be one step, exactly as asking for room and taking it do.

    Between the two there were two database writes, and every request in a burst yields
    at both — so each post of one alert passed a check no other had yet answered, and
    four agents remediated the same fault against each other, on a cluster each of them
    was also mutating.
    """
    deploy = deployment(tmp_path)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)

    async with client(app) as http:
        replies = await asyncio.gather(
            *(http.post("/api/v1/incidents", json={"alert": ALERT}) for _ in range(4))
        )
        assert [reply.status_code for reply in replies] == [202, 202, 202, 202]
        assert len({reply.json()["session_id"] for reply in replies}) == 1
        assert len(app.state.service.runs.live()) == 1
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


async def test_an_approval_that_restarts_an_incident_is_one_the_drain_can_see(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The continuation ran as a scheduler job the process was not tracking.

    So the drain saw an empty live set and returned at once, the scheduler's shutdown
    cancelled the coroutine, and the single incident a human had just approved a cluster
    write for — the one whose write has already been replayed — was left at `resuming`
    with nothing to finish it and nothing reporting it as over.
    """

    class StallsOnceResumed(ParkingModel):
        """Parks on the write, then never answers again once it has been replayed."""

        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            if caller == "execution" and self.callers.count("execution") >= 1:
                await asyncio.sleep(3600)
            return await super().complete(request, caller=caller)

    # The scheduler is a process-wide singleton, and an earlier test leaves one bound to
    # an event loop that has since closed. This lifespan starts and stops its own.
    monkeypatch.setattr(scheduler, "_SCHEDULER", None)
    deploy = dataclasses.replace(deployment(tmp_path), completion=StallsOnceResumed())
    object.__setattr__(deploy.settings.server, "drain_timeout_s", 0.0)
    resumed: list[asyncio.Task[str]] = []
    app = create_app(
        deployment=deploy,
        schedule_resume=lambda request_id: resumed.append(
            asyncio.create_task(resume_approval(request_id))
        ),
    )

    # The real lifespan, because what tells the continuation where this process tracks
    # its runs is the binding the lifespan does.
    async with app.router.lifespan_context(app), client(app) as http:
        session_id = (await http.post("/api/v1/incidents", json={"alert": ALERT})).json()[
            "session_id"
        ]
        await settle(app.state.service)
        request_id = (await http.get(f"/api/v1/incidents/{session_id}")).json()[
            "awaiting_request_id"
        ]
        await http.post(f"/api/v1/hitl/{request_id}/decide", json={"approve": True})
        for _ in range(200):  # until the continuation is under way
            await asyncio.sleep(0.01)
            if app.state.service.runs.live():
                break
        assert app.state.service.runs.live() == [session_id], "the drain waits on this"

    await asyncio.gather(*resumed, return_exceptions=True)
    record = deploy.storage.sessions.get_session(session_id)
    assert record.run_status == RunStatus.CANCELLED.value, "a continuation is still over"
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


# --- the extension seams pyproject advertises ---------------------------------------

_PLUGIN_ROLE = """
from maver.contracts.artifacts import RootCause
from maver.roles.base import RoleSpec
from maver.roles.registry import role
from maver.tools.allowlists import ROLE_TOOLSETS


async def _run(ctx):
    raise NotImplementedError


ROLE_TOOLSETS["demo_plugin_role"] = ("get_metrics", "submit_root_cause")
role(
    RoleSpec(
        name="demo_plugin_role",
        submit_tool="submit_root_cause",
        submit_model=RootCause,
        state_slot="root_cause",
        prompt="rca",
        run=_run,
    )
)
"""

_PLUGIN_TOOL = """
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.policy import read_only
from maver.tools.registry import tool


@tool(
    name="third_party_probe",
    safety="read_only",
    parameters={"type": "object", "properties": {}},
    descriptions={"third_party_probe": {"default": "A probe shipped out of tree."}},
)
async def third_party_probe(ctx):
    return "ok"


read_only("third_party_probe")
ROLE_TOOLSETS["rca"] = (*ROLE_TOOLSETS["rca"][:-1], "third_party_probe", ROLE_TOOLSETS["rca"][-1])
"""


@pytest.fixture
def plugin_registries() -> Any:
    """Put the process-wide registries back, so a plugin cannot leak into another test."""
    live_registries = (
        role_registry._ROLES,
        tool_registry._REGISTRY,
        allowlists.ROLE_TOOLSETS,
        tool_policy._POLICIES,
    )
    saved = [dict(live) for live in live_registries]
    yield
    for live, original in zip(live_registries, saved, strict=True):
        live.clear()
        live.update(original)


def _install_plugin(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, module: str, source: str, group: str
) -> None:
    """Put a real installed distribution on the path, as an adopter's package would be."""
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{module}.py").write_text(source, encoding="utf-8")
    dist = site / "maver_demo-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: maver-demo\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text(f"[{group}]\ndemo = {module}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()


@pytest.mark.parametrize(
    ("group", "module", "source", "registered_as"),
    [
        ("maver.roles", "maver_demo_role", _PLUGIN_ROLE, "demo_plugin_role"),
        ("maver.tools", "maver_demo_tool", _PLUGIN_TOOL, "third_party_probe"),
    ],
    ids=["roles", "tools"],
)
def test_a_plugin_distribution_is_loaded_at_the_composition_root(
    group: str,
    module: str,
    source: str,
    registered_as: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_registries: None,
) -> None:
    """pyproject tells an adopter these groups need no fork, and nothing imported them.

    The distribution was on the path and invisible: the role was never registered, so
    naming it in an arm file failed as an unknown role, with nothing to connect that to
    the entry point the adopter had declared.
    """
    _install_plugin(tmp_path, monkeypatch, module, source, group)
    monkeypatch.setenv("MAVER_SERVER__WORKDIR", str(tmp_path / "workspace"))

    validate_deployment(Settings())

    assert registered_as in (registered_roles() if group == "maver.roles" else registered())
