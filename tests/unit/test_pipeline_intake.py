"""Named pipeline intake keeps routing and private context outside the alert."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from maver.api import incidents
from maver.api.app import create_app
from maver.api.deps import Service
from maver.api.incidents import IncidentRequest, fingerprint
from maver.cli import app
from maver.contracts.lifecycle import RunStatus
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.orchestration.runner import Deployment
from maver.pipelines import PipelineCatalog
from maver.settings.groups import MemorySettings, PipelineSettings, ServerSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings


def test_intake_preserves_pipeline_and_private_metadata_separately() -> None:
    alert = {"alertname": "HighErrorRate", "namespace": "shop"}
    request = IncidentRequest.model_validate(
        {
            "alert": alert,
            "pipeline": "database_recovery",
            "metadata": {"tenant": "private-tenant", "routing": {"database": True}},
        }
    )

    assert request.pipeline == "database_recovery"
    assert request.metadata == {"tenant": "private-tenant", "routing": {"database": True}}
    assert request.alert == alert
    assert fingerprint(request.alert) == fingerprint(alert)


def test_cli_submits_pipeline_and_private_metadata_without_merging_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alert = {"alertname": "HighErrorRate", "namespace": "shop"}
    metadata = {"routing": {"region": "private-region"}}
    alert_path = tmp_path / "alert.json"
    metadata_path = tmp_path / "metadata.json"
    alert_path.write_text(json.dumps(alert), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def request(url: str, method: str, path: str, **kwargs: Any) -> dict[str, str]:
        calls.append({"url": url, "method": method, "path": path, **kwargs})
        return {"session_id": "session-1", "status": "accepted", "pipeline": "diagnosis_only"}

    monkeypatch.setattr("maver.cli._api", request)
    result = CliRunner().invoke(
        app,
        [
            "incidents",
            "start",
            str(alert_path),
            "--pipeline",
            "diagnosis_only",
            "--metadata",
            str(metadata_path),
            "--target-stage",
            "diagnose",
            "--url",
            "http://test.invalid:8080",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "url": "http://test.invalid:8080",
            "method": "POST",
            "path": "/incidents",
            "json": {
                "alert": alert,
                "pipeline": "diagnosis_only",
                "metadata": metadata,
                "target_stage": "diagnose",
            },
        }
    ]
    assert json.loads(result.output)["session_id"] == "session-1"
    assert "private-region" not in result.output


ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}
PRIVATE = "private-context-must-not-reach-model"


class DiagnosisModel:
    def __init__(self, *, blocked: bool = False) -> None:
        self.requests: list[ModelRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        assert caller == "rca"
        args = {
            "fault_type": "misconfiguration",
            "affected_resource": "shop/Service/user",
            "confidence": 0.9,
            "evidence": ["service has no endpoints"],
            "summary": "incorrect service target port",
        }
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id="diagnosed",
                    name="submit_root_cause",
                    arguments=args,
                    raw_arguments=json.dumps(args),
                ),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


class Transport:
    name = "fake"
    capabilities = frozenset({"kubernetes"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "service/user has no endpoints"


def _deployment(tmp_path: Path, model: DiagnosisModel) -> Deployment:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(auto_snapshot=False),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
    )
    return Deployment(
        settings=settings,
        storage=build_storage(settings),
        completion=model,
        profile=load_profile("qwen3.6-35b-a3b", Path("configs/models")),
        transport_for=lambda session_id: Transport(),
    )


async def _settle(service: Service) -> None:
    while tasks := {task for task in service.runs.tasks() if not task.done()}:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_private_metadata_is_persisted_but_absent_from_model_and_api(
    tmp_path: Path,
) -> None:
    model = DiagnosisModel()
    deploy = _deployment(tmp_path, model)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://maver"
        ) as client:
            response = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                    "metadata": {"tenant": PRIVATE},
                },
            )
            assert response.status_code == 202, response.text
            session_id = response.json()["session_id"]
            assert response.json()["pipeline"] == "diagnose"
            await _settle(app.state.service)
            polled = await client.get(f"/api/v1/incidents/{session_id}")
            assert polled.json()["run_status"] == "completed"
            assert polled.json()["pipeline"] == "diagnose"
            assert PRIVATE not in response.text + polled.text
        stored = deploy.storage.sessions.get_session(session_id)
        assert stored is not None
        assert stored.alert == ALERT
        assert stored.runtime["metadata"] == {"tenant": PRIVATE}
        assert stored.runtime["selection"]["name"] == "diagnose"
        assert model.requests
        assert PRIVATE not in repr(model.requests)
    finally:
        model.release.set()
        await _settle(app.state.service)
        deploy.storage.close()


async def test_concurrent_duplicates_select_once_and_keep_first_private_metadata(
    tmp_path: Path,
) -> None:
    model = DiagnosisModel(blocked=True)
    calls: list[Mapping[str, Any]] = []
    selector_entered = asyncio.Event()
    selector_release = asyncio.Event()

    class Selector:
        async def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
            calls.append(dict(metadata))
            selector_entered.set()
            await selector_release.wait()
            return "diagnose"

    deploy = replace(_deployment(tmp_path, model), pipeline_selectors=(Selector(),))
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://maver"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/api/v1/incidents",
                    json={
                        "alert": ALERT,
                        "metadata": {"tenant": PRIVATE},
                    },
                )
            )
            await asyncio.wait_for(selector_entered.wait(), timeout=5)
            duplicates = [
                asyncio.create_task(
                    client.post(
                        "/api/v1/incidents",
                        json={
                            "alert": ALERT,
                            "metadata": {"retry": number},
                        },
                    )
                )
                for number in range(3)
            ]
            selector_release.set()
            replies = await asyncio.gather(first, *duplicates)
            assert [reply.status_code for reply in replies] == [202] * 4
            assert len({reply.json()["session_id"] for reply in replies}) == 1
            assert calls == [{"tenant": PRIVATE}]
            session_id = replies[0].json()["session_id"]
            stored = deploy.storage.sessions.get_session(session_id)
            assert stored is not None and stored.runtime["metadata"] == {"tenant": PRIVATE}
            model.release.set()
            await _settle(app.state.service)
            assert calls == [{"tenant": PRIVATE}], "runtime must not invoke selection again"
    finally:
        selector_release.set()
        model.release.set()
        await _settle(app.state.service)
        deploy.storage.close()


async def test_explicit_selection_bypasses_selector_and_conflicting_repost_is_rejected(
    tmp_path: Path,
) -> None:
    model = DiagnosisModel(blocked=True)

    class Selector:
        def select(self, alert: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
            raise AssertionError("explicit selection must bypass the selector")

    deploy = replace(_deployment(tmp_path, model), pipeline_selectors=(Selector(),))
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://maver"
        ) as client:
            accepted = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                },
            )
            assert accepted.status_code == 202
            await asyncio.wait_for(model.started.wait(), timeout=5)
            conflicting = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "multi",
                    "metadata": {"retry": True},
                },
            )
            assert conflicting.status_code == 409
            assert "diagnose" in conflicting.json()["detail"]
            same = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                    "metadata": {"retry": 2},
                },
            )
            assert same.status_code == 202
            assert same.json()["session_id"] == accepted.json()["session_id"]
            assert len(app.state.service.runs.live()) == len(model.requests) == 1
    finally:
        model.release.set()
        await _settle(app.state.service)
        deploy.storage.close()


async def test_invalid_selection_does_not_reserve_capacity_or_start_an_incident(
    tmp_path: Path,
) -> None:
    model = DiagnosisModel()
    deploy = _deployment(tmp_path, model)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://maver"
        ) as client:
            unknown = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "unknown",
                },
            )
            mismatch = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "multi",
                    "target_stage": "diagnose",
                },
            )
            assert unknown.status_code == mismatch.status_code == 422
            assert not app.state.service.runs.live() and not model.requests
            valid = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                },
            )
            assert valid.status_code == 202
            await _settle(app.state.service)
    finally:
        deploy.storage.close()


async def test_deployment_default_runs_without_explicit_selection(tmp_path: Path) -> None:
    model = DiagnosisModel()
    base = _deployment(tmp_path, model)
    assert base.pipelines is not None
    deploy = replace(base, pipelines=PipelineCatalog(base.pipelines.pipelines, "diagnose"))
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://maver"
        ) as client:
            response = await client.post("/api/v1/incidents", json={"alert": ALERT})
            assert response.status_code == 202
            assert set(response.json()) == {"session_id", "status", "target_stage", "started_at"}
            await _settle(app.state.service)
            session = deploy.storage.sessions.get_session(response.json()["session_id"])
            assert session is not None
            assert session.runtime["selection"]["name"] == "diagnose"
            assert session.runtime["selection"]["source"] == "default"
    finally:
        deploy.storage.close()


@pytest.mark.parametrize("content", ["[]", "invalid JSON"])
def test_cli_rejects_invalid_metadata_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    alert = tmp_path / "alert.json"
    metadata = tmp_path / "metadata.json"
    alert.write_text(json.dumps(ALERT), encoding="utf-8")
    metadata.write_text(content, encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid metadata must not reach the server")

    monkeypatch.setattr("maver.cli._api", forbidden)
    result = CliRunner().invoke(
        app, ["incidents", "start", str(alert), "--metadata", str(metadata)]
    )
    assert result.exit_code == 2
    assert "JSON object" in result.output


def test_cli_lists_resolved_pipeline_names() -> None:
    result = CliRunner().invoke(app, ["config", "pipelines"])
    assert result.exit_code == 0, result.output
    assert "multi (default): multi" in result.output
    assert "single: single" in result.output
    assert "diagnose: diagnose" in result.output


@pytest.mark.parametrize("failure_at", ["runtime", "status"])
async def test_failed_initial_persistence_closes_session_and_releases_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
) -> None:
    model = DiagnosisModel()
    deploy = _deployment(tmp_path, model)
    app = create_app(deployment=deploy, schedule_resume=lambda request_id: None)
    sessions = deploy.storage.sessions
    created: list[str] = []
    original_create = sessions.create_session

    def create(alert: dict[str, Any]) -> Any:
        record = original_create(alert)
        created.append(record.session_id)
        return record

    monkeypatch.setattr(sessions, "create_session", create)
    original_write = incidents.save_runtime if failure_at == "runtime" else sessions.set_run_status
    failed = False

    def write(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError(f"database rejected row with {PRIVATE}")
        return original_write(*args, **kwargs)

    if failure_at == "runtime":
        monkeypatch.setattr("maver.api.incidents.save_runtime", write)
    else:
        monkeypatch.setattr(sessions, "set_run_status", write)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://maver",
        ) as client:
            rejected = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                    "metadata": {"tenant": PRIVATE},
                },
            )
            assert rejected.status_code == 503
            assert rejected.headers["Retry-After"] == "30"
            assert len(created) == 1
            record = sessions.get_session(created[0])
            assert record is not None and record.run_status == RunStatus.FAILED
            assert not model.requests and not app.state.service.runs.live()
            polled = await client.get(f"/api/v1/incidents/{created[0]}")
            assert PRIVATE not in rejected.text + polled.text
            retried = await client.post(
                "/api/v1/incidents",
                json={
                    "alert": ALERT,
                    "pipeline": "diagnose",
                },
            )
            assert retried.status_code == 202
            assert retried.json()["session_id"] != created[0]
            await _settle(app.state.service)
    finally:
        await _settle(app.state.service)
        deploy.storage.close()
