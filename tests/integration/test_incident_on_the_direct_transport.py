"""One whole incident through the served application, on the transport it ships with.

Every other test of the pipeline substitutes the transport. That leaves the composition
an adopter actually runs — settings resolved from the environment, storage opened,
`direct_transport` resolving a binary on PATH, toolsets built against what it says it
serves, the write gate, the roles, the graph, outcome persistence, the HTTP route —
assembled nowhere, and the defects this rebuild kept finding lived precisely in that
gap: two halves each correct against a double that supplies what production does not.

Only the model is faked, because it is the one component whose output cannot be asserted
on. `kubectl` is a script on PATH, so the transport, its capability resolution, its
timeout arithmetic and the tool's own argv normalisation all run for real.
"""

import asyncio
import json
import os
import pathlib
import stat
from dataclasses import replace
from typing import Any

import httpx
import pytest

from kira.api.app import build_deployment, create_app
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.orchestration import jobs
from kira.settings.root import load_settings, reset_settings

# Deliberately unmarked. The `integration` marker means a live cluster, database or
# model endpoint, and this needs none: the binary it calls is a shell script it writes
# itself. A composition nothing exercises by default is a composition nobody exercises,
# which is how every defect this file was written for survived two green suites.

ALERT = {"alertname": "HighErrorRate", "namespace": "shop", "service": "user"}

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
VERIFY: dict[str, Any] = {
    "recovered": True,
    "verdict": "recovered",
    "checks": ["endpoints present for svc/user"],
    "entry_point_status": "frontend returned 200 on /",
    "summary": "the symptom is gone",
    "confidence": 0.8,
}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}

FAKE_KUBECTL = """#!/bin/sh
# Answers every read the same way, and records the fact that it ran. What matters here
# is that a real binary was found, invoked with the argv the tool built, its stdout came
# back through the transport — and, for the approval drill, that it ran exactly as often
# as a person allowed.
echo "$@" >> "$TEST_KUBECTL_LOG"
echo "NAME   TYPE        CLUSTER-IP   PORT(S)"
echo "user   ClusterIP   10.0.0.1     80/TCP"
"""


class ScriptedModel:
    """Calls a real cluster tool once per role, then submits that role's artifact."""

    def __init__(self) -> None:
        self.callers: list[str] = []
        self.free_text: list[str] = []
        self.tool_results: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        first = self.callers.count(caller) == 1
        for message in request.messages:
            if message.get("role") == "tool":
                self.tool_results.append(str(message.get("content") or ""))

        submissions = {
            "rca": ("submit_root_cause", ROOT_CAUSE),
            "planning": ("submit_plan", PLAN),
            "execution": ("submit_execution_result", EXECUTION),
            "verify": ("submit_verify_result", VERIFY),
        }
        if caller not in submissions:
            # The gate, the nominator and the selector answer in free text rather than
            # through a submission tool, and they are counted like every other call.
            self.free_text.append(caller)
            return ModelResponse(
                text='{"verdict": "sound", "reason": "the cited evidence supports it"}',
                tool_calls=(),
                finish_reason="stop",
                reasoning=None,
                usage=Usage(prompt_tokens=5, completion_tokens=3),
                duration_ms=1,
                attempts=1,
            )
        if first and caller in {"rca", "execution"}:
            name: str = "exec_kubectl_cmd_safely"
            # A read while diagnosing and a write while remediating, which is what makes
            # the approval gate fire in exactly one place.
            arguments: dict[str, Any] = {
                "cmd": "kubectl get svc user -n shop"
                if caller == "rca"
                else 'kubectl patch svc user -n shop -p \'{"spec":{"ports":[{"targetPort":8080}]}}\''
            }
        else:
            name, arguments = submissions[caller]
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


@pytest.fixture
def deployment(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the deployment the way the process does, with a kubectl it can find."""
    binary = tmp_path / "bin" / "kubectl"
    binary.parent.mkdir(parents=True)
    binary.write_text(FAKE_KUBECTL, encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{binary.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TEST_KUBECTL_LOG", str(tmp_path / "kubectl.log"))

    monkeypatch.setenv("KIRA_SERVER__WORKDIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("KIRA_MEMORY__ENABLED", "false")
    monkeypatch.setenv("KIRA_MEMORY__LEARNING_ENABLED", "false")
    monkeypatch.setenv("KIRA_VERIFY__ENABLED", "false")
    monkeypatch.setenv("KIRA_AUTHORITY__MODE", "auto")
    monkeypatch.setenv("KIRA_PIPELINE__AUTO_SNAPSHOT", "false")
    reset_settings()
    return replace(build_deployment(load_settings()), completion=ScriptedModel())


async def test_an_incident_runs_end_to_end_on_the_shipped_transport(deployment: Any) -> None:
    """The check the prototype could not perform at all: no benchmark, no adapter."""
    app = create_app(deployment=deployment, schedule_resume=lambda request_id: None)
    transport = deployment.transport_for("probe")
    assert "kubernetes" in transport.capabilities, "a kubectl on PATH is the whole premise"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://kira"
    ) as http:
        created = await http.post("/api/v1/incidents", json={"alert": ALERT})
        assert created.status_code == 202, created.text
        session_id = created.json()["session_id"]

        while pending := {t for t in app.state.service.runs.tasks() if not t.done()}:
            await asyncio.gather(*pending, return_exceptions=True)

        incident = await http.get(f"/api/v1/incidents/{session_id}")
        assert incident.status_code == 200, incident.text
        body = incident.json()

    assert body["root_cause"]["affected_resource"] == "shop/Service/user"
    assert body["execution_result"]["succeeded"] is True
    # The tool result the model saw came from the script on PATH, through the transport.
    assert any("ClusterIP" in result for result in deployment.completion.tool_results)

    runs = deployment.storage.runs.runs_for(session_id)
    assert [run.agent_name for run in runs] == ["rca", "planning", "execution"]
    assert all(run.status == "succeeded" for run in runs), [r.status for r in runs]

    # What a run offered the model is reconstructible from its own row, which is the
    # point of recording it: with only kubectl on PATH the transport serves no telemetry
    # and this configuration has no experience store, so these were never offered and the
    # row says which and why rather than leaving it to be inferred from the environment.
    # ssh_exec is not among them: it is offered on an empty node allowlist, as every
    # measured campaign offered it, and refuses each call itself.
    withheld = {
        name: reason
        for run in runs
        for name, reason in (run.metadata.get("tools_withheld") or {}).items()
    }
    assert withheld == {
        "get_metrics": "transport lacks metrics",
        "get_alerts": "transport lacks metrics",
        "get_logs": "transport lacks logs",
        "get_labels": "transport lacks logs",
        "get_label_values": "transport lacks logs",
        "get_services": "transport lacks traces",
        "get_operations": "transport lacks traces",
        "get_traces": "transport lacks traces",
        "get_dependency_graph": "transport lacks traces",
        "memory_search": "no experience store in this configuration",
        "open_memory": "no experience store in this configuration",
        "session_trace_summary": "no experience store in this configuration",
    }, withheld

    # Every model call and every cluster write reached the ordered log, through the
    # sink the composition root chose rather than the discarding default.
    events = deployment.storage.runs.events_for(runs[0].agent_run_id)
    assert events, "the audit sink wrote nothing for the diagnosing role"


@pytest.fixture
def every_mechanism(deployment: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The same composition with the store, the second pass and verification on."""
    monkeypatch.setenv("KIRA_MEMORY__ENABLED", "true")
    monkeypatch.setenv("KIRA_MEMORY__SECOND_PASS__ENABLED", "true")
    monkeypatch.setenv("KIRA_MEMORY__SECOND_PASS__GATE", "quality")
    monkeypatch.setenv("KIRA_MEMORY__SECOND_PASS__DELIVERY", "context")
    monkeypatch.setenv("KIRA_VERIFY__ENABLED", "true")
    reset_settings()
    return replace(build_deployment(load_settings()), completion=ScriptedModel())


async def test_every_mechanism_composes_on_the_shipped_transport(every_mechanism: Any) -> None:
    """The configuration with the most moving parts, assembled for real.

    Two diagnoses under a gate, a blinded selection between them, a plan, an apply and an
    independent re-check — each reading a cluster through the transport this ships with.
    """
    app = create_app(deployment=every_mechanism, schedule_resume=lambda request_id: None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://kira"
    ) as http:
        created = await http.post("/api/v1/incidents", json={"alert": ALERT})
        session_id = created.json()["session_id"]
        while pending := {t for t in app.state.service.runs.tasks() if not t.done()}:
            await asyncio.gather(*pending, return_exceptions=True)
        body = (await http.get(f"/api/v1/incidents/{session_id}")).json()

    assert body["verify_result"]["verdict"] == "recovered"
    assert every_mechanism.completion.free_text == ["secondpass.gate", "secondpass.selector"]

    runs = every_mechanism.storage.runs.runs_for(session_id)
    assert [run.agent_name for run in runs] == ["rca", "rca", "planning", "execution", "verify"]
    # Both diagnoses are on the record and tellable apart. They were two rows saying only
    # `rca`, which made the pass's own dose unreadable from the run it produced.
    assert [run.metadata.get("run_label") for run in runs[:2]] == ["rca.first", "rca.second"]


@pytest.fixture
def prompting(deployment: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The same composition, holding every cluster write for a person."""
    monkeypatch.setenv("KIRA_AUTHORITY__MODE", "prompt")
    reset_settings()
    built = replace(build_deployment(load_settings()), completion=ScriptedModel())
    jobs.bind_deployment(built)
    try:
        yield built
    finally:
        jobs.unbind_deployment()


def _kubectl_calls(tmp_path: pathlib.Path) -> list[str]:
    log = tmp_path / "kubectl.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@pytest.mark.parametrize("approve", [True, False])
async def test_a_write_waits_for_a_person_and_then_does_what_they_said(
    prompting: Any, tmp_path: pathlib.Path, approve: bool
) -> None:
    """The drill the published system never ran: approval was configured and never
    invoked, so the mechanism requirement R2 rests on was effectively untested.

    Composed, on the shipped transport, with a kubectl that records every invocation:
    the incident parks with the cluster untouched, a person is shown the whole command
    rather than a preview of it, and the decision is what decides whether it runs.
    """
    resumed: list[str] = []
    app = create_app(deployment=prompting, schedule_resume=resumed.append)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://kira"
    ) as http:
        created = await http.post("/api/v1/incidents", json={"alert": ALERT})
        session_id = created.json()["session_id"]
        while pending := {t for t in app.state.service.runs.tasks() if not t.done()}:
            await asyncio.gather(*pending, return_exceptions=True)

        parked = (await http.get(f"/api/v1/incidents/{session_id}")).json()
        request_id = parked["awaiting_request_id"]
        assert request_id, parked
        before = _kubectl_calls(tmp_path)

        shown = (await http.get(f"/api/v1/hitl/{request_id}")).json()
        assert shown["payload"]["arguments"]["cmd"].startswith("kubectl ")

        decided = await http.post(
            f"/api/v1/hitl/{request_id}/decide",
            json={"approve": approve, "decided_by": "an operator"},
        )
        assert decided.status_code == 200, decided.text
        assert resumed == [request_id]

        assert await jobs.resume_approval(request_id) == "success"
        while pending := {t for t in app.state.service.runs.tasks() if not t.done()}:
            await asyncio.gather(*pending, return_exceptions=True)

    after = _kubectl_calls(tmp_path)
    # The write is the only call either role makes that mutates anything, and whether it
    # ran is exactly the operator's answer. An approved one runs once, not twice: the
    # resume replays a decided request rather than issuing a second.
    assert len(after) == len(before) + (1 if approve else 0), (before, after)
