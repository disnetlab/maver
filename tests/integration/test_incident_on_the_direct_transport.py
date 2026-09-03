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
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}

FAKE_KUBECTL = """#!/bin/sh
# Answers every read the same way. What matters here is that a real binary was found,
# invoked with the argv the tool built, and its stdout came back through the transport.
echo "NAME   TYPE        CLUSTER-IP   PORT(S)"
echo "user   ClusterIP   10.0.0.1     80/TCP"
"""


class ScriptedModel:
    """Calls a real cluster tool once per role, then submits that role's artifact."""

    def __init__(self) -> None:
        self.callers: list[str] = []
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
        }
        if first and caller in {"rca", "execution"}:
            name: str = "exec_kubectl_cmd_safely"
            arguments: dict[str, Any] = {"cmd": "kubectl get svc user -n shop"}
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

        while pending := {t for t in app.state.service.tasks if not t.done()}:
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
    # and this deployment has no node, so four tools were never offered and the row says
    # which and why rather than leaving it to be inferred from the environment.
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
        "ssh_exec": "no host-recovery node is configured",
        "memory_search": "no experience store in this configuration",
        "open_memory": "no experience store in this configuration",
        "session_trace_summary": "no experience store in this configuration",
    }, withheld

    # Every model call and every cluster write reached the ordered log, through the
    # sink the composition root chose rather than the discarding default.
    events = deployment.storage.runs.events_for(runs[0].agent_run_id)
    assert events, "the audit sink wrote nothing for the diagnosing role"
