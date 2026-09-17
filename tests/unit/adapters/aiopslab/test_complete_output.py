"""Large cluster reads stay parseable until the model-facing delivery boundary."""

import json
import shlex
from pathlib import Path

import httpx
import pytest
import respx

from maver.adapters.aiopslab.transport import AiopsLabTransport
from maver.audit.journal import Journal
from maver.tools.context import ToolContext
from maver.tools.errors import OUTPUT_LIMIT
from maver.tools.kubernetes import exec_kubectl_cmd_safely, read_kubectl
from maver.tools.snapshot import namespace_snapshot
from maver.tools.traces import get_traces
from maver.tools.workspace import SessionWorkspace

ACTION = "http://127.0.0.1:18099/v1/action"
NAMESPACE = "test-hotel-reservation"
COMMAND = f"kubectl get pods -n {NAMESPACE} -o json"
# The useful item comes after both old transport and normal tool display limits.
PODS = {
    "apiVersion": "v1",
    "kind": "PodList",
    "metadata": {"annotations": {"fixture": "x" * (OUTPUT_LIMIT + 256)}},
    "items": [
        {
            "metadata": {"name": "frontend-tail", "labels": {"app": "frontend"}},
            "status": {"phase": "Pending"},
        }
    ],
}
PAYLOAD = json.dumps(PODS)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    bound = AiopsLabTransport("http://127.0.0.1:18099", "fixture-token", session_id="s1")
    bound.namespace = NAMESPACE
    bound.stage = "rca"
    return ToolContext(
        transport=bound,
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
    )


async def test_large_inventory_remains_parseable_by_snapshot(
    ctx: ToolContext, caplog: pytest.LogCaptureFixture
) -> None:
    """HTTP 200 previously hid 10,013-character invalid JSON and eight lost reads."""

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["namespace"] == NAMESPACE
        assert body["stage"] == "rca"
        assert request.headers["Authorization"] == "Bearer fixture-token"
        argv = shlex.split(body["arguments"]["cmd"])
        payload = PAYLOAD if argv[2] == "pods" else json.dumps({"items": []})
        return httpx.Response(200, json={"ok": True, "result": payload})

    with respx.mock as router:
        router.post(ACTION).mock(side_effect=answer)
        snapshot = await namespace_snapshot(ctx, NAMESPACE)
        assert len(router.calls) == 8, "Valid reads must not be retried as unusable JSON"
        inventory = await read_kubectl(ctx, COMMAND)

    assert "frontend-tail" in snapshot
    assert "unusable" not in caplog.text
    assert inventory == PAYLOAD
    assert not (ctx.workspace.path / "tool_outputs").exists()


@pytest.mark.parametrize("save_as", [None, "inventory.json"])
async def test_large_kubectl_output_is_saved_complete_at_the_tool_boundary(
    ctx: ToolContext, save_as: str | None
) -> None:
    """Both automatic saving and save_as need the full payload from the transport."""
    with respx.mock as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": PAYLOAD})
        )
        shown = await exec_kubectl_cmd_safely(ctx, COMMAND, save_as=save_as)

    saved_name = save_as or "kubectl_0001.json"
    saved = (ctx.workspace.outputs / saved_name).read_text()
    assert json.loads(saved) == PODS
    assert f"tool_outputs/{saved_name}" in shown
    assert len(PAYLOAD) > OUTPUT_LIMIT
    if save_as is None:
        assert shown.startswith(PAYLOAD[:OUTPUT_LIMIT])
        assert f"Output truncated ({len(PAYLOAD)} chars)" in shown
    else:
        assert saved == PAYLOAD
        assert shown.startswith(f"[Saved {len(PAYLOAD)} chars")


async def test_large_traces_are_scoped_before_display_truncation(ctx: ToolContext) -> None:
    """Cutting a successful Jaeger response must not disable namespace filtering."""
    ctx.scope.namespace = NAMESPACE
    ctx.scope.trace_service_candidates = frozenset({"frontend"})
    traces = [
        {
            "traceID": "large-trace",
            "processes": {"p1": {"serviceName": "frontend"}, "p2": {"serviceName": "foreign"}},
            "spans": [
                {"processID": "p1", "operationName": "x" * (OUTPUT_LIMIT + 256)},
                {"processID": "p2", "operationName": "foreign-secret"},
            ],
        }
    ]
    with respx.mock as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": json.dumps(traces)})
        )
        shown = await get_traces(ctx, "frontend", 10)
    assert "Output truncated" in shown
    saved = (ctx.workspace.outputs / "jaeger_traces_0001.json").read_text()
    parsed = json.loads(saved)
    assert parsed[0]["spans"] == traces[0]["spans"][:1]
    assert parsed[0]["processes"] == {"p1": {"serviceName": "frontend"}}
    assert "foreign-secret" not in saved
    assert "foreign-secret" not in shown
