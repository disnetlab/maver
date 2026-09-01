"""What the applying role opens with, what it refuses, and what it serialises.

The rendered message is pinned to `tests/fixtures/prototype_role_messages.json`, built by
importing the prototype's renderer and its own `_requires_kubelet_recovery` with the
playbook feature disabled, so the expectation comes from the prototype rather than from
this implementation.
"""

import asyncio
import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import Plan, RootCause
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.roles import execution
from kira.roles.base import RoleContext
from kira.settings.groups import MemorySettings, PipelineSettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.toolsets import build_toolset
from kira.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

FIXTURE = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_role_messages.json"
    ).read_text(encoding="utf-8")
)

PLAN = Plan.model_validate(FIXTURE["execution_plan"])
ROOT_CAUSE = RootCause.model_validate(FIXTURE["root_cause"])
KUBELET_ROOT_CAUSE = RootCause.model_validate(FIXTURE["execution_kubelet_root_cause"])

VALID_RESULT: dict[str, Any] = {
    "succeeded": True,
    "steps": [
        {
            "step_index": 0,
            "succeeded": True,
            "observation": "targetPort was 8081; patched to 8080; endpoints now non-empty",
        }
    ],
    "final_state_summary": "svc/user has endpoints and the 5xx rate fell",
    "confidence": 0.8,
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "targetPort: 8081"


READ_FIRST = ("exec_kubectl_cmd_safely", {"command": "kubectl get svc user -o yaml"})


class FakeModel:
    """Reads the target, then submits — the shortest run the guard will accept."""

    def __init__(self, *submissions: dict[str, Any], delay: float = 0.0) -> None:
        self.submissions = list(submissions)
        self.requests: list[ModelRequest] = []
        self.delay = delay

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if len(self.requests) == 1:
            name, arguments = READ_FIRST
        else:
            name, arguments = "submit_execution_result", self.submissions.pop(0)
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{len(self.requests)}",
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


def context(
    tmp_path: pathlib.Path,
    *,
    root_cause: RootCause | None = ROOT_CAUSE,
    plan: Plan | None = PLAN,
    model: FakeModel | None = None,
    write_lock: asyncio.Lock | None = None,
) -> tuple[RoleContext, FakeModel]:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="execution")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=journal,
    )
    completion = model or FakeModel(VALID_RESULT)
    ctx = RoleContext(
        spec=execution.EXECUTION,
        state={
            "session_id": "s1",
            "alert": FIXTURE["alert"],
            "root_cause": root_cause,
            "plan": plan,
        },
        handback=None,
        settings=Settings(
            pipeline=PipelineSettings(),
            memory=MemorySettings(enabled=False, learning_enabled=False),
        ),
        completion=completion,
        toolset=build_toolset("execution", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        write_lock=write_lock or asyncio.Lock(),
        run_id="r1",
        session_id="s1",
    )
    return ctx, completion


async def test_the_opening_turn_carries_the_diagnosis_and_the_approved_plan(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model = context(tmp_path)

    outcome = await execution.EXECUTION.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["execution_first_entry"]
    assert outcome.artifact is not None


def test_a_stopped_kubelet_gets_the_directive_and_the_least_privileged_repair() -> None:
    """Cordoning, draining or patching cannot restore a node whose kubelet is down, so
    that one case opens on the host action instead of on a Kubernetes read."""
    assert (
        execution._initial_message(PLAN, KUBELET_ROOT_CAUSE) == FIXTURE["execution_kubelet_entry"]
    )
    assert execution.initial_execution_tool_choice(
        KUBELET_ROOT_CAUSE, {"ssh_exec", "exec_kubectl_cmd_safely"}
    ) == {"type": "function", "function": {"name": "ssh_exec"}}


async def test_an_ordinary_incident_is_forced_to_open_on_a_read(tmp_path: pathlib.Path) -> None:
    """The model cannot open with a mutation. This is one of the four code properties
    that make Check-and-Set followable rather than merely instructed."""
    ctx, model = context(tmp_path)

    await execution.EXECUTION.run(ctx)

    assert model.requests[0].tool_choice == {
        "type": "function",
        "function": {"name": "exec_kubectl_cmd_safely"},
    }


@pytest.mark.parametrize(
    ("fault_type", "summary", "routed"),
    [
        ("node_failure", "kubelet on worker-2 is stopped", True),
        # A node failure can still be storage, network or the cloud host, and none of
        # those are repaired by restarting a kubelet.
        ("node_failure", "the node's disk is full", False),
        # Kubelet evidence without the node classification is not enough either.
        ("misconfiguration", "the kubelet flag is wrong and not ready", False),
    ],
)
def test_the_host_repair_needs_both_the_classification_and_the_evidence(
    fault_type: str, summary: str, routed: bool
) -> None:
    root_cause = RootCause(
        fault_type=fault_type,
        affected_resource="node/worker-2",
        confidence=0.8,
        evidence=[summary],
        summary=summary,
    )
    assert execution.requires_kubelet_recovery(root_cause) is routed


@pytest.mark.parametrize(
    ("arguments", "marker"),
    [
        ({}, "before running a single command"),
        ({"final_state_summary": "  "}, "non-empty final_state_summary"),
        ({"final_state_summary": "patched", "steps": []}, "at least one step observation"),
    ],
)
def test_the_submit_guard_refuses_a_remediation_that_was_never_attempted(
    arguments: dict[str, Any], marker: str
) -> None:
    """Three checks in order. The first is the one that matters: a submit on turn zero
    discards the diagnosis and the plan and files an empty mitigation."""
    calls_made = 0 if not arguments else 1
    rejection = execution._submit_guard(arguments, calls_made, [])
    assert rejection is not None and marker in rejection


async def test_an_empty_plan_short_circuits_without_a_model_call(tmp_path: pathlib.Path) -> None:
    empty = Plan(steps=[], expected_outcome="nothing to do", confidence=0.1)
    ctx, model = context(tmp_path, plan=empty)

    outcome = await execution.EXECUTION.run(ctx)

    assert model.requests == []
    assert outcome.artifact is not None
    assert outcome.artifact.succeeded is False


async def test_two_remediations_never_overlap(tmp_path: pathlib.Path) -> None:
    """The sole-writer guarantee. Two runs sharing one lock must not interleave: without
    it they can issue conflicting changes, one rolling back while the other rolls
    forward."""
    lock = asyncio.Lock()
    live = 0
    peak = 0

    class Counting(FakeModel):
        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            try:
                return await super().complete(request, caller=caller)
            finally:
                live -= 1

    first, _ = context(tmp_path / "a", model=Counting(VALID_RESULT, delay=0.02), write_lock=lock)
    second, _ = context(tmp_path / "b", model=Counting(VALID_RESULT, delay=0.02), write_lock=lock)

    await asyncio.gather(execution.EXECUTION.run(first), execution.EXECUTION.run(second))

    assert peak == 1
