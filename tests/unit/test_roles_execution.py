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
from kira.contracts.events import InterventionEvent
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.prompts.loader import read_runtime
from kira.roles import execution
from kira.roles.base import RoleContext
from kira.settings.groups import MemorySettings, PipelineSettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.toolsets import build_toolset
from kira.tools.workspace import SessionWorkspace

PROFILES = {
    name: load_profile(name, pathlib.Path("configs/models"))
    for name in ("qwen3.6-35b-a3b", "qwen3.8-27b", "claude-sonnet-4-6")
}
PROFILE = PROFILES["qwen3.6-35b-a3b"]

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
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "targetPort: 8081"


READ_FIRST = ("exec_kubectl_cmd_safely", {"command": "kubectl get svc user -o yaml"})


class FakeModel:
    """Reads the target, then submits — the shortest run the guard will accept."""

    def __init__(
        self, *submissions: dict[str, Any], delay: float = 0.0, opening_read: bool = True
    ) -> None:
        self.submissions = list(submissions)
        self.requests: list[ModelRequest] = []
        self.delay = delay
        self.opening_read = opening_read

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.opening_read and len(self.requests) == 1:
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
    resume: list[dict[str, Any]] | None = None,
    profile: Any = PROFILE,
    sink: ListSink | None = None,
    transport: Any = None,
) -> tuple[RoleContext, FakeModel]:
    sink = sink or ListSink()
    journal = Journal(sink, session_id="s1", stage="execution")
    tools = ToolContext(
        transport=transport or FakeTransport(),
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
        profile=profile,
        middleware=MiddlewareStack(),
        write_lock=write_lock or asyncio.Lock(),
        run_id="r1",
        session_id="s1",
        resume=resume,
    )
    return ctx, completion


@pytest.mark.parity
async def test_the_opening_turn_carries_the_diagnosis_and_the_approved_plan(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model = context(tmp_path)

    outcome = await execution.EXECUTION.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["execution_first_entry"]
    assert outcome.artifact is not None


async def test_the_system_prompt_follows_the_route_the_transport_offers(
    tmp_path: pathlib.Path,
) -> None:
    """The prompt and the tools are two halves of one request and have to agree.

    On the proxy route the sandbox resolves kubectl to a stage-aware shim, so guidance
    saying it has no cluster credentials is a contradiction the model can disprove in
    one command — and the role composes its own prompt, so the route has to reach it
    there and not only at the tool.
    """

    class Proxied(FakeTransport):
        kubectl_proxy_url = "http://127.0.0.1:16444"

    unproxied_ctx, unproxied = context(tmp_path)
    await execution.EXECUTION.run(unproxied_ctx)
    proxied_ctx, proxied = context(tmp_path, transport=Proxied())
    await execution.EXECUTION.run(proxied_ctx)

    assert "run normal `kubectl` commands directly" not in unproxied.requests[0].system
    assert "run normal `kubectl` commands directly" in proxied.requests[0].system


@pytest.mark.parity
def test_a_stopped_kubelet_gets_the_directive_and_the_least_privileged_repair() -> None:
    """Cordoning, draining or patching cannot restore a node whose kubelet is down, so
    that one case opens on the host action instead of on a Kubernetes read."""
    assert (
        execution._initial_message(PLAN, KUBELET_ROOT_CAUSE) == FIXTURE["execution_kubelet_entry"]
    )
    assert execution.initial_execution_tool_choice(
        KUBELET_ROOT_CAUSE, {"ssh_exec", "exec_kubectl_cmd_safely"}
    ) == {"type": "function", "function": {"name": "ssh_exec"}}


@pytest.mark.parametrize("profile_name", sorted(PROFILES))
async def test_an_ordinary_incident_is_forced_to_open_on_a_read(
    tmp_path: pathlib.Path, profile_name: str
) -> None:
    """The model cannot open with a mutation. This is one of the four code properties
    that make Check-and-Set followable rather than merely instructed.

    Over every shipped profile, because one of them cannot carry it: the serving stack
    behind `qwen3.8-27b` requires `tool_choice: auto`, which is a request-level setting
    and not a per-turn one, so the forced choice is not sent. That is the profile the
    denser-model comparison runs on, and a property the README claims was being lost
    with no event, no warning and no test.
    """
    profile = PROFILES[profile_name]
    sink = ListSink()
    ctx, model = context(tmp_path, profile=profile, sink=sink)

    await execution.EXECUTION.run(ctx)

    dropped = [
        e.payload
        for e in sink.events
        if isinstance(e.payload, InterventionEvent) and e.payload.control == "tool_choice"
    ]
    if profile.quirks.tool_choice == "auto":
        assert model.requests[0].tool_choice == "auto"
        assert [(d.action, profile_name in d.detail) for d in dropped] == [("bypassed", True)]
    else:
        assert model.requests[0].tool_choice == {
            "type": "function",
            "function": {"name": "exec_kubectl_cmd_safely"},
        }
        assert dropped == []


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


REFUSAL = read_runtime("execution", "submit_without_command.txt")
"""What the guard hands back to a model that submitted having run nothing."""


def submit_results(model: FakeModel, submit_tool: str) -> list[str]:
    """What the model was told about each of its own submissions, in order."""
    messages = model.requests[-1].messages
    answered = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls", [])
        if call["function"]["name"] == submit_tool
    }
    return [str(m["content"]) for m in messages if m.get("tool_call_id") in answered]


async def test_a_submission_the_spent_guard_never_saw_is_recorded_as_one(
    tmp_path: pathlib.Path,
) -> None:
    """The guard is a budget, and a submission that outlasts it is accepted.

    That acceptance is the behaviour every published measurement was taken under, so it
    is pinned here rather than fixed: this run files a `succeeded` remediation having
    dispatched no command at all. What it cannot also do is look like a submission the
    guard approved — the artifact, the submission event and the run status are identical
    for both — so the run row and the event log carry the bypass and what it overrode.
    """
    sink = ListSink()
    ctx, model = context(
        tmp_path, model=FakeModel(*[VALID_RESULT] * 3, opening_read=False), sink=sink
    )

    outcome = await execution.EXECUTION.run(ctx)

    assert outcome.artifact is not None and outcome.artifact.succeeded is True
    guard = outcome.metadata["submit_guard"]
    assert guard["rejections"] == 2 and guard["bypassed"] is True
    assert REFUSAL.startswith(guard["finding"].removesuffix(" ..."))
    (bypass,) = [
        e.payload
        for e in sink.events
        if isinstance(e.payload, InterventionEvent) and e.payload.control == "submit_guard"
    ]
    assert bypass.action == "bypassed"
    assert "execution" in bypass.detail
    assert "2 rejections" in bypass.detail
    assert guard["finding"] in bypass.detail
    # The record changed; what the model was told did not. These are held-constant
    # inputs to every published measurement, so the asset is the expectation.
    assert submit_results(model, "submit_execution_result") == [f"error: {REFUSAL}"] * 2


async def test_a_run_the_guard_passed_carries_no_bypass(tmp_path: pathlib.Path) -> None:
    """The entry only means something if the ordinary run says the opposite."""
    ctx, _ = context(tmp_path)

    outcome = await execution.EXECUTION.run(ctx)

    assert outcome.artifact is not None
    assert outcome.metadata["submit_guard"] == {"rejections": 0, "bypassed": False}


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


APPLIED_WRITE: list[dict[str, Any]] = [
    {"role": "user", "content": "apply the plan"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "w1",
                "type": "function",
                "function": {
                    "name": "exec_kubectl_cmd_safely",
                    "arguments": json.dumps({"command": "kubectl patch svc user -p '{}'"}),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "w1",
        "name": "exec_kubectl_cmd_safely",
        "content": "service/user patched",
    },
]
"""The transcript a continuation opens on: the approved write, and its replayed result."""


async def test_a_resumed_run_may_submit_on_the_turn_after_the_approved_write(
    tmp_path: pathlib.Path,
) -> None:
    """The guard counts commands, and a continuation's commands are behind it.

    The count restarted at zero on a resume while the transcript in front of the model
    held the write and its result, so the first submit after an approval was refused
    with text saying no command had been run and instructing the model to apply the
    planned mutation. A model that complies reissues a change the cluster has taken and
    parks the incident on a second decision for it.
    """
    # Three submissions, so a run that rejects the first two and reaches the guard's
    # bypass fails on the assertion below rather than on an exhausted script.
    ctx, model = context(
        tmp_path, model=FakeModel(*[VALID_RESULT] * 3, opening_read=False), resume=APPLIED_WRITE
    )

    outcome = await execution.EXECUTION.run(ctx)

    assert outcome.artifact is not None
    # One turn: a rejection would have handed the submission back and taken another.
    assert len(model.requests) == 1
