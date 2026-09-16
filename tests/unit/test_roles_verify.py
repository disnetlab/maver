"""What the checking role is shown, and the five ways it is refused.

The opening message is pinned to `tests/fixtures/prototype_role_messages.json`, rendered
by the prototype's own renderer with the ACE and benchmark-contract blocks empty — the
configuration every published measurement ran.
"""

import asyncio
import json
import pathlib
from dataclasses import replace
from typing import Any

import pytest

from maver.audit.journal import Journal, ListSink
from maver.contracts.artifacts import ExecutionResult, Plan, RootCause
from maver.contracts.events import InterventionEvent
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.llm.types import ToolSpec as ModelToolSpec
from maver.middleware.base import MiddlewareStack
from maver.prompts.loader import markers, read_runtime, render
from maver.roles import verify
from maver.roles.base import RoleContext
from maver.roles.health_gate import HealthGate
from maver.settings.groups import MemorySettings, PipelineSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools.context import ToolContext
from maver.tools.toolsets import ResolvedToolset, build_toolset
from maver.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

FIXTURE = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "prototype_role_messages.json"
    ).read_text(encoding="utf-8")
)

RECOVERED: dict[str, Any] = {
    "recovered": True,
    "verdict": "recovered",
    "checks": ["curl http://frontend/health through the gateway returned 200"],
    "summary": "the 5xx rate is back to baseline and the entry point serves traffic",
    "entry_point_status": "HTTP 200 via service/frontend",
    "confidence": 0.8,
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "no resources found"


class FakeModel:
    """Runs one read, then submits — the shortest run the guard will accept."""

    def __init__(self, *submissions: dict[str, Any]) -> None:
        self.submissions = list(submissions)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            name: str = "exec_kubectl_cmd_safely"
            arguments: dict[str, Any] = {"command": "kubectl get pods"}
        else:
            name, arguments = "submit_verify_result", self.submissions.pop(0)
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
    health_gate: bool = False,
    model: FakeModel | None = None,
    transport: Any = None,
    sink: ListSink | None = None,
) -> tuple[RoleContext, FakeModel]:
    sink = sink or ListSink()
    journal = Journal(sink, session_id="s1", stage="verify")
    tools = ToolContext(
        transport=transport or FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="verify",
        session_id="s1",
        journal=journal,
    )
    model = model or FakeModel(RECOVERED)
    ctx = RoleContext(
        spec=verify.VERIFY,
        state={
            "session_id": "s1",
            "alert": FIXTURE["alert"],
            "root_cause": RootCause.model_validate(FIXTURE["root_cause"]),
            "plan": Plan.model_validate(FIXTURE["execution_plan"]),
            "execution_result": ExecutionResult(
                succeeded=True,
                steps=[],
                final_state_summary="patched svc/user",
                confidence=0.8,
            ),
        },
        handback=None,
        settings=Settings(
            pipeline=PipelineSettings(),
            memory=MemorySettings(enabled=False, learning_enabled=False),
            verify=VerifySettings(enabled=True, health_gate=health_gate),
        ),
        completion=model,
        toolset=build_toolset("verify", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        write_lock=asyncio.Lock(),
        run_id="r1",
        session_id="s1",
    )
    return ctx, model


@pytest.mark.parity
async def test_the_opening_turn_says_nothing_about_what_was_diagnosed_or_applied(
    tmp_path: pathlib.Path,
) -> None:
    """Independence by construction rather than by instruction. The state holds a
    diagnosis, a plan and a remediation; none of them reaches the model."""
    ctx, model = context(tmp_path)

    outcome = await verify.VERIFY.run(ctx)

    opening = model.requests[0].messages[0]["content"]
    assert opening == FIXTURE["verify_first_entry"]
    assert "targetPort" not in opening and "patch" not in opening
    assert outcome.artifact is not None


BLOCKING = HealthGate(checked=True, failures=("pod/user-1 is CrashLoopBackOff",))


@pytest.mark.parametrize(
    ("arguments", "calls", "gate", "refused"),
    [
        (RECOVERED, 0, HealthGate(checked=False), "before running a single check"),
        ({**RECOVERED, "summary": " "}, 1, HealthGate(checked=False), "non-empty summary"),
        ({**RECOVERED, "checks": []}, 1, HealthGate(checked=False), "at least one entry"),
        (
            {**RECOVERED, "entry_point_status": ""},
            1,
            HealthGate(checked=False),
            "without a non-empty entry_point_status",
        ),
        # The floor: recovery is unavailable while the namespace is measurably broken.
        (RECOVERED, 1, BLOCKING, "CrashLoopBackOff"),
    ],
)
def test_the_guard_refuses_a_verdict_it_cannot_stand_behind(
    arguments: dict[str, Any], calls: int, gate: HealthGate, refused: str
) -> None:
    rejection = verify._build_submit_guard(gate)(arguments, calls, [])
    assert rejection is not None and refused in rejection


UNHEALTHY_POD = "pod user-1 is Running but container(s) user are not Ready"

UNHEALTHY_NAMESPACE = json.dumps(
    {
        "items": [
            {
                "metadata": {"name": "user-1"},
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"name": "user", "ready": False, "restartCount": 7}],
                },
            }
        ]
    }
)
"""One pod the sweep must call unhealthy, in the shape `kubectl get pods -o json` returns."""


class UnhealthyTransport(FakeTransport):
    async def kubectl(self, argv: list[str]) -> str:
        return UNHEALTHY_NAMESPACE


def submit_results(model: FakeModel) -> list[str]:
    """What the model was told about each of its own submissions, in order."""
    messages = model.requests[-1].messages
    answered = {
        call["id"]
        for message in messages
        for call in message.get("tool_calls", [])
        if call["function"]["name"] == "submit_verify_result"
    }
    return [str(m["content"]) for m in messages if m.get("tool_call_id") in answered]


async def test_a_recovered_verdict_the_spent_guard_never_saw_is_recorded_as_one(
    tmp_path: pathlib.Path,
) -> None:
    """The health gate is a budget too, and the third `recovered` closes the incident.

    The gate found a broken pod and refused the verdict twice; the third submission is
    accepted unconsulted, so this run closes a namespace the gate says is unhealthy.
    That is the measured behaviour and it stands. The gate's own audit entry cannot
    stand in for the record of it: it says the gate blocked, which is also what it says
    on a run where the block held and the model revised.
    """
    sink = ListSink()
    ctx, model = context(
        tmp_path,
        health_gate=True,
        model=FakeModel(*[RECOVERED] * 3),
        transport=UnhealthyTransport(),
        sink=sink,
    )

    outcome = await verify.VERIFY.run(ctx)

    assert outcome.artifact is not None and outcome.artifact.recovered is True
    assert outcome.metadata["gate"]["blocked_recovered"] is True
    guard = outcome.metadata["submit_guard"]
    refused = render(read_runtime("verify", "submit_while_unhealthy.txt"), {"FAILURES": ""})
    assert guard == {
        "rejections": 2,
        "bypassed": True,
        "finding": refused.splitlines()[0],
    }
    (bypass,) = [
        e.payload
        for e in sink.events
        if isinstance(e.payload, InterventionEvent) and e.payload.control == "submit_guard"
    ]
    assert bypass.action == "bypassed"
    assert "verify" in bypass.detail
    assert "2 rejections" in bypass.detail
    assert guard["finding"] in bypass.detail
    # The record changed; what the model was told did not.
    expected = render(
        read_runtime("verify", "submit_while_unhealthy.txt"), {"FAILURES": f"  - {UNHEALTHY_POD}"}
    )
    assert submit_results(model) == [f"error: {expected}"] * 2


def test_a_gate_that_could_not_look_does_not_block_recovery() -> None:
    """Fails open at the guard as well as inside the sweep: an unavailable gate that
    manufactured a not-recovered verdict would turn an infrastructure fault into a false
    incident and spend the retry budget on nothing."""
    assert (
        verify._build_submit_guard(HealthGate(checked=False, error="timed out"))(RECOVERED, 1, [])
        is None
    )


def test_the_blind_context_is_filled_from_the_alert_and_nothing_environment_specific() -> None:
    """The template carried ``{{AIOPSLAB_VERIFY_CONNECTIVITY_CONTRACT}}``, a slot named
    for a benchmark that ships nowhere in this repository and that no adapter could have
    filled: the role hard-coded it empty. It therefore rendered nothing, and the opening
    turn above still matches the prototype fixture byte for byte with it gone. What is
    pinned here is that it may not come back: templating is strict in both directions, so
    the marker set is the whole of the contract between the template and the role."""
    assert markers(read_runtime("verify", "context_blind.txt")) == {
        "PROCEDURAL_CONTEXT",
        "ALERT_JSON",
    }


class ProbeModel(FakeModel):
    """A real role loop with a synthetic completed Verify-only probe."""

    def __init__(self, *submissions: dict[str, Any], probe_name: str = "probe_service") -> None:
        super().__init__(*submissions)
        self.probe_name = probe_name

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        response = await super().complete(request, caller=caller)
        if len(self.requests) == 1:
            return replace(
                response,
                tool_calls=(replace(response.tool_calls[0], name=self.probe_name),),
            )
        return response


def probe_context(
    tmp_path: pathlib.Path,
    *submissions: dict[str, Any],
    output: str = "HTTP 200 via service/frontend:8080/health",
    probe_name: str = "probe_service",
    health_gate: bool = False,
) -> tuple[RoleContext, ProbeModel, ListSink]:
    sink = ListSink()
    model = ProbeModel(*submissions, probe_name=probe_name)
    ctx, _ = context(
        tmp_path,
        model=model,
        sink=sink,
        health_gate=health_gate,
        transport=UnhealthyTransport() if health_gate else None,
    )

    async def probe(**kwargs: Any) -> str:
        return output

    offered = ModelToolSpec(name=probe_name, description="Probe.", parameters={"type": "object"})
    return (
        replace(
            ctx,
            toolset=ResolvedToolset(
                definitions=[offered, *ctx.toolset.definitions],
                dispatch={**ctx.toolset.dispatch, probe_name: probe},
            ),
        ),
        model,
        sink,
    )


def normalization_events(sink: ListSink) -> list[InterventionEvent]:
    return [
        event.payload
        for event in sink.events
        if isinstance(event.payload, InterventionEvent)
        and event.payload.control == "submit_normalization"
    ]


@pytest.mark.parametrize("status", ["200", "302"])
async def test_recovered_omission_is_anchored_to_completed_probe_without_rewriting_verdict(
    tmp_path: pathlib.Path, status: str
) -> None:
    submitted = {key: value for key, value in RECOVERED.items() if key != "entry_point_status"}
    observed = f"HTTP {status} via service/frontend:8080/health"
    ctx, model, sink = probe_context(tmp_path, submitted, output=observed)

    result = await verify.VERIFY.run(ctx)

    assert result.artifact is not None
    assert result.artifact.recovered is True and result.artifact.verdict == "recovered"
    assert result.artifact.entry_point_status == observed
    assert len(model.requests) == 2
    assert result.metadata["submit_guard"]["rejections"] == 0
    (event,) = normalization_events(sink)
    detail = json.loads(event.detail)
    assert event.action == "rescued"
    assert detail["raw_arguments"] == submitted
    assert detail["normalized_arguments"] == {**submitted, "entry_point_status": observed}
    assert detail["repairs"] == [
        {"field": "entry_point_status", "source": "last_successful_probe_service_result"}
    ]
    original_call = result.transcript[-2]["tool_calls"][0]
    assert json.loads(original_call["function"]["arguments"]) == submitted


@pytest.mark.parametrize(
    ("probe_name", "output"),
    [
        ("probe_service", "error: unavailable; previous HTTP 200 via service/frontend"),
        ("probe_service", "HTTP 500 via service/frontend"),
        ("get_logs", "HTTP 200 via service/frontend"),
    ],
)
async def test_a_failed_or_unrelated_tool_cannot_supply_the_recovery_anchor(
    tmp_path: pathlib.Path, probe_name: str, output: str
) -> None:
    missing = {**RECOVERED, "entry_point_status": ""}
    ctx, model, sink = probe_context(
        tmp_path, missing, RECOVERED, output=output, probe_name=probe_name
    )

    result = await verify.VERIFY.run(ctx)

    assert (
        result.artifact is not None
        and result.artifact.entry_point_status == RECOVERED["entry_point_status"]
    )
    assert len(model.requests) == 3
    assert result.metadata["submit_guard"]["rejections"] == 1
    assert normalization_events(sink) == []


@pytest.mark.parametrize("route", [None, "replan", "reassessment"])
async def test_not_recovered_fills_only_route_omissions_from_submitted_evidence(
    tmp_path: pathlib.Path, route: str | None
) -> None:
    submitted = {
        "recovered": False,
        "verdict": "not_recovered",
        "summary": "Frontend requests still fail.",
        "checks": [
            "curl returned HTTP 500",
            "traces show database errors",
            "alerts remain firing",
            "fourth check is omitted from the reason",
        ],
        "confidence": 0.8,
    }
    if route is not None:
        submitted[f"needs_{route}"] = True
    ctx, model, sink = probe_context(tmp_path, submitted, output="HTTP 500 via service/frontend")

    result = await verify.VERIFY.run(ctx)

    assert result.artifact is not None
    assert result.artifact.recovered is False and result.artifact.verdict == "not_recovered"
    assert result.artifact.needs_replan is (route == "replan")
    assert result.artifact.needs_reassessment is (route != "replan")
    evidence = " ".join([submitted["summary"], *submitted["checks"][:3]])
    expected = (
        evidence
        if route is not None
        else ("Verify submitted not_recovered. Evidence from its own checks: " + evidence)
    )
    assert getattr(result.artifact, f"{route or 'reassessment'}_reason") == expected
    assert len(model.requests) == 2
    (event,) = normalization_events(sink)
    assert json.loads(event.detail)["raw_arguments"] == submitted


async def test_normalized_recovery_still_passes_through_the_existing_health_gate(
    tmp_path: pathlib.Path,
) -> None:
    missing = {**RECOVERED, "entry_point_status": ""}
    ctx, model, sink = probe_context(tmp_path, *[missing] * 3, health_gate=True)

    result = await verify.VERIFY.run(ctx)

    assert result.artifact is not None
    assert result.metadata["gate"]["blocked_recovered"] is True
    assert result.metadata["submit_guard"]["rejections"] == 2
    assert result.metadata["submit_guard"]["bypassed"] is True
    assert len(normalization_events(sink)) == 3
    assert all("not Ready" in reply for reply in submit_results(model))


def test_normalizer_uses_the_latest_success_and_bounds_the_models_own_evidence() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "first", "function": {"name": "probe_service"}}],
        },
        {"role": "tool", "tool_call_id": "first", "content": "HTTP 200 via service/first/"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "second", "function": {"name": "probe_service"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "second",
            "content": "HTTP 301 via service/second:80/health",
        },
        {"role": "assistant", "tool_calls": [{"id": "logs", "function": {"name": "get_logs"}}]},
        {
            "role": "tool",
            "tool_call_id": "logs",
            "name": "probe_service",
            "content": "HTTP 200 via service/unrelated/",
        },
    ]
    args, _ = verify._normalize_verify_submit_args(
        {**RECOVERED, "entry_point_status": ""}, messages
    )
    assert args["entry_point_status"] == "HTTP 301 via service/second:80/health"
    evidence = {
        "recovered": False,
        "verdict": "not_recovered",
        "needs_replan": True,
        "summary": "a" * 1900,
        "checks": ["curl failed"],
    }
    args, _ = verify._normalize_verify_submit_args(evidence, messages)
    assert args["replan_reason"] == "a" * 1800
