"""What the checking role is shown, and the five ways it is refused.

The opening message is pinned to `tests/fixtures/prototype_role_messages.json`, rendered
by the prototype's own renderer with the ACE and benchmark-contract blocks empty — the
configuration every published measurement ran.
"""

import asyncio
import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import ExecutionResult, Plan, RootCause
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.roles import verify
from kira.roles.base import RoleContext
from kira.roles.health_gate import HealthGate
from kira.settings.groups import MemorySettings, PipelineSettings, VerifySettings
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


def context(tmp_path: pathlib.Path, *, health_gate: bool = False) -> tuple[RoleContext, FakeModel]:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="verify")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="verify",
        session_id="s1",
        journal=journal,
    )
    model = FakeModel(RECOVERED)
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


def test_a_gate_that_could_not_look_does_not_block_recovery() -> None:
    """Fails open at the guard as well as inside the sweep: an unavailable gate that
    manufactured a not-recovered verdict would turn an infrastructure fault into a false
    incident and spend the retry budget on nothing."""
    assert (
        verify._build_submit_guard(HealthGate(checked=False, error="timed out"))(RECOVERED, 1, [])
        is None
    )
