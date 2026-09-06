"""The minimal baseline's two-phase contract.

What is pinned here is the shape of the arm rather than any rendered text: one run over
one transcript, two separately graded phases, and a mitigation phase that still reports
a result when it never finalises — because the benchmark grades the cluster, not the
submission.
"""

import asyncio
import json
import pathlib
from typing import Any

from kira.audit.journal import Journal, ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.roles import single
from kira.roles.base import RoleContext
from kira.settings.groups import MemorySettings, PipelineSettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.toolsets import build_toolset
from kira.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ALERT = {"alertname": "HighErrorRate", "namespace": "shop", "_problem_id": "leak-the-answer"}
DIAGNOSIS = {"solution": "svc/user targets a port nothing listens on"}
MITIGATION = {"succeeded": True, "summary": "patched targetPort to 8080; 5xx cleared"}
FINALIZERS = {"submit_diagnosis", "submit_mitigation"}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "targetPort: 8081"


class ScriptedModel:
    """Replays a scripted call per turn and records the tools it was offered."""

    def __init__(self, *script: tuple[str, dict[str, Any]] | None) -> None:
        self.script = list(script)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        step = self.script.pop(0) if self.script else None
        if step is None:
            return ModelResponse(
                text="I am not going to call anything.",
                tool_calls=(),
                finish_reason="stop",
                reasoning=None,
                usage=Usage(),
                duration_ms=1,
                attempts=1,
            )
        name, arguments = step
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


def context(tmp_path: pathlib.Path, model: ScriptedModel) -> RoleContext:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="single")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="single",
        session_id="s1",
        journal=journal,
    )
    return RoleContext(
        spec=single.SINGLE,
        state={"session_id": "s1", "alert": ALERT},
        handback=None,
        settings=Settings(
            pipeline=PipelineSettings(),
            memory=MemorySettings(enabled=False, learning_enabled=False),
        ),
        completion=model,
        toolset=build_toolset("single", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        write_lock=asyncio.Lock(),
        run_id="r1",
        session_id="s1",
    )


def offered(request: ModelRequest) -> set[str]:
    return {tool.name for tool in request.tools}


async def test_the_second_phase_continues_the_first_phase_transcript(
    tmp_path: pathlib.Path,
) -> None:
    """Continuity is the point of the arm: a baseline that re-read the namespace from
    scratch for its remediation would be a different and weaker system."""
    model = ScriptedModel(("submit_diagnosis", DIAGNOSIS), ("submit_mitigation", MITIGATION))
    ctx = context(tmp_path, model)

    outcome = await single.SINGLE.run(ctx)

    opening, second = model.requests[0].messages, model.requests[1].messages
    assert len(opening) == 1
    assert second[: len(opening)] == opening
    assert second[-1]["content"].startswith("TASK 1 diagnosis submitted.")
    assert outcome.delta["root_cause"].solution == DIAGNOSIS["solution"]
    assert outcome.delta["execution_result"].succeeded is True


async def test_only_one_finalizer_is_terminal_per_phase(tmp_path: pathlib.Path) -> None:
    """Offering both lets the model short-circuit the sequence by submitting a
    mitigation it has not performed while still diagnosing."""
    model = ScriptedModel(("submit_diagnosis", DIAGNOSIS), ("submit_mitigation", MITIGATION))
    ctx = context(tmp_path, model)

    await single.SINGLE.run(ctx)

    assert offered(model.requests[0]) & FINALIZERS == {"submit_diagnosis"}
    assert offered(model.requests[1]) & FINALIZERS == {"submit_mitigation"}


async def test_an_unfinalised_mitigation_still_reports_a_result(
    tmp_path: pathlib.Path,
) -> None:
    """The agent may already have applied changes, and the benchmark grades the cluster
    rather than the submission — so the remediation slot is filled either way."""
    model = ScriptedModel(("submit_diagnosis", DIAGNOSIS))
    ctx = context(tmp_path, model)

    outcome = await single.SINGLE.run(ctx)

    assert outcome.delta["execution_result"].succeeded is False
    assert "not finalized" in outcome.delta["execution_result"].summary


async def test_a_failed_diagnosis_ends_the_run_with_nothing(tmp_path: pathlib.Path) -> None:
    model = ScriptedModel()
    ctx = context(tmp_path, model)

    outcome = await single.SINGLE.run(ctx)

    assert outcome.artifact is None
    assert outcome.error is not None and outcome.error.startswith("single-agent diagnosis failed")
    assert outcome.delta == {}
