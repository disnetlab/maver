"""What the remediating role opens with, and the one submission it refuses.

The rendered messages are pinned to `tests/fixtures/prototype_role_messages.json`, built
by importing the prototype's `agent.agents.planning._replan_block` and its renderer with
the playbook and ACE features disabled, so the expectation comes from the prototype
rather than from this implementation.
"""

import asyncio
import json
import pathlib
from typing import Any

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import RootCause
from kira.contracts.routing import HandBack, Route
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.roles import planning
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

VALID_PLAN: dict[str, Any] = {
    "steps": [
        {
            "action": "patch",
            "target": "svc/user",
            "params": {"targetPort": 8080},
            "justification": "point the Service at the port the container listens on",
        }
    ],
    "expected_outcome": "svc/user has endpoints and the 5xx rate falls",
    "confidence": 0.6,
}
EMPTY_PLAN: dict[str, Any] = {"steps": [], "expected_outcome": "nothing", "confidence": 0.1}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces", "host"})
    timeout_seconds = 30


class FakeStore:
    """Stands in for the experience store, so the retrieval tools survive resolution."""


class FakeModel:
    """Replays scripted submissions and records what it was sent."""

    def __init__(self, *submissions: dict[str, Any]) -> None:
        self.submissions = list(submissions)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        arguments = self.submissions.pop(0)
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id=f"c{len(self.requests)}",
                    name="submit_plan",
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
    memory: bool = False,
    handback: HandBack | None = None,
    state: dict[str, Any] | None = None,
    submissions: tuple[dict[str, Any], ...] = (VALID_PLAN,),
) -> tuple[RoleContext, FakeModel, ListSink]:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="planning")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="planning",
        session_id="s1",
        journal=journal,
        memory=FakeStore() if memory else None,
    )
    model = FakeModel(*submissions)
    ctx = RoleContext(
        spec=planning.PLANNING,
        state={
            "session_id": "s1",
            "alert": FIXTURE["alert"],
            "root_cause": RootCause.model_validate(FIXTURE["root_cause"]),
            **(state or {}),
        },
        handback=handback,
        settings=Settings(
            pipeline=PipelineSettings(),
            memory=MemorySettings(enabled=memory, learning_enabled=False),
        ),
        completion=model,
        toolset=build_toolset("planning", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        write_lock=asyncio.Lock(),
        run_id="r1",
        session_id="s1",
        memory=tools.memory,
    )
    return ctx, model, sink


async def test_first_entry_carries_the_alert_and_the_diagnosis_it_must_not_revisit(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model, _ = context(tmp_path)

    outcome = await planning.PLANNING.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["planning_first_entry"]
    assert outcome.artifact is not None


async def test_a_replan_keeps_the_cause_and_lists_the_plans_that_failed(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model, _ = context(
        tmp_path,
        memory=True,
        handback=HandBack(
            target=Route.PLANNING,
            reason="kubectl patch was rejected: the field is immutable",
            kind="replan_from_execution",
        ),
        state={"prior_plans": FIXTURE["prior_plans"]},
    )

    await planning.PLANNING.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["planning_replan"]


async def test_an_empty_plan_that_is_not_a_halt_is_handed_back(tmp_path: pathlib.Path) -> None:
    """Execution would skip with nothing applied, and the stage would fail with no signal."""
    ctx, _, _ = context(tmp_path, submissions=(EMPTY_PLAN, VALID_PLAN))

    outcome = await planning.PLANNING.run(ctx)

    rejections = [
        message["content"]
        for message in outcome.transcript
        if message.get("role") == "tool" and "An empty plan" in str(message.get("content"))
    ]
    assert len(rejections) == 1
    assert outcome.artifact is not None


async def test_planning_without_a_diagnosis_stops_before_the_model(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model, _ = context(tmp_path, state={"root_cause": None})

    outcome = await planning.PLANNING.run(ctx)

    assert outcome.error == "Planning skipped: no root cause from RCA"
    assert (outcome.artifact, outcome.transcript, model.requests) == (None, [], [])
