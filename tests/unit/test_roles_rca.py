"""What the diagnosing role puts in front of the model, and what it records.

The rendered messages are pinned to `tests/fixtures/prototype_role_messages.json`, which
was produced by importing the prototype's own builders and its strict renderer
(`agent.agents.rca._reassessment_block`, `agent.prompt_loader.render_prompt`) with the
playbook, ACE and assignment features disabled — the configuration every published
measurement ran. The expectation is therefore the prototype's output, not a transcript
of what this implementation happens to produce.
"""

import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.events import ToolCallEvent
from kira.contracts.routing import HandBack, Route
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.middleware.base import MiddlewareStack
from kira.roles import rca
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

CAPABILITIES = frozenset(
    {"kubernetes", "kubernetes.write", "kubernetes.rollback", "metrics", "logs", "traces", "host"}
)


class FakeTransport:
    name = "fake"
    capabilities = CAPABILITIES
    timeout_seconds = 30


class FakeStore:
    """Stands in for the experience store, so the retrieval tools survive resolution."""


class FakeModel:
    """Submits the fixture's diagnosis on its first turn and records what it was sent."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        arguments = dict(FIXTURE["root_cause"])
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="submit_root_cause",
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
    **pipeline: Any,
) -> tuple[RoleContext, FakeModel, ListSink]:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="rca")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=journal,
        memory=FakeStore() if memory else None,
    )
    model = FakeModel()
    ctx = RoleContext(
        spec=rca.RCA,
        state={"session_id": "s1", "alert": FIXTURE["alert"], **(state or {})},
        handback=handback,
        settings=Settings(
            pipeline=PipelineSettings(**pipeline),
            memory=MemorySettings(enabled=memory, learning_enabled=False),
        ),
        completion=model,
        toolset=build_toolset("rca", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        run_id="r1",
        session_id="s1",
        memory=tools.memory,
    )
    return ctx, model, sink


REASSESSMENT = HandBack(
    target=Route.RCA, reason="the 5xx rate is unchanged after the patch", kind="reassessment"
)


async def test_first_entry_carries_the_memory_line_and_the_snapshot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both `{{MEMORY_GUIDANCE}}` paths are wired: this is the user-turn one."""

    async def snapshot(_ctx: object, _namespace: str) -> str:
        return FIXTURE["snapshot"]

    monkeypatch.setattr(rca, "namespace_snapshot", snapshot)
    ctx, model, _ = context(tmp_path, memory=True)

    outcome = await rca.RCA.run(ctx)

    opening = model.requests[0].messages[0]["content"]
    assert opening == FIXTURE["rca_first_entry_memory_and_snapshot"]
    assert outcome.artifact is not None


async def test_a_failed_snapshot_is_recorded_rather_than_silently_empty(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prototype swallowed the failure, and the block was empty in ~93% of runs."""

    async def boom(_ctx: object, _namespace: str) -> str:
        raise RuntimeError("apiserver read timed out")

    monkeypatch.setattr(rca, "namespace_snapshot", boom)
    ctx, model, sink = context(tmp_path)

    await rca.RCA.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["rca_first_entry"]
    recorded = [
        event.payload
        for event in sink.events
        if isinstance(event.payload, ToolCallEvent) and event.payload.tool == "namespace_snapshot"
    ]
    assert [event.succeeded for event in recorded] == [False]


async def test_a_reassessment_entry_asks_for_the_three_way_discrimination(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model, _ = context(
        tmp_path,
        auto_snapshot=False,
        handback=REASSESSMENT,
        state={"prior_attempts": FIXTURE["prior_attempts"]},
    )

    await rca.RCA.run(ctx)

    assert model.requests[0].messages[0]["content"] == FIXTURE["rca_reassessment"]


@pytest.mark.parametrize(
    ("prior", "reason"),
    [
        ([], "no_prior_transcript"),
        ([{"role": "user", "content": "x" * 50}], "transcript_too_large"),
    ],
)
async def test_talkback_degrades_to_fresh_with_the_reason_recorded(
    tmp_path: pathlib.Path, prior: list[dict[str, Any]], reason: str
) -> None:
    ctx, model, _ = context(
        tmp_path,
        auto_snapshot=False,
        reassess_mode="talkback",
        talkback_max_chars=10,
        handback=REASSESSMENT,
        state={"prior_attempts": FIXTURE["prior_attempts"], "messages": {"rca": prior}},
    )

    outcome = await rca.RCA.run(ctx)

    assert outcome.metadata == {"reassess_mode": "fresh", "talkback_fallback": reason}
    assert model.requests[0].messages[0]["content"] == FIXTURE["rca_reassessment"]


def test_every_dangling_tool_call_is_answered_before_a_transcript_is_replayed() -> None:
    """An unanswered tool_call_id is a hard 400 from an OpenAI-compatible server."""
    assert rca.close_dangling_tool_calls(FIXTURE["transcript"]) == FIXTURE["closed_transcript"]
