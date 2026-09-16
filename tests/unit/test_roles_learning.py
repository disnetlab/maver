"""What the distilling role is given, and what it refuses to run without."""

import asyncio
import json
import pathlib
from dataclasses import replace
from typing import Any

import pytest

from maver.audit.journal import Journal, ListSink
from maver.contracts.artifacts import LearningPacket
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.middleware.base import MiddlewareStack
from maver.roles import learning
from maver.roles.base import RoleContext
from maver.settings.groups import MemorySettings, PipelineSettings, PromptSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools.context import ToolContext
from maver.tools.toolsets import build_toolset
from maver.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
TRIGGERS = pathlib.Path("configs/learning")

PACKET = LearningPacket(
    session_id="s1",
    lifecycle_state="tentative_success",
    alert={"alertname": "HighErrorRate", "namespace": "shop"},
    root_cause={"fault_type": "misconfiguration", "summary": "wrong targetPort"},
)

RESULT: dict[str, Any] = {
    "observations": ["svc/user had no endpoints"],
    "attempted_hypothesis": "the service targeted a port nothing listens on",
    "attempted_action": "patched targetPort to 8080",
    "observed_outcome": "endpoints appeared and the 5xx rate fell",
    "case_negative_constraints": ["restarting the deployment does not help"],
    "corrected_cause": "",
    "discriminator": "check svc endpoints before restarting anything",
    "summary": "a Service targeting a port no container listens on",
    "fault_category": "misconfiguration",
    "verification_verdict": "recovered",
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30


class FakeModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="submit_learning_result",
                    arguments=RESULT,
                    raw_arguments=json.dumps(RESULT),
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
    packet: LearningPacket | None = PACKET,
    verify_live: bool = False,
) -> tuple[RoleContext, FakeModel]:
    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="learning")
    tools = ToolContext(
        transport=FakeTransport(),
        workspace=SessionWorkspace(tmp_path),
        role="learning",
        session_id="s1",
        journal=journal,
    )
    model = FakeModel()
    ctx = RoleContext(
        spec=learning.LEARNING,
        state={"session_id": "s1"},
        handback=None,
        settings=Settings(
            pipeline=PipelineSettings(),
            memory=MemorySettings(
                enabled=True,
                learning_enabled=True,
                learning_verify=verify_live,
                learning_config_dir=TRIGGERS,
            ),
        ),
        completion=model,
        toolset=build_toolset("learning", tools),
        tool_context=tools,
        journal=journal,
        profile=PROFILE,
        middleware=MiddlewareStack(),
        write_lock=asyncio.Lock(),
        run_id="r1",
        session_id="s1",
        packet=packet,
    )
    return ctx, model


async def test_a_deployments_own_prompt_reaches_the_distilling_pass(
    tmp_path: pathlib.Path,
) -> None:
    """Five roles passed the override directory into their composition and this one did
    not, while the snapshot digested `learning` and startup validated it. The one role
    whose output persists into the retrieval corpus was the one that could not be
    steered, and the run record said otherwise."""
    overrides = tmp_path / "prompts"
    (overrides / "roles").mkdir(parents=True)
    (overrides / "roles" / "learning.txt").write_text(
        "DEPLOYMENT OVERRIDE FOR LEARNING.\n", encoding="utf-8"
    )
    ctx, model = context(tmp_path)
    ctx = replace(
        ctx,
        settings=ctx.settings.model_copy(
            update={"prompts": PromptSettings(override_dir=overrides)}
        ),
    )

    await learning.LEARNING.run(ctx)

    assert "DEPLOYMENT OVERRIDE FOR LEARNING." in model.requests[0].system


async def test_the_packet_is_what_the_pass_reads(tmp_path: pathlib.Path) -> None:
    """Learning runs minutes after the incident closed, on a fresh task, so there is no
    live state to read: what it gets is what was written down."""
    ctx, model = context(tmp_path)

    outcome = await learning.LEARNING.run(ctx)

    opening = model.requests[0].messages[0]["content"]
    assert "Bounded Learning Packet:" in opening
    assert '"session_id": "s1"' in opening
    assert outcome.artifact is not None


async def test_an_external_evaluation_replaces_the_live_self_check(
    tmp_path: pathlib.Path,
) -> None:
    """A source of truth supersedes a self-check, and by the time one exists the
    environment has usually been torn down, so the guidance would only spend turns."""
    graded = PACKET.model_copy(update={"external_feedback": {"diagnosis": {"success": False}}})
    ctx, model = context(tmp_path, packet=graded, verify_live=True)

    await learning.LEARNING.run(ctx)

    opening = model.requests[0].messages[0]["content"]
    assert "source of truth" in opening
    assert "Live Recovery Verification" not in model.requests[0].system


async def test_the_live_check_is_appended_when_nothing_graded_the_session(
    tmp_path: pathlib.Path,
) -> None:
    ctx, model = context(tmp_path, verify_live=True)

    await learning.LEARNING.run(ctx)

    assert "Live Recovery Verification" in model.requests[0].system
    assert "source of truth" not in model.requests[0].messages[0]["content"]


async def test_the_turn_budget_comes_from_the_trigger(tmp_path: pathlib.Path) -> None:
    """Not from the pipeline's: a distillation pass and an incident stage are bounded
    for different reasons."""
    ctx, _ = context(tmp_path)
    assert ctx.settings.pipeline.max_turns != 30

    await learning.LEARNING.run(ctx)

    assert learning.load_trigger("post_incident", TRIGGERS).max_turns == 30


async def test_a_pass_with_no_packet_does_not_run(tmp_path: pathlib.Path) -> None:
    ctx, model = context(tmp_path, packet=None)

    outcome = await learning.LEARNING.run(ctx)

    assert outcome.artifact is None
    assert model.requests == []


def test_an_unconfigured_trigger_is_refused_rather_than_defaulted(
    tmp_path: pathlib.Path,
) -> None:
    """A pass that ran under a trigger nobody configured would write records nobody
    asked for."""
    (tmp_path / "triggers.yaml").write_text(
        "schema_version: 1\ntriggers:\n  post_incident:\n    enabled: false\n"
    )

    with pytest.raises(ValueError, match="disabled"):
        learning.load_trigger("post_incident", tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        learning.load_trigger("domain", tmp_path)
