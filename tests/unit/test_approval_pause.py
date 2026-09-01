"""Parking an incident on a human, and what the pause leaves behind.

The approval mechanism is load-bearing for what the system claims about supervised
autonomy, and in the prototype it was implemented for one stage and invoked in no
measured run. These are the tests that make it a mechanism rather than a claim.
"""

import json
import pathlib
from typing import Any

import pytest

from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.memory.ports import ApprovalRecord
from kira.middleware.authority import ApprovalPending
from kira.middleware.human_approval import HumanApproval, approval_granted, take_grant
from kira.orchestration.resume import ApprovalCheckpoint, denial_result, load_checkpoint
from kira.orchestration.runner import Deployment, run_incident
from kira.settings.authority import AuthoritySettings
from kira.settings.groups import (
    MemorySettings,
    PipelineSettings,
    ServerSettings,
    VerifySettings,
)
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}
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
PATCH = {"cmd": "kubectl patch svc user -p '{}'"}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "service/user patched"


class FakeModel:
    """Diagnoses, plans, then tries to write — which is where a human is asked."""

    def __init__(self) -> None:
        self.callers: list[str] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        submissions = {
            "rca": ("submit_root_cause", ROOT_CAUSE),
            "planning": ("submit_plan", PLAN),
            "execution": ("exec_kubectl_cmd_safely", PATCH),
        }
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
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


def deployment(tmp_path: pathlib.Path) -> tuple[Deployment, list[tuple[str, str]]]:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(mode="prompt", timeout_s=1800),
    )
    storage = build_storage(settings)
    armed: list[tuple[str, str]] = []
    return (
        Deployment(
            settings=settings,
            storage=storage,
            completion=FakeModel(),
            profile=PROFILE,
            transport_for=lambda session_id: FakeTransport(),
            authority=HumanApproval(
                store=storage.approvals,
                settings=settings.authority,
                arm_timeout=lambda request_id, expires_at: armed.append((request_id, expires_at)),
            ),
            sink=ListSink(),
        ),
        armed,
    )


async def park(tmp_path: pathlib.Path) -> tuple[Deployment, str, list[tuple[str, str]]]:
    deploy, armed = deployment(tmp_path)
    session = deploy.storage.sessions.create_session(ALERT)
    with pytest.raises(ApprovalPending):
        await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)
    return deploy, session.session_id, armed


async def test_a_write_needing_a_human_parks_the_incident(tmp_path: pathlib.Path) -> None:
    deploy, session_id, armed = await park(tmp_path)

    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None
    assert stored.run_status == "awaiting_human"
    assert stored.current_stage == "execution"
    assert stored.awaiting_request_id is not None
    assert stored.checkpoint is not None

    # Atomic from a reader's point of view: the request exists by the time the session
    # says it is waiting on one, and its expiry is already armed.
    request = deploy.storage.approvals.get_request(stored.awaiting_request_id)
    assert request is not None and request.status == "pending"
    assert armed and armed[0][0] == request.request_id
    deploy.storage.close()


async def test_the_request_records_the_whole_call(tmp_path: pathlib.Path) -> None:
    """The prototype summarised a write as its first six argument tokens, so a patch
    showed the resource and not the patch body. Approving that is not review."""
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.awaiting_request_id
    request = deploy.storage.approvals.get_request(stored.awaiting_request_id)

    assert request is not None
    assert request.payload["arguments"] == PATCH
    assert request.tool_name == "exec_kubectl_cmd_safely"
    assert request.stage == "execution"
    deploy.storage.close()


async def test_the_checkpoint_carries_the_transcript_and_the_budget(
    tmp_path: pathlib.Path,
) -> None:
    """The counters are carried, not reset. The prototype reset them, so an operator's
    approval silently bought a fresh remediation budget and made a resumed run
    incomparable with one that never paused."""
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None

    checkpoint = ApprovalCheckpoint.model_validate(stored.checkpoint)
    assert checkpoint.stage == "execution"
    assert checkpoint.tool_name == "exec_kubectl_cmd_safely"
    assert checkpoint.messages, "the paused role's transcript has to survive the pause"
    assert checkpoint.state["root_cause"] is not None
    assert checkpoint.state["plan"] is not None
    assert checkpoint.remediation_retries == 0
    deploy.storage.close()


async def test_nothing_terminal_is_recorded_while_a_run_is_waiting(
    tmp_path: pathlib.Path,
) -> None:
    """A paused run has not finished. Recording an outcome now would make a poller read
    the incident as decided before anyone had decided anything."""
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)

    assert stored is not None
    assert stored.state == "pending"
    assert stored.execution_result is None
    deploy.storage.close()


async def test_a_decision_is_matched_to_the_session_that_is_waiting(
    tmp_path: pathlib.Path,
) -> None:
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.awaiting_request_id
    request_id = stored.awaiting_request_id
    sessions, approvals = deploy.storage.sessions, deploy.storage.approvals

    assert load_checkpoint(sessions, approvals, request_id) == "not_ready:pending"
    assert load_checkpoint(sessions, approvals, "no-such-request") == "missing_request"

    approvals.decide(request_id, status="approved", decided_by="an operator", reason="ok")
    loaded = load_checkpoint(sessions, approvals, request_id)
    assert not isinstance(loaded, str)
    checkpoint, record = loaded
    assert checkpoint.request_id == request_id and record.status == "approved"
    deploy.storage.close()


@pytest.mark.parametrize(
    ("status", "reason", "expected"),
    [
        ("denied", "that patch is wrong", "denied by HITL gate: that patch is wrong"),
        ("denied", None, "denied by HITL gate: denied"),
        ("timed_out", None, "denied by HITL gate: timed out with no operator decision"),
    ],
)
def test_a_refusal_reads_as_a_refusal_to_the_role(
    status: str, reason: str | None, expected: str
) -> None:
    """The role sees a tool result, not an exception: a denied write is something to
    report or work around, not a crash."""
    record = ApprovalRecord(
        request_id="r1",
        session_id="s1",
        kind="tool_call",
        stage="execution",
        status=status,
        summary="patch svc/user",
        payload={},
        reason=reason,
    )
    assert denial_result(record) == expected


def test_a_grant_covers_exactly_the_call_it_was_given_for() -> None:
    """A grant that covered a different call would let a resume dispatch something no
    operator ever saw, with an approval recorded beside it."""
    with approval_granted("r1", tool="exec_kubectl_cmd_safely", arguments=PATCH):
        grant = take_grant()
        assert grant is not None
        assert grant.matches("exec_kubectl_cmd_safely", PATCH)
        assert not grant.matches("exec_kubectl_cmd_safely", {"command": "kubectl delete ns shop"})
        assert not grant.matches("ssh_exec", PATCH)
        # Taken rather than read: one grant is one dispatch.
        assert take_grant() is None
