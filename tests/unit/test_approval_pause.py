"""Parking an incident on a human, and what the pause leaves behind.

The approval mechanism is load-bearing for what the system claims about supervised
autonomy, and in the prototype it was implemented for one stage and invoked in no
measured run. These are the tests that make it a mechanism rather than a claim.
"""

import json
import pathlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kira.audit.journal import ListSink
from kira.contracts.lifecycle import RunStatus
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.memory.ports import ApprovalRecord
from kira.middleware.authority import ApprovalPending
from kira.middleware.human_approval import approval_granted, authorities, take_grant
from kira.orchestration import jobs
from kira.orchestration.resume import ApprovalCheckpoint, denial_result, load_checkpoint
from kira.orchestration.runner import Deployment, resume_after_decision, run_incident
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
READ = {"cmd": "kubectl get svc user -o json"}


COMMANDS: list[list[str]] = []
"""Every argv the fake cluster was actually asked to run, so a test can assert that a
refused write did not reach it."""


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        COMMANDS.append(list(argv))
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
            authorities=authorities(
                settings.authority,
                storage.approvals,
                lambda request_id, expires_at: armed.append((request_id, expires_at)),
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


# --- the continuation -----------------------------------------------------------------


class ResumingModel(FakeModel):
    """Reads and writes in one turn, then submits whatever the answer turned out to be.

    The write is issued beside a read because a turn with one call is the one shape that
    cannot show what a pause leaves behind: the sibling has already run, and the resumed
    request has to answer its call as well as the one the operator decided.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append(request)
        if caller != "execution":
            return await super().complete(request, caller=caller)
        self.callers.append(caller)
        if self.callers.count("execution") == 1:
            return self._turn(
                ("read", "exec_kubectl_cmd_safely", READ),
                ("write", "exec_kubectl_cmd_safely", PATCH),
            )
        arguments: dict[str, Any] = {
            "succeeded": True,
            "steps": [
                {
                    "step_index": 0,
                    "succeeded": True,
                    "observation": "targetPort was 8081; patched; endpoints non-empty",
                }
            ],
            "final_state_summary": "svc/user has endpoints",
            "confidence": 0.8,
        }
        return self._turn(("submit", "submit_execution_result", arguments))

    @staticmethod
    def _turn(*calls: tuple[str, str, dict[str, Any]]) -> ModelResponse:
        return ModelResponse(
            text=None,
            tool_calls=tuple(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=arguments,
                    raw_arguments=json.dumps(arguments),
                )
                for call_id, name, arguments in calls
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


async def parked_with(tmp_path: pathlib.Path) -> tuple[Deployment, str, str]:
    deploy, _ = deployment(tmp_path)
    object.__setattr__(deploy, "completion", ResumingModel())
    session = deploy.storage.sessions.create_session(ALERT)
    with pytest.raises(ApprovalPending):
        await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)
    stored = deploy.storage.sessions.get_session(session.session_id)
    assert stored is not None and stored.awaiting_request_id
    return deploy, session.session_id, stored.awaiting_request_id


async def test_an_approval_resumes_the_same_incident(tmp_path: pathlib.Path) -> None:
    """The whole point of a durable pause: the decision continues the run it belongs to
    rather than starting another one.

    And it continues on a transcript a server will accept. The paused turn carried a
    read beside the write; an OpenAI-compatible server rejects a request where any
    tool_call_id lacks a reply, so a continuation that answers only the decided call
    dies at the resume rather than at anything the operator did.
    """
    deploy, session_id, request_id = await parked_with(tmp_path)
    deploy.storage.approvals.decide(
        request_id, status="approved", decided_by="an operator", reason="looks right"
    )
    sent = len(deploy.completion.requests)

    status = await resume_after_decision(request_id, deploy)

    resumed = deploy.completion.requests[sent]
    issued = [
        call["id"]
        for message in resumed.messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    answered = {m.get("tool_call_id") for m in resumed.messages if m.get("role") == "tool"}
    assert issued and not [call_id for call_id in issued if call_id not in answered]

    assert status == "success"
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None
    assert stored.run_status == "completed"
    assert stored.execution_result is not None
    assert stored.awaiting_request_id is None and stored.checkpoint is None
    deploy.storage.close()


async def test_a_denial_comes_back_as_a_tool_result_and_the_run_finishes(
    tmp_path: pathlib.Path,
) -> None:
    """A refused write is something the role reports, not a crash. The incident still
    reaches an outcome."""
    deploy, session_id, request_id = await parked_with(tmp_path)
    deploy.storage.approvals.decide(
        request_id, status="denied", decided_by="an operator", reason="wrong resource"
    )

    status = await resume_after_decision(request_id, deploy)

    assert status in ("success", "failure")
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.run_status == "completed"
    deploy.storage.close()


async def test_a_second_resume_does_not_re_dispatch_the_write(
    tmp_path: pathlib.Path,
) -> None:
    """A duplicate job is a real possibility — a retried timeout, two pollers — and the
    write it would replay has already been applied. Finishing the run clears the
    checkpoint, and its absence is what makes the second attempt a no-op."""
    deploy, session_id, request_id = await parked_with(tmp_path)
    deploy.storage.approvals.decide(request_id, status="approved", decided_by="an operator")
    await resume_after_decision(request_id, deploy)
    runs = len(deploy.storage.runs.runs_for(session_id))

    assert await resume_after_decision(request_id, deploy) == "missing_checkpoint"
    assert len(deploy.storage.runs.runs_for(session_id)) == runs
    deploy.storage.close()


async def test_an_undecided_request_does_not_resume(tmp_path: pathlib.Path) -> None:
    deploy, _, request_id = await parked_with(tmp_path)

    assert await resume_after_decision(request_id, deploy) == "not_ready:pending"
    deploy.storage.close()


async def test_the_approved_write_is_dispatched_through_the_gate(
    tmp_path: pathlib.Path,
) -> None:
    """The one call an operator personally approved was the one that skipped the gate.

    The continuation called the bound tool directly, so the grant had nothing to
    present itself to and the gate recorded no decision for the write that actually
    ran. A reader of the audit log saw a request approved and never saw it executed.
    """
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.awaiting_request_id
    request_id = stored.awaiting_request_id
    deploy.storage.approvals.decide(
        request_id, status="approved", decided_by="an operator", reason="ok"
    )

    before = len(deploy.sink.events)
    await resume_after_decision(request_id, deploy)

    replayed = [
        e.payload
        for e in deploy.sink.events[before:]
        if e.payload.kind == "policy_decision" and e.payload.tool == "exec_kubectl_cmd_safely"
    ]
    assert any(
        d.decision == "approved"
        and d.reason == "replaying an approved request"
        and d.request_id == request_id
        for d in replayed
    )
    deploy.storage.close()


async def test_a_checkpoint_edited_after_approval_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """The gate's tamper check was unreachable in production, which made it decoration.

    An approval is for a specific call. If the arguments can change between the
    decision and the dispatch, the operator approved a description of something else.
    """
    deploy, session_id, _ = await park(tmp_path)
    sessions = deploy.storage.sessions
    stored = sessions.get_session(session_id)
    assert stored is not None and stored.awaiting_request_id and stored.checkpoint
    request_id = stored.awaiting_request_id
    deploy.storage.approvals.decide(
        request_id, status="approved", decided_by="an operator", reason="ok"
    )

    COMMANDS.clear()
    tampered = dict(stored.checkpoint)
    tampered["tool_arguments"] = {"cmd": "kubectl delete svc user"}
    sessions.set_run_status(
        session_id,
        RunStatus.AWAITING_HUMAN,
        current_stage=stored.current_stage,
        checkpoint=tampered,
        awaiting_request_id=request_id,
    )

    await resume_after_decision(request_id, deploy)

    denials = [
        e.payload
        for e in deploy.sink.events
        if e.payload.kind == "policy_decision" and e.payload.decision == "denied"
    ]
    assert any("does not match the recorded request" in d.reason for d in denials)
    assert COMMANDS == []
    deploy.storage.close()


class SingleModeModel:
    """Diagnoses, then tries to write during the mitigation phase.

    It decides from the transcript it is sent rather than from how many times it has
    been called, so a continuation and a restart are distinguishable: a run re-entered
    at turn zero has not seen the write's result and is answered with the write again.
    """

    def __init__(self) -> None:
        self.callers: list[str] = []
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        self.requests.append(request)
        applied = any(
            m.get("role") == "tool" and m.get("name") == "exec_kubectl_cmd_safely"
            for m in request.messages
        )
        if caller == "single.diagnosis":
            name: str = "submit_diagnosis"
            arguments: dict[str, Any] = {"solution": "svc/user targets a port nothing listens on"}
        elif not applied:
            name, arguments = "exec_kubectl_cmd_safely", PATCH
        else:
            name, arguments = "submit_mitigation", {"succeeded": True, "summary": "patched"}
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


async def test_a_single_agent_run_parks_and_resumes(tmp_path: pathlib.Path) -> None:
    """A phase label is not a role, and every map on the resume path is keyed by role.

    The single-agent configuration labels its phases `single.diagnosis` and
    `single.mitigation`, because that is what the write policy names and what the gate
    keys on. Resuming looked those labels up in the toolset registry and the node map,
    found nothing, and raised — leaving the session in `resuming` with the checkpoint
    intact and the approved write never dispatched. A retried job re-entered and raised
    again, so the session never left that state.
    """
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(mode="single"),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        authority=AuthoritySettings(mode="prompt", timeout_s=1800),
    )
    storage = build_storage(settings)
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=SingleModeModel(),
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        authorities=authorities(
            settings.authority,
            storage.approvals,
            lambda request_id, expires_at: None,
        ),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)
    with pytest.raises(ApprovalPending):
        await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    parked = storage.sessions.get_session(session.session_id)
    assert parked is not None and parked.awaiting_request_id and parked.checkpoint
    assert parked.current_stage == "single.mitigation"
    request_id = parked.awaiting_request_id
    checkpointed = list(parked.checkpoint["messages"])
    storage.approvals.decide(request_id, status="approved", decided_by="an operator", reason="ok")

    COMMANDS.clear()
    sent = len(deploy.completion.requests)
    status = await resume_after_decision(request_id, deploy)

    assert status != "resuming"
    # The phase that paused, and only that phase. Re-entering at the diagnosis task
    # re-asks a question already answered and throws away the result of the write the
    # operator has just approved.
    assert deploy.completion.callers[sent:] == ["single.mitigation"]
    resumed = list(deploy.completion.requests[sent].messages)
    assert resumed[: len(checkpointed)] == checkpointed
    assert resumed[len(checkpointed) :] == [
        {
            "role": "tool",
            "tool_call_id": parked.checkpoint["tool_call_id"],
            "name": "exec_kubectl_cmd_safely",
            "content": "service/user patched",
        }
    ]
    # The replay, and no second attempt at the mutation the cluster has already taken.
    assert COMMANDS == [["patch", "svc", "user", "-p", "{}"]]
    finished = storage.sessions.get_session(session.session_id)
    assert finished is not None and finished.run_status != "resuming"
    storage.close()


EXECUTION_RESULT: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched targetPort to 8080"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}


class FinishingModel(FakeModel):
    """Writes once, then submits — so the continuation reaches an outcome."""

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        if caller == "execution" and self.callers.count("execution") >= 1:
            self.callers.append(caller)
            arguments: dict[str, Any] = EXECUTION_RESULT
            return ModelResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        id=f"c{len(self.callers)}",
                        name="submit_execution_result",
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
        return await super().complete(request, caller=caller)


async def test_a_resume_does_not_close_a_run_its_driver_owns(tmp_path: pathlib.Path) -> None:
    """An evaluation driver submits, collects the grade and releases the environment.

    It parks the run with `finalize_run_status=False` for exactly that reason. The
    continuation ignored the flag and finalised the status itself, so the driver's
    poller read the episode as finished before its answer had been submitted — and the
    driver, which never learned the run resumed, released an environment it still owned.
    """
    deploy, _ = deployment(tmp_path)
    deploy = replace(deploy, completion=FinishingModel())
    session = deploy.storage.sessions.create_session(ALERT)
    with pytest.raises(ApprovalPending):
        await run_incident(
            ALERT,
            session.session_id,
            deploy,
            fire_post_hooks=False,
            finalize_run_status=False,
        )
    parked = deploy.storage.sessions.get_session(session.session_id)
    assert parked is not None and parked.awaiting_request_id
    request_id = parked.awaiting_request_id
    deploy.storage.approvals.decide(
        request_id, status="approved", decided_by="an operator", reason="ok"
    )

    stored = deploy.storage.sessions.get_session(session.session_id)
    assert stored is not None and stored.checkpoint is not None
    assert stored.checkpoint["finalize_run_status"] is False
    assert stored.checkpoint["fire_post_hooks"] is False

    status = await resume_after_decision(request_id, deploy)

    # It ran to an outcome; what it must not do is announce one.
    assert status == "success"
    finished = deploy.storage.sessions.get_session(session.session_id)
    assert finished is not None
    assert finished.run_status not in ("completed", "failed")
    deploy.storage.close()


async def test_a_decision_recorded_in_the_store_is_noticed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three ways to decide were offered and only one of them worked.

    The API route schedules the continuation itself. The CLI's default path writes to the
    store directly — deliberately, because that is what works from inside a pod with no
    port open — and nothing then woke the incident: not the API, and not the expiry sweep,
    which passes over a request that has already been decided. The operator saw the
    approval recorded and the run stayed parked. The setting describing how often a
    decision made elsewhere is noticed had been declared from the start, and read nowhere.
    """
    deploy, session_id, _ = await park(tmp_path)
    stored = deploy.storage.sessions.get_session(session_id)
    assert stored is not None and stored.awaiting_request_id
    request_id = stored.awaiting_request_id

    scheduled: list[str] = []
    monkeypatch.setattr(jobs, "schedule_approval_resume", scheduled.append)
    jobs.bind_deployment(deploy)
    try:
        assert await jobs.sweep_decided_approvals() == "nothing waiting"

        # The CLI's local path: straight into the store, no API, no scheduler.
        deploy.storage.approvals.decide(
            request_id, status="approved", decided_by="an operator", reason="ok"
        )

        assert await jobs.sweep_decided_approvals() == "resumed 1"
        assert scheduled == [request_id]
    finally:
        jobs.unbind_deployment()
    deploy.storage.close()


async def test_a_request_whose_timeout_was_never_armed_still_expires(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arming the expiry job can fail, and the pause deliberately survives it: losing the
    write would be worse than a request with no expiry. Nothing then ever looked at that
    request again — not the expiry job, which was never scheduled, and not the sweep,
    which read only requests already decided. The session waited for a person forever,
    which is the outcome the timeout's own default exists to prevent.

    The deadline is on the record, so the sweep reads it.
    """
    deploy, _, _ = await park(tmp_path)
    approvals, sessions = deploy.storage.approvals, deploy.storage.sessions

    session = sessions.create_session({"alertname": "Overdue", "namespace": "shop"})
    approvals.create_request(
        ApprovalRecord(
            request_id="never-armed",
            session_id=session.session_id,
            kind="tool_call",
            stage="execution",
            status="pending",
            summary="kubectl delete pod checkout-0 -n shop",
            payload={},
            expires_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        )
    )
    sessions.set_run_status(
        session.session_id, RunStatus.AWAITING_HUMAN, awaiting_request_id="never-armed"
    )

    scheduled: list[str] = []
    monkeypatch.setattr(jobs, "schedule_approval_resume", scheduled.append)
    jobs.bind_deployment(deploy)
    try:
        assert await jobs.sweep_decided_approvals() == "resumed 1"
        assert scheduled == ["never-armed"]
        assert approvals.get_request("never-armed").status == "timed_out"
    finally:
        jobs.unbind_deployment()
    deploy.storage.close()
