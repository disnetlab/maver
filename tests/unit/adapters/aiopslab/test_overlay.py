"""What each role is told when the incident is an AIOpsLab task.

The five branches the prototype carried inside the roles are one table here, so the test
is a table too: for each role and each graded task, what the environment narrows and what
it leaves alone. The rest of the module is the detection task, which is the only place
the adapter does something stronger than adding text — it withholds tools, shortens the
budget, and refuses a submission the live cluster contradicts.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from maver.adapters.aiopslab.config import AiopsLabConfig
from maver.adapters.aiopslab.overlay import role_overlay
from maver.audit.context import bind_journal, current_journal
from maver.audit.journal import Journal, ListSink
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.roles.loop import run_until_submit
from maver.roles.policy import LoopLimits
from maver.settings.groups import RuntimeSettings

NAMESPACE = "test-hotel-reservation"
BOUNDARY = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

CONFIG = AiopsLabConfig()

REJECTION = (
    "submit_root_cause is inconsistent with the live episode evidence: {decisive}. An "
    "abnormal termination AT/AFTER the incident boundary is an anomaly even if "
    "Kubernetes self-healed and later requests are healthy. Resubmit Yes now using this "
    "resource/evidence; do not run more investigation tools."
)
"""Carried word for word from `agent/agents/rca.py:118-124`. Pinned here rather than
read from the asset, because an asset compared against itself pins nothing."""

SCOPE = (
    "## AIOpsLab incident scope\n\n"
    f"The active incident namespace is `{NAMESPACE}`. Inspect and mutate "
    "only that namespace. Do not use another namespace, `-A`, or cluster-wide "
    "discovery. If the plan names a different namespace, treat it as a plan defect "
    "and hand back for replanning; do not try the command outside the active scope.\n\n"
)
"""Carried word for word from `agent/agents/execution.py:76-84`, up to the approval
sentence, which the two tests below pin separately."""

AUTOMATIC = (
    "This run is non-interactive: an allowed mutation is automatically approved and "
    "executes immediately. Call the narrow mutation directly; never ask for or wait for "
    "approval."
)
DELEGATED = (
    "Call an allowed mutation directly. Authorization is handled by the runtime; do not "
    "ask for or manually wait on approval."
)

NO_SUBMISSION: dict[str, Any] = {
    "fault_type": "none",
    "affected_resource": "none",
    "confidence": 0.6,
    "evidence": ["every pod is Ready"],
    "summary": "AIOpsLab detection: has_anomaly=No.",
    "halt": True,
}
YES_SUBMISSION: dict[str, Any] = {
    "fault_type": "misconfiguration",
    "affected_resource": f"{NAMESPACE}/Deployment/geo",
    "confidence": 0.8,
    "evidence": ["geo terminated after the boundary"],
    "summary": "AIOpsLab detection: has_anomaly=Yes.",
}


def alert(task: str, *, namespace: str = NAMESPACE, benchmark: str = "aiopslab") -> dict[str, Any]:
    return {
        "alertname": "ServiceHealthDegraded",
        "benchmark": benchmark,
        "benchmark_task": task,
        "namespace": namespace,
        "observation_window_start": BOUNDARY.isoformat().replace("+00:00", "Z"),
    }


def state(task: str, **kw: Any) -> Any:
    return {"session_id": "s1", "alert": alert(task, **kw)}


def pods(finished: datetime) -> str:
    """One pod that is Ready now and terminated abnormally at a given moment."""
    return json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "geo-0", "labels": {}},
                    "spec": {"nodeName": "worker-1", "containers": []},
                    "status": {
                        "phase": "Running",
                        "startTime": "2026-01-01T11:00:00Z",
                        "containerStatuses": [
                            {
                                "ready": True,
                                "restartCount": 1,
                                "state": {"running": {}},
                                "lastState": {
                                    "terminated": {
                                        "reason": "Error",
                                        "exitCode": 1,
                                        "finishedAt": finished.isoformat().replace("+00:00", "Z"),
                                    }
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )


class FakeBridge:
    """A transport that answers the pod read and records that it was asked at all."""

    name = "aiopslab"
    capabilities = frozenset({"kubernetes", "metrics", "traces"})
    timeout_seconds = 30

    def __init__(self, terminated: datetime | None) -> None:
        self._terminated = terminated
        self.reads: list[str] = []

    async def kubectl(self, argv: list[str]) -> str:
        self.reads.append(" ".join(argv))
        if "pods" in " ".join(argv) and self._terminated is not None:
            return pods(self._terminated)
        return "error: this read is not served in the test"


def overlay(role: str, task: str, *, transport: Any = None, mode: str = "auto", **kw: Any) -> Any:
    return role_overlay(
        role,
        state(task, **kw),
        config=CONFIG,
        transport=transport,
        authority_mode=mode,
    )


@pytest.mark.parametrize(
    ("role", "task"),
    [
        ("rca", "detection"),
        ("rca", "localization"),
        ("rca", "analysis"),
        ("rca", "mitigation"),
        ("planning", "mitigation"),
        ("execution", "mitigation"),
        ("verify", "mitigation"),
    ],
)
def test_every_role_and_task_the_environment_narrows(role: str, task: str) -> None:
    """The five decisions the prototype spread across four role modules."""
    assert overlay(role, task) is not None


@pytest.mark.parametrize(
    ("role", "task"),
    [
        # The three roles that only run for the graded remediation task, on a task
        # where they do not run at all. A suffix leaking onto a detection incident
        # would tell a diagnosing model about a repair it is not being asked for.
        ("planning", "detection"),
        ("execution", "localization"),
        ("verify", "analysis"),
        # Roles this environment has nothing to say to, on the task it says most about.
        ("learning", "mitigation"),
        ("single", "mitigation"),
    ],
)
def test_every_role_and_task_the_environment_leaves_alone(role: str, task: str) -> None:
    assert overlay(role, task) is None


def test_the_diagnosis_is_replaced_for_detection_and_qualified_for_the_rest() -> None:
    """Detection asks a yes-or-no question the role was not written for, so its prompt
    is replaced; the other three ask the role's own question and get a suffix."""
    detection = overlay("rca", "detection")
    assert detection.system_replacement is not None
    assert detection.system_replacement.startswith("## Goal")
    assert detection.system_suffix == ""

    for task in ("localization", "analysis", "mitigation"):
        qualified = overlay("rca", task)
        assert qualified.system_replacement is None
        # Rendered under the loader's strict two-way templating, so a renamed marker
        # fails loudly instead of shipping a literal `{{TASK_TYPE}}` to the model.
        assert f"AIOpsLab `{task}` task" in qualified.system_suffix
        assert "{{" not in qualified.system_suffix


@pytest.mark.parity
def test_the_remediating_roles_are_told_what_the_bridge_will_refuse() -> None:
    """Both halves of the remediation carry the transport contract; only the one that
    writes is also told which namespace it is confined to and who authorises it."""
    planning = overlay("planning", "mitigation").system_suffix
    execution = overlay("execution", "mitigation").system_suffix

    assert planning.startswith("## AIOpsLab remediation transport")
    assert "## AIOpsLab incident scope" not in planning
    assert execution == f"{planning}\n\n{SCOPE}{AUTOMATIC}"


@pytest.mark.parity
def test_the_approval_sentence_follows_the_deployments_own_authority() -> None:
    """An unattended run must not be told to wait for a human: it would spend its whole
    budget waiting for nobody."""
    unattended = overlay("execution", "mitigation", mode="auto").system_suffix
    supervised = overlay("execution", "mitigation", mode="prompt").system_suffix

    assert unattended.endswith(f"{SCOPE}{AUTOMATIC}")
    assert supervised.endswith(f"{SCOPE}{DELEGATED}")


@pytest.mark.parity
def test_the_connectivity_contract_is_given_only_for_an_application_we_deployed() -> None:
    """B2. Deployment metadata, keyed only by namespace: a Service to probe, and nothing
    about the incident's causal answer. A namespace no application was configured for has
    no entry point to name, and inventing one would send the probe where nothing serves.
    """
    block = overlay("verify", "mitigation").context_blocks[0]

    assert block.startswith("## Benchmark-supplied application connectivity contract")
    body = json.loads(block.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert body == CONFIG.connectivity_contracts[NAMESPACE]
    # Rendered into text on the way out, so the process-wide mapping every incident
    # reads is never handed to anything that could mutate it.
    assert body is not CONFIG.connectivity_contracts[NAMESPACE]

    unknown = overlay("verify", "mitigation", namespace="some-other-namespace")
    assert unknown.context_blocks == ()
    assert "Generic `kubectl exec` is rejected" in unknown.prompt_blocks["KUBECTL_GUIDANCE"]


@pytest.mark.parity
def test_detection_withholds_exactly_three_tools_and_can_only_shorten_the_budget() -> None:
    """B15. Deep endpoint enumeration and local execution were the main sources of
    wandering calls in the detection transcript; localization and analysis keep them."""
    detection = overlay("rca", "detection")

    assert detection.drop_tools == frozenset({"bash_exec", "code_exec", "get_operations"})
    assert overlay("rca", "localization").drop_tools == frozenset()

    runtime = RuntimeSettings()
    assert LoopLimits.from_settings(runtime, max_turns=15, overlay=detection).max_turns == 10
    # `min`, so a deployment that configured a shorter run is not lengthened to ten.
    assert LoopLimits.from_settings(runtime, max_turns=4, overlay=detection).max_turns == 4


@pytest.mark.parity
async def test_the_guard_refuses_a_no_the_live_episode_contradicts() -> None:
    """B14, first case. An abnormal termination at or after the neutral boundary is an
    anomaly even though Kubernetes self-healed it, so the model is sent back to answer
    again — with the evidence, not with the answer."""
    bridge = FakeBridge(BOUNDARY + timedelta(minutes=3))
    guard = overlay("rca", "detection", transport=bridge).submit_guard

    sink = ListSink()
    journal = Journal(sink, session_id="s1", stage="rca", run_id="agent-rca")
    with bind_journal(journal):
        verdict = await guard(NO_SUBMISSION, 4, [])
    assert current_journal() is None
    (started,) = [e for e in sink.events if e.payload.kind == "tool_call_started"]
    (finished,) = [e for e in sink.events if e.payload.kind == "tool_call"]
    assert started.run_id == finished.run_id == "agent-rca"
    assert started.payload.invocation_id == finished.payload.invocation_id
    assert finished.payload.tool == "namespace_snapshot"
    assert finished.payload.status == "incomplete"
    assert finished.payload.transport == "aiopslab"
    assert finished.payload.output_artifacts
    assert "geo-0" in finished.payload.result["inline"]["content"]

    assert verdict is not None
    assert "[AT/AFTER incident boundary]" in verdict
    decisive = verdict.split(": ", 1)[1].split(". An abnormal", 1)[0]
    assert verdict == REJECTION.format(decisive=decisive)
    assert "geo-0" in decisive


@pytest.mark.parity
async def test_the_guard_accepts_a_no_with_nothing_after_the_boundary() -> None:
    """B14, second case. A termination the snapshot dated before the boundary is
    deployment history, and the guard has nothing to say about it."""
    bridge = FakeBridge(BOUNDARY - timedelta(hours=2))
    guard = overlay("rca", "detection", transport=bridge).submit_guard

    assert await guard(NO_SUBMISSION, 4, []) is None
    assert bridge.reads, "the guard has to look before it can accept"


@pytest.mark.parity
async def test_the_guard_never_looks_at_the_cluster_to_confirm_a_yes() -> None:
    """B14, third case. It is a consistency check on a claim the evidence contradicts,
    not an oracle: an answer it would accept either way is not worth a cluster read."""
    bridge = FakeBridge(BOUNDARY + timedelta(minutes=3))
    guard = overlay("rca", "detection", transport=bridge).submit_guard

    assert await guard(YES_SUBMISSION, 4, []) is None
    assert bridge.reads == []


async def test_the_guard_accepts_when_it_could_not_read_the_cluster_at_all() -> None:
    """A guard that refused because it could not look would reject on its own outage,
    and the detection answer would be decided by a transport failure."""
    guard = overlay("rca", "detection", transport=None).submit_guard

    assert await guard(NO_SUBMISSION, 4, []) is None


async def test_the_refusal_is_recorded_as_a_rejected_submission() -> None:
    """B14, fourth case. It is a harness intervention on the model's answer, so the run
    is only reconstructible if the event log says it happened — through the same path a
    schema rejection takes, rather than a second event kind."""
    bridge = FakeBridge(BOUNDARY + timedelta(minutes=3))
    guard = overlay("rca", "detection", transport=bridge).submit_guard
    answers = [NO_SUBMISSION, YES_SUBMISSION]

    class Answering:
        """Submits No, then Yes: the correction the guard asked the model to make."""

        async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
            arguments = answers.pop(0)
            return ModelResponse(
                text=None,
                tool_calls=(
                    ToolCall(
                        id=f"c{len(answers)}",
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

    sink = ListSink()
    result = await run_until_submit(
        model=Answering(),
        role="rca",
        system="detect",
        tools=[],
        dispatch={},
        submit_tool="submit_root_cause",
        messages=[{"role": "user", "content": "decide"}],
        journal=Journal(sink, session_id="s1", stage="rca"),
        submit_guard=guard,
    )

    submissions = [event.payload for event in sink.events if event.payload.kind == "submission"]
    assert [event.accepted for event in submissions] == [False, True]
    assert submissions[0].validation_error is not None
    assert submissions[0].validation_error.startswith("submit_root_cause is inconsistent")
    # The guard cannot write the answer: the corrected artifact is the model's own.
    assert result.submission == YES_SUBMISSION
