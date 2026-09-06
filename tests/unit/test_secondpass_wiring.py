"""The gated second opinion, composed around the diagnosing role.

The package itself is tested in isolation. What is pinned here is the composition: that
the mechanism replaces the diagnosis node, that both of its runs are ordinary role runs
with their own rows, and that both identifiers enter the incident's ordered path — the
prototype returned one of the two states, so the losing run's identifier never did.
"""

import json
import pathlib
from typing import Any

from kira.audit.journal import ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.orchestration.runner import Deployment, run_incident
from kira.settings.groups import (
    MemorySettings,
    PipelineSettings,
    SecondPassSettings,
    ServerSettings,
    VerifySettings,
)
from kira.settings.profiles import load_profile
from kira.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))
ALERT = {"alertname": "HighErrorRate", "namespace": "shop"}

DIAGNOSIS: dict[str, Any] = {
    "fault_type": "misconfiguration",
    "affected_resource": "shop/Service/user",
    "confidence": 0.8,
    "evidence": ["endpoints empty for svc/user"],
    "summary": "the user Service targets a port nothing listens on",
}
PLAN: dict[str, Any] = {
    "steps": [
        {
            "action": "kubectl patch svc/user",
            "target": "shop/Service/user",
            "justification": "point the selector at the serving port",
        }
    ],
    "expected_outcome": "endpoints populate",
    "confidence": 0.8,
}
APPLIED: dict[str, Any] = {
    "succeeded": True,
    "steps": [],
    "final_state_summary": "the patch applied",
    "confidence": 0.8,
}
HANDS_BACK: dict[str, Any] = {
    "recovered": False,
    "verdict": "not_recovered",
    "checks": ["curl http://user.shop/health returned 503"],
    "summary": "still failing",
    "confidence": 0.7,
    "needs_reassessment": True,
    "reassessment_reason": "the named Service was not the origin",
}


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "no resources found"


class FakeModel:
    """Answers each caller in the mechanism, and halts planning so the run ends there."""

    def __init__(self) -> None:
        self.callers: list[str] = []
        self.sampling: list[Any] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.callers.append(caller)
        self.sampling.append(request.sampling)
        if caller == "secondpass.gate":
            return _text('{"needs_second_opinion": true, "weakness": "no probe of the path"}')
        if caller == "secondpass.selector":
            return _text('{"choice": "A", "reason": "cites the endpoint directly"}')
        if caller == "planning":
            plan = {
                "steps": [],
                "expected_outcome": "nothing",
                "confidence": 0.1,
                "halt": True,
                "halt_reason": "stopping the test here",
            }
            return _call("submit_plan", plan, len(self.callers))
        return _call("submit_root_cause", DIAGNOSIS, len(self.callers))


class HandingBackModel(FakeModel):
    """Verification never accepts the repair, so every cycle takes the back-edge.

    Everything else answers normally, which makes the diagnosis node the only place the
    hand-back accounting can be lost.
    """

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        if caller in {"planning", "execution", "verify"}:
            self.callers.append(caller)
            self.sampling.append(request.sampling)
            payload = {"planning": PLAN, "execution": APPLIED, "verify": HANDS_BACK}[caller]
            submit = {
                "planning": "submit_plan",
                "execution": "submit_execution_result",
                "verify": "submit_verify_result",
            }[caller]
            return _call(submit, payload, len(self.callers))
        return await super().complete(request, caller=caller)


def _text(content: str) -> ModelResponse:
    return ModelResponse(
        text=content,
        tool_calls=(),
        finish_reason="stop",
        reasoning=None,
        usage=Usage(),
        duration_ms=1,
        attempts=1,
    )


def _call(name: str, arguments: dict[str, Any], index: int) -> ModelResponse:
    return ModelResponse(
        text=None,
        tool_calls=(
            ToolCall(
                id=f"c{index}",
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


async def test_the_mechanism_replaces_the_diagnosis_and_keeps_both_runs(
    tmp_path: pathlib.Path,
) -> None:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(),
        memory=MemorySettings(
            enabled=True,
            learning_enabled=False,
            role_retrieval=False,
            second_pass=SecondPassSettings(enabled=True, gate="quality", delivery="context"),
        ),
        verify=VerifySettings(enabled=False),
    )
    storage = build_storage(settings)
    model = FakeModel()
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    # One gate, one selector, and two diagnosis runs — the dose an arm gap is divided by.
    assert model.callers.count("secondpass.gate") == 1
    assert model.callers.count("secondpass.selector") == 1
    assert model.callers.count("rca") == 2
    assert len([r for r in storage.runs.runs_for(session.session_id) if r.agent_name == "rca"]) == 2
    assert len([rid for rid in final["run_ids"] if ":rca:" in rid]) == 2
    assert final["root_cause"] is not None

    # And the two are told apart. Both rows say `rca`, which is right — they are the same
    # role — so without the label nothing downstream could tell the retrieval-free first
    # opinion from the precedent-informed second, which is the comparison this arm exists
    # to make. The mechanism passed the label from the start; the runner dropped it.
    labels = [
        r.metadata.get("run_label")
        for r in storage.runs.runs_for(session.session_id)
        if r.agent_name == "rca"
    ]
    assert sorted(labels) == ["rca.first", "rca.second"]
    storage.close()


async def test_the_helpers_run_on_the_auxiliary_sampling_set(
    tmp_path: pathlib.Path,
) -> None:
    """A gate decided at the profile's open-ended sampling gives a different answer on
    repeated runs of the same cell, which is a research-validity failure rather than a
    performance one."""
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(
            enabled=True,
            learning_enabled=False,
            role_retrieval=False,
            second_pass=SecondPassSettings(enabled=True, gate="quality", delivery="context"),
        ),
        verify=VerifySettings(enabled=False),
    )
    storage = build_storage(settings)
    model = FakeModel()
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)

    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    for caller, sampling in zip(model.callers, model.sampling, strict=True):
        if caller.startswith("secondpass."):
            assert sampling is not None and sampling.temperature == 0.0
            assert sampling.presence_penalty is None
    storage.close()


async def test_a_diagnosis_only_task_still_gets_its_second_opinion(
    tmp_path: pathlib.Path,
) -> None:
    """A mode that built its own node would run the diagnosis without whatever the arm
    wraps around it, so a diagnosis-only task under an arm that second-guesses its
    diagnosis would silently be a different arm."""
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(
            enabled=True,
            learning_enabled=False,
            role_retrieval=False,
            second_pass=SecondPassSettings(enabled=True, gate="quality", delivery="context"),
        ),
        verify=VerifySettings(enabled=False),
    )
    storage = build_storage(settings)
    model = FakeModel()
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)

    final = await run_incident(
        ALERT, session.session_id, deploy, target_stage="diagnose", fire_post_hooks=False
    )

    assert model.callers.count("secondpass.gate") == 1
    assert model.callers.count("rca") == 2
    assert final["plan"] is None
    storage.close()


# --- composed with verification ------------------------------------------------------


def verifying_settings(tmp_path: pathlib.Path) -> Settings:
    """The published MA+S+V shape: both mechanisms on, budget of two."""
    return Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(remediation_retries=2),
        memory=MemorySettings(
            enabled=True,
            learning_enabled=False,
            role_retrieval=False,
            second_pass=SecondPassSettings(enabled=True, gate="quality", delivery="context"),
        ),
        verify=VerifySettings(enabled=True),
    )


async def test_a_hand_back_through_the_composed_node_charges_the_shared_budget(
    tmp_path: pathlib.Path,
) -> None:
    """The mechanism wraps the diagnosis node, so it inherits that node's obligations.

    Reducing its two runs to the diagnosis they produced dropped the rest of what the
    wrapper decided: the verdict stayed live in the state and was re-derived at every
    later entry, so nothing charged the budget and planning, execution and verification
    cycled until the step limit killed the incident — each cycle another mutation of the
    cluster, and an exception in place of an outcome.
    """
    settings = verifying_settings(tmp_path)
    storage = build_storage(settings)
    model = HandingBackModel()
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    # The verdict is consumed as it is acted on, so each hand-back buys exactly one more
    # cycle: the gated pass's two diagnosis runs, then one re-diagnosis per verdict.
    runs = storage.runs.runs_for(session.session_id)
    stages = [run.agent_name for run in runs]
    assert stages.count("rca") == 4
    assert stages.count("verify") == 3

    # And the budget runs out, so the incident ends with a result rather than an
    # exception the harness records as a crash.
    assert final["remediation_retries"] == settings.pipeline.remediation_retries
    assert final["reassessment_count"] == 2
    storage.close()


async def test_the_second_opinion_runs_on_the_first_diagnosis_only(
    tmp_path: pathlib.Path,
) -> None:
    """A re-diagnosis re-enters plain diagnosis carrying the reassessment context.

    Repeating the pass would weigh retrieved precedent against the verification this
    incident just performed, which is the stronger signal, and would make precedent
    exposure a function of how the first repair turned out — the confound the mechanism
    is measured against. It would also double the dose in the one cell of the factorial
    that carries the interaction, so the arm's cost accounting would describe a
    configuration nothing ran.
    """
    settings = verifying_settings(tmp_path)
    storage = build_storage(settings)
    model = HandingBackModel()
    deploy = Deployment(
        settings=settings,
        storage=storage,
        completion=model,
        profile=PROFILE,
        transport_for=lambda session_id: FakeTransport(),
        sink=ListSink(),
    )
    session = storage.sessions.create_session(ALERT)

    final = await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)

    assert model.callers.count("secondpass.gate") == 1
    assert model.callers.count("secondpass.selector") == 1

    # Three diagnosis entries: the composed one, then a plain one per hand-back. Only
    # the first is a pass, which is what the run labels say.
    diagnoses = [r for r in storage.runs.runs_for(session.session_id) if r.agent_name == "rca"]
    assert len(diagnoses) == 4
    labelled = [r.metadata["run_label"] for r in diagnoses if "run_label" in r.metadata]
    assert sorted(labelled) == ["rca.first", "rca.second"]

    # And the re-entries were re-diagnoses rather than fresh starts.
    assert len(final["prior_attempts"]) == 2
    storage.close()
