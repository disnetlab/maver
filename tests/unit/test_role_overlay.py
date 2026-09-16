"""What an environment may do to a role, and what it may not.

Every assertion here is about a branch that used to live inside a role. The prototype
had five of them — a replaced detection prompt, a shortened budget, a tool subset, a
consistency guard and a connectivity block — each written as `if benchmark == …` in the
role that carried it, so a second environment meant a sixth branch in the same files.
The overlay is the whole of what replaced them, which is why it is tested against a fake
environment rather than against the one that motivated it: a seam whose only client is
one benchmark is that benchmark's branch under another name.

The load-bearing half is the negative one. An overlay of None has to leave a role
byte-identical to what the archive was measured on, so the roles the fake does not name
are compared against a run with no environment at all.
"""

import json
import pathlib
from typing import Any

from maver.adapters.base import RoleOverlay
from maver.audit.journal import ListSink
from maver.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from maver.memory.factory import build_storage
from maver.orchestration.runner import Deployment, run_incident
from maver.roles.policy import control
from maver.settings.groups import MemorySettings, PipelineSettings, ServerSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

ALERT: dict[str, Any] = {"alertname": "ServiceHealthDegraded", "namespace": "shop"}

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
EMPTY_PLAN: dict[str, Any] = {"steps": [], "expected_outcome": "nothing", "confidence": 0.1}
EXECUTION: dict[str, Any] = {
    "succeeded": True,
    "steps": [{"step_index": 0, "succeeded": True, "observation": "patched"}],
    "final_state_summary": "svc/user has endpoints",
    "confidence": 0.8,
}
SUBMISSIONS = {
    "rca": ("submit_root_cause", ROOT_CAUSE),
    "planning": ("submit_plan", PLAN),
    "execution": ("submit_execution_result", EXECUTION),
}

SUFFIX = "## Environment contract\nAnswer only about the namespace named above."
BLOCK = "## Benchmark-supplied connectivity contract\nProbe service/frontend on 8080."
REPLACEMENT = "You are answering a yes-or-no detection question. Nothing else."


class FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "no resources found"


class OverlayAdapter:
    """An environment that supplies overlays and nothing else.

    Only the agent half of the protocol is exercised by a run, and only one method of
    it: the runner asks for an overlay per role run and takes what it is given.
    """

    name = "fake"

    def __init__(self, **overlays: RoleOverlay) -> None:
        self.asked: list[str] = []
        self._overlays = overlays

    def role_overlay(self, role: str, state: Any) -> RoleOverlay | None:
        self.asked.append(role)
        return self._overlays.get(role)


class Recorder:
    """Answers every role by submitting, after a configurable number of reads."""

    def __init__(self, *, reads: int = 0, submissions: dict[str, Any] | None = None) -> None:
        self.requests: list[tuple[str, ModelRequest]] = []
        self._reads = reads
        self._overrides = submissions or {}

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.requests.append((caller, request))
        turns = sum(1 for role, _ in self.requests if role == caller)
        if turns <= self._reads:
            name: str = "exec_kubectl_cmd_safely"
            arguments: dict[str, Any] = {"command": "kubectl get svc"}
        else:
            name, default = SUBMISSIONS[caller]
            arguments = self._overrides.get(caller, default)
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


def deployment(
    tmp_path: pathlib.Path,
    *,
    model: Recorder,
    adapter: Any = None,
    max_turns: int = 20,
) -> tuple[Deployment, ListSink]:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        pipeline=PipelineSettings(max_turns=max_turns),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
    )
    sink = ListSink()
    return (
        Deployment(
            settings=settings,
            storage=build_storage(settings),
            completion=model,
            profile=PROFILE,
            transport_for=lambda session_id: FakeTransport(),
            sink=sink,
            adapter=adapter,
        ),
        sink,
    )


async def run(deploy: Deployment) -> str:
    session = deploy.storage.sessions.create_session(ALERT)
    await run_incident(ALERT, session.session_id, deploy, fire_post_hooks=False)
    return str(session.session_id)


def first(model: Recorder, role: str) -> ModelRequest:
    """The request that opened one role's run."""
    return next(request for caller, request in model.requests if caller == role)


def body(request: ModelRequest) -> str:
    """The system prompt without its timestamp, which moves between two runs."""
    return str(request.system).split("\n\n", 1)[1]


def opening(request: ModelRequest) -> str:
    """The text of the turn the role opened on."""
    return str(request.messages[0].get("content"))


async def test_an_overlay_narrows_the_role_it_names_and_no_other(
    tmp_path: pathlib.Path,
) -> None:
    """The seam's whole claim, in one comparison against a run with no environment.

    The role the environment names is prompted, opened, and equipped differently. Every
    other role has to come out byte-identical, because the same code path now runs for
    all of them and an overlay that leaked into one of the others would silently change
    what an unrelated arm measured.
    """
    plain_model = Recorder()
    plain, _ = deployment(tmp_path / "plain", model=plain_model)
    await run(plain)

    overlaid_model = Recorder()
    adapter = OverlayAdapter(
        rca=RoleOverlay(
            system_suffix=SUFFIX,
            context_blocks=(BLOCK,),
            drop_tools=frozenset({"bash_exec"}),
        )
    )
    overlaid, _ = deployment(tmp_path / "overlaid", model=overlaid_model, adapter=adapter)
    session_id = await run(overlaid)

    assert adapter.asked == ["rca", "planning", "execution"]

    plain_rca, overlaid_rca = first(plain_model, "rca"), first(overlaid_model, "rca")
    # After the role's own prompt and before the tool-call protocol every role is sent,
    # because the suffix is a statement about the role and the protocol is plumbing.
    guidance = control("tool_failure_guidance")
    composed = body(plain_rca).removesuffix(f"\n\n{guidance}")
    assert body(overlaid_rca) == f"{composed}\n\n{SUFFIX}\n\n{guidance}"
    assert opening(overlaid_rca) == f"{BLOCK}\n\n{opening(plain_rca)}"

    offered = {tool.name for tool in overlaid_rca.tools}
    assert "bash_exec" in {tool.name for tool in plain_rca.tools}
    assert "bash_exec" not in offered
    # Recorded, not merely absent. What the model was offered is not reconstructible
    # from the arm alone, so the reason is written on the run row beside the result.
    withheld = [
        run_.metadata["tools_withheld"].get("bash_exec")
        for run_ in overlaid.storage.runs.runs_for(session_id)
        if run_.agent_name == "rca"
    ]
    assert withheld == ["withheld by the environment"]

    for role in ("planning", "execution"):
        assert body(first(overlaid_model, role)) == body(first(plain_model, role))
        assert opening(first(overlaid_model, role)) == opening(first(plain_model, role))


async def test_a_replacement_takes_the_place_of_the_composed_prompt(
    tmp_path: pathlib.Path,
) -> None:
    """A benchmark asking a question the role was not written for replaces the prompt
    rather than arguing with it in a suffix."""
    model = Recorder()
    adapter = OverlayAdapter(
        rca=RoleOverlay(system_replacement=REPLACEMENT, system_suffix="ignored")
    )
    deploy, _ = deployment(tmp_path, model=model, adapter=adapter)
    await run(deploy)

    system = body(first(model, "rca"))
    assert system == f"{REPLACEMENT}\n\n{control('tool_failure_guidance')}"
    assert "ignored" not in system


async def test_a_shorter_budget_is_taken_and_a_longer_one_is_not(
    tmp_path: pathlib.Path,
) -> None:
    """`min`, in both directions, is what makes this a narrowing rather than a lever.

    Counted in normal turns — the ones the model is offered its investigation tools on —
    because a rescue turn is offered the submission alone and does not spend the budget.
    """

    async def normal_turns(overlay: RoleOverlay, at: pathlib.Path) -> int:
        model = Recorder(reads=99)
        deploy, _ = deployment(at, model=model, adapter=OverlayAdapter(rca=overlay), max_turns=3)
        await run(deploy)
        return sum(
            1
            for caller, request in model.requests
            if caller == "rca"
            and any(tool.name == "exec_kubectl_cmd_safely" for tool in request.tools)
        )

    assert await normal_turns(RoleOverlay(max_turns=1), tmp_path / "short") == 1
    assert await normal_turns(RoleOverlay(max_turns=99), tmp_path / "long") == 3


async def test_the_environments_guard_answers_first_and_the_roles_own_still_answers(
    tmp_path: pathlib.Path,
) -> None:
    """Composition, and the record of it.

    The environment's guard is consulted in front of the role's own, so a submission it
    refuses never reaches a policy that would have accepted it. It cannot write the
    answer: the model produces the correction itself, and the refusal lands in the event
    log through the same path a schema rejection does, because it is an intervention on
    the model's answer and the run is only reconstructible if it says so.
    """
    refusals: list[int] = []

    def refuse_once(args: dict[str, Any], tool_calls_made: int, messages: list[Any]) -> Any:
        refusals.append(tool_calls_made)
        return "the answer does not name a namespace" if len(refusals) == 1 else None

    model = Recorder(submissions={"planning": EMPTY_PLAN})
    adapter = OverlayAdapter(planning=RoleOverlay(submit_guard=refuse_once))
    deploy, sink = deployment(tmp_path, model=model, adapter=adapter)
    await run(deploy)

    rejections = [
        event.payload.validation_error
        for event in sink.events
        if event.payload.kind == "submission" and not event.payload.accepted
    ]
    # The environment's refusal first, then the role's own on the next submission, which
    # the environment accepted. A composition that stopped at the first guard would have
    # let an empty plan through.
    assert rejections[0] == "the answer does not name a namespace"
    assert "submit_plan has no steps" in str(rejections[1])
