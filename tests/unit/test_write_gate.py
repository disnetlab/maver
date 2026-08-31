import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.events import PolicyDecisionEvent
from kira.llm.types import ToolResult
from kira.middleware.authority import ActionRequest, ApprovalPending, AutoApprove, Decision
from kira.middleware.base import ToolRequest
from kira.middleware.write_gate import WriteGate
from kira.settings.authority import AuthorityRule, AuthoritySettings
from kira.tools import kubernetes  # noqa: F401 - registers the kubectl policy
from kira.tools.policy import policy_for

DISPATCHED = "dispatched"


async def handler(request: ToolRequest) -> ToolResult:
    return ToolResult(tool_call_id=request.call_id, name=request.name, content=DISPATCHED)


def call(
    cmd: str, *, role: str = "execution", tool: str = "exec_kubectl_cmd_safely"
) -> ToolRequest:
    return ToolRequest(call_id="c1", name=tool, arguments={"cmd": cmd}, role=role, session_id="s1")


def gate(settings: AuthoritySettings, sink: ListSink, authority: object = None) -> WriteGate:
    return WriteGate(
        authority=authority or AutoApprove(),
        settings=settings,
        journal=Journal(sink, session_id="s1", stage="execution"),
    )


def decisions(sink: ListSink) -> list[PolicyDecisionEvent]:
    return [e.payload for e in sink.events if isinstance(e.payload, PolicyDecisionEvent)]


# --- coverage: the property the prototype's four hand-picked call sites could not hold --


async def test_a_tool_with_no_policy_is_treated_as_a_write() -> None:
    """A tool nobody classified is a tool nobody vouched for.

    This is the property that makes coverage structural: a write-capable tool added
    later is gated because of where the hook sits, not because its author remembered.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink).wrap_tool_call(
        call("anything", tool="a_tool_added_later"), handler
    )
    assert result.content != DISPATCHED
    # It declares no write stage, so no stage may call it until someone classifies it.
    assert result.content == "error: writes are not permitted for the execution agent (read-only)"


@pytest.mark.parametrize("stage", ["execution", "single.mitigation"])
async def test_both_write_capable_stages_may_write(stage: str) -> None:
    """The single-agent configuration writes under its own stage name.

    A gate keyed on a list that names only `execution` lets that entire configuration's
    writes through ungated, and nothing reports it.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="auto"), sink).wrap_tool_call(
        call("kubectl delete pod x -n shop", role=stage), handler
    )
    assert result.content == DISPATCHED


async def test_a_diagnostic_role_cannot_write() -> None:
    """Refused before dispatch, in the tool's own wording, and recorded as a refusal."""
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="auto"), sink).wrap_tool_call(
        call("kubectl delete pod x -n shop", role="rca"), handler
    )
    assert result.content == "error: kubectl writes are not permitted for the rca agent (read-only)"
    assert [d.decision for d in decisions(sink)] == ["denied"]


async def test_a_read_passes_without_an_authority_event() -> None:
    """A read is not an authority decision.

    One event per telemetry call would bury the record that matters under the record
    that does not.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink).wrap_tool_call(
        call("kubectl get pods -n shop", role="rca"), handler
    )
    assert result.content == DISPATCHED
    assert decisions(sink) == []


async def test_a_blocked_command_keeps_the_tool_s_own_refusal() -> None:
    """The gate must not offer for approval something nobody may run.

    The tool's wording is what the role prompts describe, so intercepting it here would
    tell the model something different from what its prompt says.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink).wrap_tool_call(
        call("kubectl explain pod", role="execution"), handler
    )
    assert result.content == DISPATCHED  # passed through to the tool, which refuses it
    assert decisions(sink) == []


# --- the exec class ---------------------------------------------------------------


async def test_an_in_container_command_passes_but_is_recorded() -> None:
    """Permitted for every role as measured, but never silently.

    Its authority cannot be proven, so the record is what makes the residual risk
    auditable instead of merely disclosed.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="prompt"), sink).wrap_tool_call(
        call("kubectl exec pod -- cat /etc/hosts", role="rca"), handler
    )
    assert result.content == DISPATCHED
    assert [d.reason for d in decisions(sink)] == ["in-container command; authority not provable"]


async def test_a_rule_can_hold_an_in_container_command() -> None:
    """A production profile can deny exec without a code change."""
    settings = AuthoritySettings(
        mode="auto",
        rules=(AuthorityRule(name="no-exec", selectors=("verb:exec",), mode="deny"),),
    )
    sink = ListSink()
    result = await gate(settings, sink).wrap_tool_call(
        call("kubectl exec pod -- sh -c whoami", role="execution"), handler
    )
    assert result.content == "denied by HITL gate: denied by rule no-exec"


# --- rules ------------------------------------------------------------------------


async def test_a_rule_can_auto_approve_inside_a_prompting_deployment() -> None:
    """Neither a global mode nor a list of exempt tools can express this."""
    settings = AuthoritySettings(
        mode="prompt",
        rules=(AuthorityRule(name="routine-scaling", selectors=("verb:scale",), mode="auto"),),
    )
    sink = ListSink()
    result = await gate(settings, sink).wrap_tool_call(
        call("kubectl scale deploy/web --replicas=3 -n shop"), handler
    )
    assert result.content == DISPATCHED
    assert decisions(sink)[0].reason == "auto-approved by rule routine-scaling"


async def test_a_rule_can_deny_a_resource_class_inside_an_auto_deployment() -> None:
    """The case two frozensets could not say, which is why rules replaced them."""
    settings = AuthoritySettings(
        mode="auto",
        rules=(
            AuthorityRule(name="never-touch-secrets", selectors=("resource:secret*",), mode="deny"),
        ),
    )
    sink = ListSink()
    result = await gate(settings, sink).wrap_tool_call(
        call("kubectl delete secret db-creds -n shop"), handler
    )
    assert result.content == "denied by HITL gate: denied by rule never-touch-secrets"


async def test_the_first_matching_rule_wins() -> None:
    settings = AuthoritySettings(
        mode="deny",
        rules=(
            AuthorityRule(name="first", tools=("exec_kubectl_cmd_safely",), mode="auto"),
            AuthorityRule(name="second", tools=("*",), mode="deny"),
        ),
    )
    sink = ListSink()
    result = await gate(settings, sink).wrap_tool_call(
        call("kubectl delete pod x -n shop"), handler
    )
    assert result.content == DISPATCHED


# --- failing closed ---------------------------------------------------------------


async def test_an_authority_that_raises_denies_rather_than_ending_the_run() -> None:
    """A gate failure must not kill a run, and must not let the call through either.

    Denying is survivable: the model reads the refusal and keeps working. Dispatching
    something nobody authorised is not.
    """

    class Broken:
        async def authorise(self, request: ActionRequest) -> Decision:
            raise RuntimeError("store unreachable")

    sink = ListSink()
    result = await gate(AuthoritySettings(mode="prompt"), sink, Broken()).wrap_tool_call(
        call("kubectl delete pod x -n shop"), handler
    )
    assert result.content == "denied by HITL gate: authority unavailable: RuntimeError"
    assert [d.decision for d in decisions(sink)] == ["denied"]


async def test_a_pending_approval_unwinds_the_run_and_is_recorded() -> None:
    """The pause is not a failure. It propagates so the pipeline can checkpoint."""

    class Human:
        async def authorise(self, request: ActionRequest) -> Decision:
            raise ApprovalPending(
                request_id="r1",
                session_id=request.session_id,
                stage=request.stage,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool,
                tool_arguments=dict(request.arguments),
            )

    sink = ListSink()
    with pytest.raises(ApprovalPending):
        await gate(AuthoritySettings(mode="prompt"), sink, Human()).wrap_tool_call(
            call("kubectl delete pod x -n shop"), handler
        )
    assert [d.decision for d in decisions(sink)] == ["pending"]


def test_the_operator_sees_the_whole_command_not_a_preview() -> None:
    """A reviewer who cannot see the patch body is not reviewing anything.

    The prototype summarised a write as its first six argument tokens, so the part that
    says what actually changes appeared nowhere.
    """
    policy = policy_for("exec_kubectl_cmd_safely")
    assert policy is not None
    cmd = 'kubectl patch deployment web -n shop --type merge -p \'{"spec":{"replicas":0}}\''
    assert policy.summarise({"cmd": cmd}) == cmd
    assert "deployment/web" in policy.targets({"cmd": cmd})
