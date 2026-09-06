import pathlib

import pytest

from kira.audit.journal import Journal, ListSink
from kira.audit.store_sink import StoreSink
from kira.contracts.events import PolicyDecisionEvent, ToolCallEvent
from kira.llm.types import ToolResult
from kira.memory.factory import build_storage
from kira.memory.ports import AgentRunRecord
from kira.middleware.authority import (
    ActionRequest,
    ApprovalPending,
    Authority,
    AutoApprove,
    Decision,
    DenyAll,
)
from kira.middleware.base import ToolRequest
from kira.middleware.write_gate import WriteGate
from kira.settings.authority import AuthorityRule, AuthoritySettings
from kira.settings.groups import ServerSettings, ToolSettings
from kira.settings.root import Settings
from kira.tools import kubernetes  # noqa: F401 - registers the kubectl policy
from kira.tools.policy import policy_for

DISPATCHED = "dispatched"


async def handler(request: ToolRequest) -> ToolResult:
    return ToolResult(tool_call_id=request.call_id, name=request.name, content=DISPATCHED)


def call(
    cmd: str, *, role: str = "execution", tool: str = "exec_kubectl_cmd_safely"
) -> ToolRequest:
    return ToolRequest(call_id="c1", name=tool, arguments={"cmd": cmd}, role=role, session_id="s1")


class NeverAnswers:
    """Stands in for the person, and never gets round to it.

    Every test here that expects a dispatch under a prompting mode is only meaningful
    if reaching the human is loud. Wiring one auto-approving object into all three modes
    made those tests pass while the rules that were supposed to produce the dispatch
    were doing nothing.
    """

    async def authorise(self, request: ActionRequest) -> Decision:
        raise ApprovalPending(
            request_id="r1",
            session_id=request.session_id,
            stage=request.stage,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool,
            tool_arguments=dict(request.arguments),
        )


def gate(
    settings: AuthoritySettings,
    sink: ListSink,
    human: Authority | None = None,
    tools: ToolSettings | None = None,
) -> WriteGate:
    return WriteGate(
        authorities={"auto": AutoApprove(), "deny": DenyAll(), "prompt": human or NeverAnswers()},
        settings=settings,
        tools=tools or ToolSettings(),
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
    """Permitted for every role under an unattended deployment, as measured, but never
    silently. Its authority cannot be proven, so the record is what makes the residual
    risk auditable instead of merely disclosed.
    """
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="auto"), sink).wrap_tool_call(
        call("kubectl exec pod -- cat /etc/hosts", role="rca"), handler
    )
    assert result.content == DISPATCHED
    assert [d.reason for d in decisions(sink)] == ["in-container command; authority not provable"]


async def test_an_in_container_command_is_not_exempt_from_the_authority() -> None:
    """`kubectl exec pod -- rm -rf /data` dispatched under `prompt` and under `deny`.

    The exec class exists so an action whose authority cannot be proven is recorded as
    one, and so a role that may not write can still read from inside a container. It was
    read as a pass out of the gate altogether, which made the deployment posture that
    exists to refuse cluster writes unable to refuse the least provable of them.
    """
    destructive = "kubectl exec pod/x -n shop -- rm -rf /data"

    sink = ListSink()
    with pytest.raises(ApprovalPending):
        await gate(AuthoritySettings(mode="prompt"), sink).wrap_tool_call(
            call(destructive), handler
        )
    assert [d.decision for d in decisions(sink)] == ["pending"]

    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink).wrap_tool_call(
        call(destructive), handler
    )
    assert result.content == "denied by HITL gate: writes are disabled"


async def test_an_in_container_command_keeps_its_exemption_from_the_stage_check() -> None:
    """Only the authority binds it. Reading a file inside a container is how a
    diagnosing role sees a config the API server never had, and every measured run
    allowed it from every role."""
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="auto"), sink).wrap_tool_call(
        call("kubectl exec pod -- cat /etc/nginx/nginx.conf", role="rca"), handler
    )
    assert result.content == DISPATCHED


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
    """Neither a global mode nor a list of exempt tools can express this.

    The human here never answers, so an inert rule raises instead of dispatching.
    """
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


async def test_a_resource_rule_holds_however_the_command_was_written() -> None:
    """The worked example, defeated by ordinary flag order and by a plural.

    Every command here is one kubectl accepts and runs against the same secret or the
    same ingress. The selectors were read off the token after the verb and folded with
    `rstrip("s")`, so the deployment held one spelling and dispatched the rest — with
    the audit event recording `rule=None`, so nothing said the rule was meant to apply.
    """
    settings = AuthoritySettings(
        mode="auto",
        rules=(
            AuthorityRule(name="never-touch-secrets", selectors=("resource:secret",), mode="deny"),
            AuthorityRule(name="hold-ingress", selectors=("resource:ingress",), mode="deny"),
        ),
    )
    spellings = [
        ("kubectl patch -n shop --type merge -p {} secret db-creds", "never-touch-secrets"),
        ("kubectl delete -o name secret db-creds -n shop", "never-touch-secrets"),
        ("kubectl delete -- secret db-creds -n shop", "never-touch-secrets"),
        ("kubectl delete ingress web -n shop", "hold-ingress"),
        ("kubectl delete ingresses web -n shop", "hold-ingress"),
    ]
    for cmd, rule in spellings:
        result = await gate(settings, ListSink()).wrap_tool_call(call(cmd), handler)
        assert result.content == f"denied by HITL gate: denied by rule {rule}", cmd


async def test_a_rule_can_hold_a_resource_class_for_a_person_inside_an_auto_deployment() -> None:
    """The mode a rule names has to reach a decider that can honour it.

    One authority object was chosen from the global mode and held for the whole process,
    so a rule was only ever able to tighten to `deny` — every other mode it named landed
    on whatever the deployment had already picked. A rule reading `mode: prompt` under a
    global `auto`, which is the worked example in this project's own specification,
    auto-approved and recorded the rule's name beside an approval it never made.
    """
    settings = AuthoritySettings(
        mode="auto",
        rules=(
            AuthorityRule(
                name="secrets-need-a-human", selectors=("resource:secret*",), mode="prompt"
            ),
        ),
    )
    sink = ListSink()
    with pytest.raises(ApprovalPending):
        await gate(settings, sink).wrap_tool_call(
            call("kubectl delete secret db-password -n shop"), handler
        )
    assert [d.decision for d in decisions(sink)] == ["pending"]

    # Anything the rule does not match still runs unattended.
    sink = ListSink()
    result = await gate(settings, sink).wrap_tool_call(
        call("kubectl delete pod x -n shop"), handler
    )
    assert result.content == DISPATCHED


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


async def test_a_screen_the_deployment_lifted_does_not_send_a_write_past_the_gate() -> None:
    """`blocked` is the one class the gate passes straight through, on the grounds that
    the tool refuses it in its own wording. That reasoning holds only while the gate and
    the tool read the same screens.

    The kubectl argument screen is configurable — it is an evaluation-integrity knob and
    a real cluster may have to clear it — and a gate classifying against the shipped
    pattern would call this `blocked`, pass it through, and hand a cluster write to a
    tool that no longer refuses it, with no authority decision anywhere.
    """
    lifted = ToolSettings(kubectl_denied_argument_pattern="")
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink, tools=lifted).wrap_tool_call(
        call("kubectl delete deployment chaos-controller -n chaos-mesh"), handler
    )
    assert result.content == "denied by HITL gate: writes are disabled"
    assert [d.decision for d in decisions(sink)] == ["denied"]

    # With the shipped pattern the tool still refuses it, in its own wording, and the
    # gate still stays out of the way.
    sink = ListSink()
    result = await gate(AuthoritySettings(mode="deny"), sink).wrap_tool_call(
        call("kubectl delete deployment chaos-controller -n chaos-mesh"), handler
    )
    assert result.content == DISPATCHED  # passed through to the tool, which refuses it
    assert decisions(sink) == []


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
    result = await gate(AuthoritySettings(mode="prompt"), sink, human=Broken()).wrap_tool_call(
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
        await gate(AuthoritySettings(mode="prompt"), sink, human=Human()).wrap_tool_call(
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


async def test_a_refusal_joins_to_the_call_it_refused(tmp_path: pathlib.Path) -> None:
    """The store carries a ``tool_call_id`` column and the sink filled it from an
    attribute no payload declared, so it was unconditionally NULL: a tool call could
    not be aligned to its position in the stored transcript, and a refusal could be
    joined to the call it gated only by adjacency in a log that had no order either."""
    stack = build_storage(Settings(server=ServerSettings(workdir=tmp_path)))
    stack.runs.start_run(
        AgentRunRecord(
            agent_run_id="r1",
            pipeline_id="s1",
            agent_name="rca",
            attempt=1,
            status="running",
            started_at="2026-09-01T00:00:00Z",
        )
    )
    sink = StoreSink(stack.runs)
    journal = Journal(sink, session_id="s1", stage="rca", run_id="r1")
    gate = WriteGate(
        authorities={"auto": AutoApprove()},
        settings=AuthoritySettings(mode="auto"),
        tools=ToolSettings(),
        journal=journal,
    )
    request = call("kubectl delete pod x -n shop", role="rca")

    result = await gate.wrap_tool_call(request, handler)
    journal.record(
        ToolCallEvent(
            tool=request.name,
            call_id=request.call_id,
            arguments=dict(request.arguments),
            succeeded=False,
            result_chars=len(result.content),
        )
    )
    sink.close()

    stored = {record.event_type: record for record in stack.runs.events_for("r1")}
    assert stored["policy_decision"].tool_call_id == stored["tool_call"].tool_call_id == "c1"
    stack.close()
