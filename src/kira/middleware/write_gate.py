"""One gate, covering every tool, in front of every call.

The prototype called its approval check from four hand-picked places. That works exactly
until someone adds a fifth write-capable tool, and nothing anywhere reports that it was
missed. Here the gate is a ``wrap_tool_call`` hook: a tool added later is gated because
of where the hook sits, not because its author remembered.

Everything about it fails closed. A tool with no policy is treated as a write. A
classifier that raises denies. A gate that cannot reach its own store denies. Denying is
survivable — the model reads the refusal as a tool result and keeps working — whereas
dispatching something nobody authorised is not.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal

from kira.audit.journal import Journal
from kira.contracts.events import PolicyDecisionEvent
from kira.llm.types import ToolResult
from kira.middleware.authority import ActionRequest, ApprovalPending, Authority, Decision
from kira.middleware.base import ToolHandler, ToolRequest
from kira.settings.authority import AuthoritySettings
from kira.tools.policy import ActionClass, ToolPolicy, policy_for

__all__ = ["WriteGate"]

_log = logging.getLogger(__name__)

_UNCLASSIFIED = ToolPolicy(classify=lambda _: ActionClass.WRITE)
"""What an unregistered tool gets. A tool nobody classified is a tool nobody vouched for."""


class WriteGate:
    """Puts every write-capable call through the configured authority."""

    name = "write_gate"

    def __init__(
        self,
        *,
        authority: Authority,
        settings: AuthoritySettings,
        journal: Journal,
    ) -> None:
        """Build the gate.

        Args:
            authority: Who decides.
            settings: The policy rules and the stages allowed to write.
            journal: Where decisions are recorded.
        """
        self._authority = authority
        self._settings = settings
        self._journal = journal

    def _record(
        self,
        request: ToolRequest,
        *,
        policy: Literal["auto", "prompt", "deny"],
        decision: Literal["approved", "denied", "timed_out", "pending"],
        reason: str,
        request_id: str | None = None,
        decided_by: str | None = None,
    ) -> None:
        self._journal.record(
            PolicyDecisionEvent(
                tool=request.name,
                policy=policy,
                decision=decision,
                reason=reason,
                request_id=request_id,
                decided_by=decided_by,
            )
        )

    @staticmethod
    def _refuse(request: ToolRequest, text: str) -> ToolResult:
        return ToolResult(tool_call_id=request.call_id, name=request.name, content=text)

    def _may_write(self, policy: ToolPolicy, stage: str) -> bool:
        return stage in (policy.write_stages | self._settings.additional_write_stages)

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> ToolResult:
        """Decide, then dispatch or refuse.

        Args:
            request: The pending call.
            handler: The next hook, or dispatch.

        Returns:
            The tool's result, or a refusal in its place.

        Raises:
            ApprovalPending: When a human has to answer. Propagated unchanged so the
                pipeline can checkpoint and exit.
        """
        policy = policy_for(request.name) or _UNCLASSIFIED
        arguments: Mapping[str, Any] = request.arguments

        try:
            action = policy.classify(arguments)
        except Exception as exc:  # a classifier that cannot answer has not said "yes"
            _log.warning("policy for %s failed to classify and was denied", request.name)
            self._record(
                request,
                policy="deny",
                decision="denied",
                reason=f"authority unavailable: {type(exc).__name__}",
            )
            return self._refuse(
                request, f"denied by HITL gate: authority unavailable: {type(exc).__name__}"
            )

        # A blocked command is the tool's own refusal, with its own wording, and that
        # wording is what the role prompts describe. Do not intercept it, and do not
        # offer it for approval: a command nobody may run is not one an operator may
        # authorise. Reads pass silently — a read is not an authority decision, and one
        # event per telemetry call would bury the record that matters in noise.
        if action in (ActionClass.READ, ActionClass.BLOCKED):
            return await handler(request)

        selectors_now = policy.selectors(arguments)
        governing = self._settings.rule_for(
            tool=request.name, stage=request.role, selectors=selectors_now
        )
        if action is ActionClass.EXEC and (governing is None or governing.mode == "auto"):
            # Permitted for every role, as measured, but never silently: the record is
            # what makes the residual risk auditable rather than merely disclosed.
            self._record(
                request,
                policy="auto",
                decision="approved",
                reason="in-container command; authority not provable",
            )
            return await handler(request)

        if not self._may_write(policy, request.role):
            refusal = policy.stage_refusal(request.role)
            self._record(
                request,
                policy="deny",
                decision="denied",
                reason=f"stage {request.role} may not write",
            )
            return self._refuse(request, refusal)

        selectors = selectors_now
        rule = governing
        mode = rule.mode if rule is not None else self._settings.mode

        action_request = ActionRequest(
            session_id=request.session_id,
            stage=request.role,
            tool=request.name,
            tool_call_id=request.call_id,
            arguments=arguments,
            action=action,
            summary=policy.summarise(arguments),
            selectors=selectors,
            targets=policy.targets(arguments),
            rule=rule.name if rule is not None else None,
        )

        if mode == "deny":
            decision = Decision(
                outcome="denied",
                policy="deny",
                reason=f"denied by rule {rule.name}" if rule else "writes are disabled",
            )
        else:
            try:
                decision = await self._authority.authorise(action_request)
            except ApprovalPending:
                self._record(
                    request, policy="prompt", decision="pending", reason="awaiting a human"
                )
                raise
            except Exception as exc:  # an authority that cannot answer has not said "yes"
                _log.warning("authority failed for %s and the call was denied", request.name)
                self._record(
                    request,
                    policy=mode,
                    decision="denied",
                    reason=f"authority unavailable: {type(exc).__name__}",
                )
                return self._refuse(
                    request, f"denied by HITL gate: authority unavailable: {type(exc).__name__}"
                )

        self._record(
            request,
            policy=decision.policy,
            decision=decision.outcome,
            reason=decision.reason,
            request_id=decision.request_id,
            decided_by=decision.decided_by,
        )
        if decision.outcome == "denied":
            # Byte-identical to the prototype's wording. The model reads this as a tool
            # result and carries on, and the containment analysis keys on it.
            return self._refuse(request, f"denied by HITL gate: {decision.reason}")
        return await handler(request)
