"""The decision itself: who says yes, and how a run waits for them.

Three implementations, because the three modes are genuinely different shapes. Two of
them answer immediately. The third cannot: a human may take half an hour, and holding a
model loop open that long would burn a session's budget on waiting. So it unwinds the run
instead, and the pipeline resumes from persisted state once a decision arrives.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from kira.tools.policy import ActionClass

__all__ = [
    "ActionRequest",
    "ApprovalPending",
    "Authority",
    "AutoApprove",
    "Decision",
    "DenyAll",
]


@dataclass(frozen=True)
class ActionRequest:
    """A resolved description of what is about to happen.

    Carries the whole call rather than a truncated preview. The prototype summarised a
    write as its first six argument tokens, so a patch rendered as
    ``kubectl patch deployment foo -n bar --type strategic …`` and the patch body — the
    only part that says what actually changes — appeared nowhere an operator could see.
    Approving that is not review.
    """

    session_id: str
    stage: str
    tool: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    action: ActionClass
    summary: str
    selectors: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    run_id: str | None = None
    rule: str | None = None
    """Which policy rule matched, recorded so a decision can be explained afterwards."""


@dataclass(frozen=True)
class Decision:
    """What the authority said."""

    outcome: Literal["approved", "denied"]
    policy: Literal["auto", "prompt", "deny"]
    reason: str
    request_id: str | None = None
    decided_by: str | None = None


@dataclass
class ApprovalPending(Exception):
    """Raised to unwind a run that is waiting on a human.

    Not an error. It carries the state needed to resume, and the pipeline catches it,
    checkpoints, and exits cleanly.

    ``messages`` is empty where this is raised and filled in by the role loop on the way
    out, because the loop owns the transcript and the gate does not see it.
    """

    request_id: str
    session_id: str
    stage: str
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    """The incident state the paused role was entered with, attached by the node wrapper
    on the way out. Empty here for the same reason ``messages`` is: the gate sees one
    call, not the run around it."""
    run_id: str = ""


@runtime_checkable
class Authority(Protocol):
    """Decides whether one action may proceed."""

    async def authorise(self, request: ActionRequest) -> Decision:
        """Rule on an action.

        Args:
            request: What is about to happen.

        Returns:
            The decision.

        Raises:
            ApprovalPending: When a human has to answer and the run must unwind.
        """
        ...


class AutoApprove:
    """Approves every action, and records that it did.

    Automatic is not the same as unrecorded. A run under this authority still writes a
    policy decision per write, which is what lets a later reader tell "a human approved
    this" from "nobody was asked".
    """

    async def authorise(self, request: ActionRequest) -> Decision:
        """Approve the action.

        Args:
            request: What is about to happen.

        Returns:
            An approval.
        """
        return Decision(
            outcome="approved",
            policy="auto",
            reason=f"auto-approved by rule {request.rule}" if request.rule else "auto-approved",
        )


class DenyAll:
    """Refuses every action.

    Useful as a real deployment posture — run the diagnosis, never touch the cluster —
    and not only as a test double.
    """

    async def authorise(self, request: ActionRequest) -> Decision:
        """Refuse the action.

        Args:
            request: What is about to happen.

        Returns:
            A refusal.
        """
        return Decision(
            outcome="denied",
            policy="deny",
            reason=f"denied by rule {request.rule}" if request.rule else "writes are disabled",
        )
