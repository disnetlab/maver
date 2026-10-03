"""The authority that asks a person.

Asking is not blocking. A write that needs a human raises, the run unwinds cleanly, and
everything needed to continue is written to the database — so an approval that arrives
twenty minutes later, or after a restart, resumes the same incident rather than starting
a new one. Holding an event loop open for the length of a human's attention span is the
alternative, and it makes an approval a liveness problem.

The pause is deliberately atomic from a reader's point of view: the request row, the
timeout job and the session's parked status are written before the exception leaves, so
a poller never sees a session waiting on a request that does not exist yet.

A request records the whole call, not a preview of it. The prototype summarised a write
as its first six argument tokens, so a patch showed the resource and not the patch body
— the only part that says what changes. Approving that is not review.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from maver.memory.ports import ApprovalRecord, ApprovalStore
from maver.middleware.authority import (
    ActionRequest,
    ApprovalPending,
    Authority,
    AutoApprove,
    Decision,
    DenyAll,
)
from maver.settings.authority import AuthoritySettings, Mode
from maver.tools.policy import ActionClass

__all__ = [
    "ApprovalGrant",
    "HumanApproval",
    "TimeoutScheduler",
    "approval_granted",
    "authorities",
    "take_grant",
]

_log = logging.getLogger(__name__)

TimeoutScheduler = Callable[[str, str], None]
"""Arms one request's expiry, given its id and when it expires."""

_NOTIFICATIONS: set[asyncio.Task[None]] = set()
"""Notifications still in flight.

A task with no reference is collectable mid-await, so the set is what keeps a
fire-and-forget post alive long enough to be sent.
"""


class ApprovalGrant:
    """A decided request, being replayed exactly once."""

    def __init__(self, request_id: str, tool: str, arguments: Mapping[str, Any]) -> None:
        """Record what was approved.

        Args:
            request_id: The decided request.
            tool: The tool that was approved.
            arguments: The arguments that were approved.
        """
        self.request_id = request_id
        self.tool = tool
        self.arguments = dict(arguments)

    def matches(self, tool: str, arguments: Mapping[str, Any]) -> bool:
        """Say whether a pending call is the one this grant covers.

        Args:
            tool: The tool about to run.
            arguments: The arguments it was given.

        Returns:
            True only when both are identical to what was approved.
        """
        return self.tool == tool and self.arguments == dict(arguments)


_GRANT: ContextVar[ApprovalGrant | None] = ContextVar("approval_grant", default=None)


@contextmanager
def approval_granted(request_id: str, *, tool: str, arguments: Mapping[str, Any]) -> Iterator[None]:
    """Replay one approved call, once.

    Args:
        request_id: The decided request.
        tool: The tool that was approved.
        arguments: The arguments that were approved.

    Yields:
        Nothing. The grant is visible to the gate for the length of the block.
    """
    token = _GRANT.set(ApprovalGrant(request_id, tool, arguments))
    try:
        yield
    finally:
        _GRANT.reset(token)


def take_grant() -> ApprovalGrant | None:
    """Consume the grant covering this call, if there is one.

    Taken rather than read, so a grant covers exactly one dispatch. A grant left in
    place would approve every subsequent write in the same block, which is the failure
    mode a resume path is most likely to have and least likely to notice.

    Returns:
        The grant, or None.
    """
    grant = _GRANT.get()
    if grant is not None:
        _GRANT.set(None)
    return grant


class HumanApproval:
    """Puts a write to a person and unwinds the run until they answer."""

    def __init__(
        self,
        *,
        store: ApprovalStore,
        settings: AuthoritySettings,
        arm_timeout: TimeoutScheduler | None = None,
    ) -> None:
        """Build the authority.

        Args:
            store: Where requests live.
            settings: The gate's configuration, including how long a request waits.
            arm_timeout: Schedules the expiry. Injected rather than reached for, because
                the scheduler lives above this layer and a request that waits forever is
                a policy choice a deployment is allowed to make.
        """
        self._store = store
        self._settings = settings
        self._arm = arm_timeout

    async def authorise(self, request: ActionRequest) -> Decision:
        """Record the request, arm its timeout, and unwind.

        Args:
            request: What is about to happen.

        Returns:
            Never returns normally.

        Raises:
            ApprovalPending: Always. A decision that has not been made cannot be
                represented as an outcome, and returning "denied" here would record a
                refusal nobody made.
        """
        request_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        expires_at = (
            (now + timedelta(seconds=self._settings.timeout_s)).isoformat()
            if self._settings.timeout_s is not None
            else None
        )
        record = ApprovalRecord(
            request_id=request_id,
            session_id=request.session_id,
            kind="tool_call",
            stage=request.stage,
            status="pending",
            summary=request.summary,
            payload={
                "tool": request.tool,
                "arguments": dict(request.arguments),
                "action": request.action.value,
                "selectors": list(request.selectors),
                "targets": list(request.targets),
                "rule": request.rule,
                "run_id": request.run_id,
                "requires_human_approval": (
                    request.action is ActionClass.EXEC
                    and self._settings.kubectl_exec_requires_approval
                ),
            },
            tool_call_id=request.tool_call_id,
            tool_name=request.tool,
            created_at=now.isoformat(),
            expires_at=expires_at,
        )
        # Off the loop. This runs on every gated write, and it is the row a person is
        # waiting to be shown: written inline it would hold the loop, and with it every
        # other incident in the process and the route the operator reads it through.
        # Ordered before the raise as it always was, so the pause is still atomic to a
        # reader — a session never waits on a request that does not exist yet.
        await asyncio.to_thread(self._store.create_request, record)
        if expires_at is not None and self._arm is not None:
            try:
                self._arm(request_id, expires_at)
            except Exception:
                # A request with no expiry waits for a person indefinitely. A pause that
                # failed because a queue was unavailable would lose the write entirely,
                # which is worse.
                _log.exception("could not arm the timeout for approval request %s", request_id)

        self._notify(record)

        raise ApprovalPending(
            request_id=request_id,
            session_id=request.session_id,
            stage=request.stage,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool,
            tool_arguments=dict(request.arguments),
        )

    def _notify(self, record: ApprovalRecord) -> None:
        """Tell whoever is meant to answer that there is something to answer.

        Detached from the pause on purpose. The row is already written, so the request
        is reachable through the API and the CLI whatever happens here; a notification
        awaited inline would put a third party's availability in front of a write the
        run has already unwound for.
        """
        if self._settings.webhook_url is None:
            return
        task = asyncio.create_task(self._post(record))
        _NOTIFICATIONS.add(task)
        task.add_done_callback(_NOTIFICATIONS.discard)

    async def _post(self, record: ApprovalRecord) -> None:
        """Send one notification, and swallow whatever goes wrong with it."""
        try:
            async with httpx.AsyncClient(timeout=self._settings.webhook_timeout_s) as client:
                await client.post(
                    str(self._settings.webhook_url),
                    json={
                        "request_id": record.request_id,
                        "session_id": record.session_id,
                        "stage": record.stage,
                        "tool": record.tool_name,
                        "summary": record.summary,
                        "expires_at": record.expires_at,
                        "link": f"/api/v1/hitl/{record.request_id}",
                    },
                )
        except Exception:
            _log.warning(
                "could not notify %s about approval request %s; it is still waiting in the store",
                self._settings.webhook_url,
                record.request_id,
            )


def authorities(
    settings: AuthoritySettings,
    store: ApprovalStore,
    arm_timeout: TimeoutScheduler,
) -> dict[Mode, Authority]:
    """Build one decider per mode.

    All three exist regardless of the deployment's own mode, because a policy rule names
    a mode of its own and the gate has to be able to honour it. Choosing a single object
    from the global mode made every rule inert but ``deny``: a rule tightening a routine
    deployment to ``prompt`` — the worked example in this project's own specification —
    approved automatically and recorded the rule's name beside the approval it never
    made.

    Args:
        settings: The gate's configuration, passed to the human authority.
        store: Where a pending request is written.
        arm_timeout: Arms one request's expiry.

    Returns:
        A decider for each of the three modes.
    """
    return {
        "auto": AutoApprove(),
        "deny": DenyAll(),
        "prompt": HumanApproval(store=store, settings=settings, arm_timeout=arm_timeout),
    }
