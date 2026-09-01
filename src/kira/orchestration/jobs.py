"""Work the scheduler restores by name.

Every function here must be top-level: APScheduler serialises a job by dotted import
path and restores it that way after a restart, so a closure or a lambda is a job that
does not survive one. That constraint is the reason this module exists at all, and it is
why these functions take a session or request identifier rather than the objects they
will need.

The deployment is bound once, by the composition root, and read back here. It is the one
process-wide handle in the system and it is deliberate: a restored job arrives with
nothing but its keyword arguments, so something has to tell it which deployment it
belongs to. Everything else takes its dependencies as parameters.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from kira.orchestration.runner import Deployment, resume_after_decision
from kira.orchestration.scheduler import scheduler_for

__all__ = [
    "bind_deployment",
    "bound_deployment",
    "expire_approval",
    "resume_approval",
    "schedule_approval_resume",
    "schedule_approval_timeout",
    "unbind_deployment",
]

_log = logging.getLogger(__name__)

_DEPLOYMENT: "Deployment | None" = None


def bind_deployment(deployment: "Deployment") -> None:
    """Tell restored jobs which deployment they belong to.

    Args:
        deployment: This process's resources.
    """
    global _DEPLOYMENT
    _DEPLOYMENT = deployment


def unbind_deployment() -> None:
    """Forget the bound deployment. For a clean shutdown, and for tests."""
    global _DEPLOYMENT
    _DEPLOYMENT = None


def bound_deployment() -> "Deployment":
    """Read the bound deployment.

    Returns:
        The deployment.

    Raises:
        RuntimeError: When nothing bound one. A job that ran against no deployment
            would fail somewhere further in with a less useful message.
    """
    if _DEPLOYMENT is None:
        raise RuntimeError("no deployment is bound; a scheduled job cannot run")
    return _DEPLOYMENT


async def expire_approval(request_id: str, **_: Any) -> str:
    """Time out a request nobody answered, and let the incident continue.

    A request left pending forever parks its incident forever. Timing out is a decision
    like any other — it is recorded, it is attributable to nobody, and the paused call
    comes back as a refusal rather than as an error.

    Args:
        request_id: The request that has run out of time.
        **_: Ignored. A restored job may carry arguments a later version added.

    Returns:
        A short status for the job log.
    """
    deployment = bound_deployment()
    now = datetime.now(UTC).isoformat()
    expired = {record.request_id for record in deployment.storage.approvals.expire_due(now=now)}
    if request_id not in expired:
        # Answered in the meantime, or already expired by an earlier sweep. Either way
        # there is nothing to do, and doing it twice would resume the same incident twice.
        return "already_decided"
    _log.info("approval request %s timed out with no decision", request_id)
    schedule_approval_resume(request_id)
    return "timed_out"


async def resume_approval(request_id: str, **_: Any) -> str:
    """Continue the incident a decided request was holding.

    Args:
        request_id: The request that was decided, approved, denied or timed out.
        **_: Ignored. A restored job may carry arguments a later version added.

    Returns:
        A short status for the job log.
    """
    return await resume_after_decision(request_id, bound_deployment())


def schedule_approval_timeout(request_id: str, expires_at: str) -> None:
    """Arm one request's expiry.

    Args:
        request_id: The pending request.
        expires_at: When it runs out of time, ISO-8601.
    """
    _schedule(
        expire_approval,
        f"approval.timeout.{request_id}",
        request_id,
        datetime.fromisoformat(expires_at),
    )


def schedule_approval_resume(request_id: str) -> None:
    """Continue a decided request's incident as soon as the loop is free.

    Through the scheduler rather than as a task on whichever event loop recorded the
    decision. A continuation runs for as long as the rest of the incident does, while
    the decision that triggered it is already durable in the store, so a process that
    stops in between should pick the incident back up rather than lose it. A job in the
    persistent store does; a task created inside a request handler does not.

    Args:
        request_id: The decided request.
    """
    _schedule(resume_approval, f"approval.resume.{request_id}", request_id, datetime.now(UTC))


def _schedule(
    job: Callable[..., Awaitable[Any]], job_id: str, request_id: str, when: datetime
) -> None:
    """Queue one approval job, replacing any earlier one for the same request."""
    scheduler_for(bound_deployment().settings.server.workdir).add_job(
        job,
        "date",
        run_date=when,
        kwargs={"request_id": request_id},
        id=job_id,
        replace_existing=True,
    )
