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
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kira.orchestration.runner import Deployment

__all__ = ["bind_deployment", "bound_deployment", "expire_approval", "unbind_deployment"]

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
    return "timed_out"
