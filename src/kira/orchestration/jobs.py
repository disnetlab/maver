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

from kira.adapters.registry import get_adapter
from kira.audit.journal import Journal
from kira.contracts.events import EnvironmentEvent
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.orchestration.inflight import clear_held_episodes, held_episode
from kira.orchestration.resume import load_checkpoint
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
    """Forget the bound deployment, and every episode that was running under it.

    An episode parked on a person closes over the deployment it opened under, so one
    that outlives the unbinding is a continuation nothing can serve.
    """
    global _DEPLOYMENT
    _DEPLOYMENT = None
    clear_held_episodes()


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
    """Continue the incident a decided request was holding, under whatever still holds it.

    An incident an environment adapter opened is continued by that adapter's own episode,
    because the approved call has to go over the transport the episode ran on and because
    the episode still owes its harness an answer and a released environment. A plain
    incident is continued by the runner alone. An episode this process no longer holds is
    continued by nobody, on purpose.

    Args:
        request_id: The request that was decided, approved, denied or timed out.
        **_: Ignored. A restored job may carry arguments a later version added.

    Returns:
        A short status for the job log.
    """
    deployment = bound_deployment()
    loaded = load_checkpoint(deployment.storage.sessions, deployment.storage.approvals, request_id)
    if isinstance(loaded, str):
        return loaded
    checkpoint = loaded[0]

    held = held_episode(checkpoint.session_id)
    if held is not None:
        return await held.finish(request_id)
    if checkpoint.adapter:
        return await _abandon(checkpoint.session_id, checkpoint.adapter, request_id, deployment)
    return await resume_after_decision(request_id, deployment)


async def _abandon(
    session_id: str, adapter_name: str, request_id: str, deployment: Deployment
) -> str:
    """End an episode whose live binding this process no longer has.

    Reached when a decision outlives the process that parked on it. The checkpoint names
    the environment, and a name is not an adapter: the instance that opened the episode
    held the transport its tools speak and the correlation id its harness keys its own
    state on, and one built fresh from the registry has acquired none of that. Continuing
    anyway would replay the write an operator approved over the process's default
    transport, outside the session being graded, and would lose every tool the adapter
    itself supplied — a run that looks graded and was not, which is worse than a run that
    is plainly lost.

    So the episode is ended instead, in the two places that matter: the run is marked
    failed, which is what stops a driver polling it and lets a harness tear the
    environment down, and the environment is handed back so it is not left waiting on an
    answer that is not coming.
    """
    _log.error(
        "incident %s parked on approval %s under the %s environment, whose episode this "
        "process no longer holds; it cannot be continued under the composition it ran "
        "under, so it is being ended instead. A pending approval does not survive a "
        "restart of this service while an environment adapter is serving the incident.",
        session_id,
        request_id,
        adapter_name,
    )
    detail = f"the {adapter_name} episode was lost before approval {request_id} was answered"
    sessions = deployment.storage.sessions
    try:
        sessions.update_artefacts(session_id, notes=detail)
        sessions.set_state(session_id, LifecycleState.INVALIDATED)
        sessions.set_run_status(session_id, RunStatus.FAILED)
    except Exception:
        _log.exception("could not record the abandonment of incident %s", session_id)
    Journal(deployment.sink, session_id=session_id, stage="environment").record(
        EnvironmentEvent(adapter=adapter_name, action="failed", detail=detail, ok=False)
    )
    await _release_abandoned(session_id, adapter_name, deployment)
    return "abandoned"


async def _release_abandoned(session_id: str, adapter_name: str, deployment: Deployment) -> None:
    """Hand an abandoned episode's environment back, best effort.

    ``release`` is the one call on the adapter protocol a fresh instance can make sense
    of: it is documented idempotent, it must not raise, and it is keyed on the session
    just bound to it. Everything else needs what ``acquire`` produced, which is precisely
    what was lost. A failure here is logged and nothing more, because the alternative to
    trying is a stranded application.
    """
    try:
        adapter = get_adapter(adapter_name)(deployment.settings)
        adapter.bind_session(session_id)
        await adapter.release(state=None)
    except Exception:
        _log.exception(
            "could not release the environment of abandoned incident %s; it may have to "
            "be torn down by hand",
            session_id,
        )


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
