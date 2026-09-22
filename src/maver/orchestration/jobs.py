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

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from maver.adapters.registry import get_adapter
from maver.audit.journal import Journal
from maver.contracts.events import EnvironmentEvent
from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.memory.ports import ApprovalRecord
from maver.orchestration.control import RunRegistry
from maver.orchestration.inflight import clear_held_episodes, held_episode
from maver.orchestration.resume import ApprovalCheckpoint, approves_on_timeout, load_checkpoint
from maver.orchestration.runner import Deployment, resume_after_decision
from maver.orchestration.scheduler import scheduler_for

__all__ = [
    "bind_deployment",
    "bound_deployment",
    "expire_approval",
    "record_cancellation",
    "resume_approval",
    "schedule_approval_resume",
    "schedule_approval_timeout",
    "unbind_deployment",
]

_log = logging.getLogger(__name__)

_DEPLOYMENT: "Deployment | None" = None
_RUNS = RunRegistry()

_UNFINISHED = ("running", "awaiting_human")


def bind_deployment(deployment: "Deployment", *, runs: RunRegistry | None = None) -> None:
    """Tell restored jobs which deployment they belong to, and who tracks its runs.

    Args:
        deployment: This process's resources.
        runs: Where a continuation this module starts is recorded, so that the drain
            waits for it like any other incident. A registry of its own when the caller
            has none, so the continuation path has one shape rather than two.
    """
    global _DEPLOYMENT, _RUNS
    _DEPLOYMENT = deployment
    _RUNS = runs if runs is not None else RunRegistry()


def unbind_deployment() -> None:
    """Forget the bound deployment, and every episode that was running under it.

    An episode parked on a person closes over the deployment it opened under, so one
    that outlives the unbinding is a continuation nothing can serve.
    """
    global _DEPLOYMENT, _RUNS
    _DEPLOYMENT = None
    _RUNS = RunRegistry()
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


def _approved_on_expiry(deployment: "Deployment", record: ApprovalRecord, now: str) -> bool:
    """Settle one overdue request in favour of the rule that raised it, if it has one.

    Both ways an expiry is noticed come through here — the job armed at the pause and
    the sweep that catches a request whose job was never armed — because a rule meaning
    one thing when the job fires and another when the sweep finds it is not a policy.

    Args:
        deployment: This process's resources.
        record: A request that may have run out of time.
        now: The current timestamp.

    Returns:
        True when the request was approved, so a caller can tell it from a timeout.
    """
    decided_by = approves_on_timeout(record, deployment.settings.authority, now=now)
    if decided_by is None or not deployment.storage.approvals.decide(
        record.request_id,
        status="approved",
        decided_by=decided_by,
        reason="no operator answered, and the rule that raised it approves on expiry",
    ):
        return False
    _log.info("approval request %s was approved on expiry by %s", record.request_id, decided_by)
    return True


async def expire_approval(request_id: str, **_: Any) -> str:
    """Time out a request nobody answered, and let the incident continue.

    A request left pending forever parks its incident forever. Timing out is a decision
    like any other — it is recorded, it is attributable to nobody, and the paused call
    comes back as a refusal rather than as an error.

    Unless the rule that raised it said otherwise. An operator who names the narrow
    class of actions that are safe unattended is making the whole safety property work
    for them, and denying there is the safe answer rather than the configured one.

    Args:
        request_id: The request that has run out of time.
        **_: Ignored. A restored job may carry arguments a later version added.

    Returns:
        A short status for the job log.
    """
    deployment = bound_deployment()
    approvals = deployment.storage.approvals
    now = datetime.now(UTC).isoformat()
    pending = approvals.get_request(request_id)
    # Asked before the expiry, because a decision is only recorded on a request that is
    # still pending: expiring first would leave the rule nothing to decide.
    if pending is not None and _approved_on_expiry(deployment, pending, now):
        schedule_approval_resume(request_id)
        return "approved_on_timeout"
    expired = {record.request_id for record in approvals.expire_due(now=now)}
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

    session = deployment.storage.sessions.get_session(checkpoint.session_id)
    if session is None or session.run_status == RunStatus.NEEDS_RECONCILIATION:
        # A restored scheduler job can race startup recovery. In particular, an
        # adapter's abandon path must not close a run already parked for review.
        return "already_resumed"

    held = held_episode(checkpoint.session_id)
    if (
        deployment.settings.server.resume_interrupted_incidents
        and session.run_status == RunStatus.RESUMING
        and checkpoint.session_id not in _RUNS.live()
        and held is None
    ):
        # A restored scheduler job can fire before startup recovery scans this row.
        # RESUMING means the previous process may already have dispatched the write.
        deployment.storage.sessions.set_run_status(
            checkpoint.session_id,
            RunStatus.NEEDS_RECONCILIATION,
            current_stage=session.current_stage,
            checkpoint=session.checkpoint,
            awaiting_request_id=request_id,
        )
        return "needs_reconciliation"

    if held is not None:
        return await _resume_tracked(
            checkpoint.session_id, request_id, deployment, checkpoint, held.finish
        )
    if checkpoint.adapter:
        return await _abandon(checkpoint.session_id, checkpoint.adapter, request_id, deployment)
    return await _resume_tracked(checkpoint.session_id, request_id, deployment, checkpoint)


async def _resume_tracked(
    session_id: str,
    request_id: str,
    deployment: Deployment,
    approval_checkpoint: ApprovalCheckpoint,
    continuation: Callable[[str], Awaitable[str]] | None = None,
) -> str:
    """Continue a parked incident as a run this process can see it is carrying.

    A continuation is the rest of an incident — minutes to hours of model calls against
    a cluster the approved write has already changed — and it ran as a scheduler job in
    neither the run registry nor the task set. So the drain returned at once and the
    scheduler's shutdown cancelled the coroutine. Registering it makes the drain wait;
    when opt-in recovery is enabled, cancellation after the approved call starts leaves
    the incident for explicit reconciliation because the write may have happened.
    """
    # No await between these checks and registration: two decided-request jobs in
    # this process must not both replay the same approved call. A drain also keeps
    # the durable awaiting_human row for the next process's approval sweep.
    draining_at_entry = _RUNS.draining
    if draining_at_entry:
        return "draining"
    if session_id in _RUNS.live():
        return "already_resuming"
    _RUNS.register(session_id)
    task = asyncio.current_task()
    if task is not None:
        _RUNS.attach(session_id, task)
    try:
        if continuation is not None:
            return await continuation(request_id)
        return await resume_after_decision(request_id, deployment)
    except asyncio.CancelledError:
        _log.info("the continuation of session %s was cancelled", session_id)
        session = deployment.storage.sessions.get_session(session_id)
        if (
            deployment.settings.server.resume_interrupted_incidents
            and _RUNS.draining
            and session is not None
            and session.run_status in (RunStatus.RESUMING, RunStatus.RUNNING)
        ):
            # The approved call may have crossed the transport boundary already.
            # Keep its original checkpoint for an operator to inspect or replan from.
            deployment.storage.sessions.set_run_status(
                session_id,
                RunStatus.NEEDS_RECONCILIATION,
                current_stage=session.current_stage,
                checkpoint=session.checkpoint or approval_checkpoint.model_dump(mode="json"),
                awaiting_request_id=request_id,
            )
        else:
            record_cancellation(deployment, session_id)
        raise
    finally:
        _RUNS.unregister(session_id)


def record_cancellation(deployment: Deployment, session_id: str) -> None:
    """Leave a cancelled incident in a state a reader can tell is over.

    Written inline rather than through a thread, because a coroutine being cancelled
    cannot reliably await anything: the writes are the store's own synchronous calls.

    The lifecycle state is not touched. A cancellation disproves nothing about the claim
    the incident was making, and the offline pass distils completed sessions only, so a
    pending record stays pending and out of the corpus.

    Args:
        deployment: This process's resources.
        session_id: The incident that was stopped.
    """
    try:
        sessions = deployment.storage.sessions
        sessions.update_artefacts(session_id, notes="cancelled before it finished")
        sessions.set_run_status(session_id, RunStatus.CANCELLED)
        for run in reversed(deployment.storage.runs.runs_for(session_id)):
            if run.status in _UNFINISHED:
                deployment.storage.runs.finish_run(
                    run.agent_run_id, status="cancelled", error="cancelled before it finished"
                )
                break
    except Exception:
        _log.exception("could not record the cancellation of session %s", session_id)


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
    Journal(
        deployment.sink,
        session_id=session_id,
        stage="environment",
        sequence=deployment.sequences.for_session(session_id),
    ).record(EnvironmentEvent(adapter=adapter_name, action="failed", detail=detail, ok=False))
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


async def sweep_decided_approvals(**_: Any) -> str:
    """Continue any incident whose request has been settled, however it was settled.

    Three ways to decide were offered, and only one of them worked. The API route
    schedules the continuation itself, but the CLI's default path writes to the store
    directly — deliberately, because that is what works from inside a pod with no port
    open — and a webhook could do the same. Those decisions were durable, correct, and
    inert: the operator saw the approval recorded, and the incident stayed parked with
    nothing left to wake it.

    Expiry is settled here too, rather than only by the job armed when the request was
    raised. Arming that job can fail — the pause deliberately survives it, because losing
    the write would be worse than a request with no expiry — and nothing else ever looked
    at a pending request again. The deadline is on the record, so the sweep reads it, and
    the timeout becomes a property of what is stored rather than of a job having been
    successfully scheduled.

    So the store is the source of truth for a decision, and this is what notices one. The
    setting that says how often has existed since the beginning, describing exactly this.

    Args:
        **_: Ignored. A restored job may carry arguments a later version added.

    Returns:
        A short status for the job log.
    """
    return await asyncio.to_thread(_sweep, bound_deployment())


def _sweep(deployment: Deployment) -> str:
    """The sweep's store work, off the event loop and off the approval history.

    Every store call in this process goes to a thread, and this one ran inline on a
    repeating schedule: three unbounded queries and a session read per historical
    record, once every poll interval, growing for as long as the store is kept. Nothing
    prunes decided approvals, so at two thousand of them over a database on another host
    it held the loop that serves every incident's model calls, every tool result and the
    readiness probe for seconds at a time, and never came back down.

    Driven from the sessions parked right now rather than from every request ever
    answered, because that set is bounded by the incidents in flight. Each parked session
    names the one request it is waiting on, so the question is a lookup rather than a
    scan.
    """
    approvals = deployment.storage.approvals
    now = datetime.now(UTC).isoformat()
    awaiting = deployment.storage.sessions.awaiting_sessions()
    for session in awaiting:
        if session.awaiting_request_id is None:
            continue
        pending = approvals.get_request(session.awaiting_request_id)
        if pending is not None and pending.status == "pending":
            _approved_on_expiry(deployment, pending, now)
    for record in approvals.expire_due(now=now):
        _log.info(
            "approval %s passed its deadline with no decision and no armed timeout",
            record.request_id,
        )

    resumed = 0
    for session in awaiting:
        request_id = session.awaiting_request_id
        if request_id is None:
            continue
        decided = approvals.get_request(request_id)
        if decided is None or decided.status == "pending":
            continue
        _log.info(
            "approval %s was decided elsewhere; continuing incident %s",
            request_id,
            session.session_id,
        )
        schedule_approval_resume(request_id)
        resumed += 1
    return f"resumed {resumed}" if resumed else "nothing waiting"


def schedule_approval_sweep(interval_s: int) -> None:
    """Run the decided-elsewhere sweep on a repeating schedule.

    Args:
        interval_s: Seconds between sweeps.
    """
    scheduler_for(bound_deployment().settings.server.workdir).add_job(
        sweep_decided_approvals,
        "interval",
        seconds=max(1, interval_s),
        id="approval.sweep",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
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
