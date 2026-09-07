"""Driving one adapter-supplied episode, end to end.

This function knows nothing about any environment. It is the prototype's SREGym episode
driver with every benchmark-shaped decision moved behind the adapter protocol, and with
each of that function's hard-won orderings preserved and named, because every one of
them cost a run to learn.

The episode opens in two steps rather than one. Taking the environment and creating the
session has to finish before the caller is answered, since the answer carries the
session to poll; everything after that is minutes to hours of model calls and runs in
the background. A harness that watches our process rather than our replies is why the
whole of it must still be reachable from one blocking caller.

It can also end in two steps. A write that needs a person parks the incident and hands
the environment to the resume path, which comes back through here rather than through
the runner alone: an episode's answers, its grades and its release are this driver's
business, and the approved call has to go over the transport the rest of the episode
ran on.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from kira.adapters.base import (
    EXTERNAL_KEY,
    EnvironmentAdapter,
    EpisodeContext,
    ExternalFeedback,
    LifecycleRunsElsewhere,
)
from kira.audit.journal import Journal
from kira.contracts.events import EnvironmentEvent
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import IncidentState
from kira.middleware.authority import ApprovalPending
from kira.orchestration.inflight import drop_episode, hold_episode
from kira.orchestration.learning import run_learning
from kira.orchestration.outcomes import resolve_outcome
from kira.orchestration.resume import stored_state
from kira.orchestration.runner import Deployment, resume_after_decision, run_incident
from kira.tools.transport.base import ToolTransport

__all__ = ["EXTERNAL_KEY", "EpisodeResult", "EpisodeStart", "open_episode", "run_episode"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeStart:
    """One episode, taken and bound to a session, before any role has run."""

    session_id: str
    started_at: str
    context: EpisodeContext
    signal: dict[str, Any]
    withheld_record_ids: tuple[str, ...] = ()
    """Corpus records the adapter withheld from this episode, on top of the floor."""


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """How one episode ended, for the caller's log."""

    session_id: str
    submitted: tuple[str, ...] = ()
    graded: bool = False
    paused: bool = False
    failed: bool = False


async def open_episode(adapter: EnvironmentAdapter, deployment: Deployment) -> EpisodeStart:
    """Take one episode from the environment and open the session it will run as.

    Args:
        adapter: The environment adapter, freshly built for this episode.
        deployment: The process's resources.

    Returns:
        The identity of the episode, which is what a caller polls.

    Raises:
        LifecycleRunsElsewhere: When the adapter mounts no episode route. Such an
            adapter is bound for the whole process and its environment is held by
            another process, so an episode opened here would acquire nothing and
            release nothing. Unreachable today, because the only thing that opens an
            episode is a route; the refusal is for whatever calls this next.
        Exception: Whatever the adapter raised. Nothing has started yet, so a failure
            here is a clean refusal rather than an abandoned environment. That includes a
            withheld set the adapter could not compute: an exclusion that fails open is
            worse than an episode that does not run, because it voids the cell silently.
    """
    if adapter.routes() is None:
        raise LifecycleRunsElsewhere(
            f"the {adapter.name} environment mounts no episode route, so its episodes "
            "are opened by whatever holds the environment and this process supplies "
            "the pipeline alone"
        )
    ctx = await adapter.acquire()
    signal = adapter.build_signal(ctx)
    signal[EXTERNAL_KEY] = ctx.external_key
    record = deployment.storage.sessions.create_session(signal)
    adapter.bind_session(record.session_id)
    return EpisodeStart(
        session_id=record.session_id,
        started_at=record.started_at,
        context=ctx,
        signal=signal,
        withheld_record_ids=tuple(adapter.withheld_records(ctx, deployment.storage.index)),
    )


async def run_episode(
    adapter: EnvironmentAdapter,
    deployment: Deployment,
    start: EpisodeStart,
    *,
    submit: bool = True,
) -> EpisodeResult:
    """Run one opened episode to its end and hand the environment back.

    Args:
        adapter: The adapter that opened the episode.
        deployment: The process's resources.
        start: What :func:`open_episode` returned.
        submit: Whether to hand answers over for grading. False runs the pipeline and
            the distilling pass without submitting anything, for a local rehearsal.

    Returns:
        How the episode ended. A paused episode has not ended: the environment is still
        held and the continuation closes it out.
    """
    bound = _bind(deployment, adapter)
    journal = _journal(bound, start.session_id)
    journal.record(
        EnvironmentEvent(
            adapter=adapter.name,
            action="acquired",
            # Written before the pipeline runs, and as JSON like the release event's own
            # detail, because a cell's withheld set is otherwise unreconstructible: the
            # count reaches the second pass's event and the identifiers reach nothing.
            # A crashed episode has still recorded what it was denied.
            detail=json.dumps(
                {
                    "namespace": start.context.namespace or "",
                    "withheld": list(start.withheld_record_ids),
                },
                sort_keys=True,
            ),
        )
    )

    async def finish(request_id: str) -> str:
        return await _continue(adapter, bound, start, request_id, submit=submit)

    # Held before the pipeline can pause rather than after it has. An operator who
    # answers in the same breath as the request still has to find the episode that owns
    # the environment, and the checkpoint the pause writes records this from here.
    hold_episode(start.session_id, adapter, finish)

    state: IncidentState | None = None
    try:
        # Two flags, two recorded costs. A poller tears the environment down the moment
        # the run status goes terminal, so it must stay running through the distilling
        # pass and the submissions; and the driver owns that pass's lifecycle here,
        # because it has to finish while the environment is still up.
        state = await run_incident(
            start.signal,
            start.session_id,
            bound,
            target_stage=start.context.target_stage,
            fire_post_hooks=False,
            finalize_run_status=False,
            withheld_record_ids=start.withheld_record_ids,
        )
    except ApprovalPending:
        # The environment belongs to the resume path. Releasing it now would tear down
        # the cluster the approved write is meant to act on, so the episode stays held
        # and :func:`_continue` finishes it once the decision lands.
        _log.info("episode %s is parked on an approval; not releasing", start.session_id)
        return EpisodeResult(session_id=start.session_id, paused=True)
    except asyncio.CancelledError:
        drop_episode(start.session_id)
        raise
    except Exception as exc:
        drop_episode(start.session_id)
        _fail(deployment, start.session_id, exc)
        journal.record(
            EnvironmentEvent(
                adapter=adapter.name,
                action="failed",
                detail=f"{type(exc).__name__}: {exc}",
                ok=False,
            )
        )
        # Closed like any other ending. A crash still owes the environment a release and
        # the harness an answer: without them the conductor stays parked on a deployed
        # application until its own teardown timeout, and the problem produces no result
        # row at all, so it leaves the denominator instead of counting as a failure.
        closed = await _close(adapter, bound, start, journal, None, submit=submit, crashed=True)
        return replace(closed, failed=True)

    drop_episode(start.session_id)
    return await _close(adapter, bound, start, journal, state, submit=submit)


async def _continue(
    adapter: EnvironmentAdapter,
    deployment: Deployment,
    start: EpisodeStart,
    request_id: str,
    *,
    submit: bool,
) -> str:
    """Carry a parked episode through the decision and out the other side.

    Under the episode's own composition rather than the process's, which is the whole
    reason this lives here. The approved call has to go over the transport the rest of
    the episode ran on — for an evaluation harness that transport carries the correlation
    id its rollback journal is keyed on, so a write sent over any other one lands outside
    the session being graded — and an adapter-supplied tool exists on no other
    composition at all: replayed against the default one it comes back as "not available
    in this configuration", and the operator's approval is honoured on paper and dropped
    in fact.

    Args:
        adapter: The adapter still holding this episode.
        deployment: The episode's composition, as :func:`_bind` made it.
        start: What :func:`open_episode` returned.
        request_id: The request that was decided.
        submit: Whether answers are handed over for grading.

    Returns:
        The continuation's status, for the job log.
    """
    status = await resume_after_decision(request_id, deployment)
    if status == "awaiting_human":
        # A second write wants a second answer. The episode stays held.
        return status
    if drop_episode(start.session_id) is None:
        # Another continuation already took it, and an episode has one ending.
        return status

    # No state to hand over: a continuation reports a status, so :func:`_close` reads
    # the answers back from the store itself. Closed even when the continuation could
    # not run, because an episode that ends here owes the harness an answer and the
    # environment a release either way — without them the problem leaves the run with no
    # result at all rather than a failed one.
    await _close(
        adapter, deployment, start, _journal(deployment, start.session_id), None, submit=submit
    )
    return status


async def _close(
    adapter: EnvironmentAdapter,
    deployment: Deployment,
    start: EpisodeStart,
    journal: Journal,
    state: IncidentState | None,
    *,
    submit: bool,
    crashed: bool = False,
) -> EpisodeResult:
    """Hand the answers over, distil what happened, and give the environment back.

    Args:
        adapter: The adapter holding this episode.
        deployment: The episode's own composition.
        start: What :func:`open_episode` returned.
        journal: The episode's log.
        state: The incident state, or None when the pipeline produced none.
        submit: Whether answers are handed over for grading.
        crashed: Whether the pipeline died rather than finished, in which case the
            terminal run status has already been written and must stand.
    """
    loop_ended_at = _utcnow()
    submitted: list[str] = []
    feedback: ExternalFeedback | None = None
    error: str | None = None
    # A pipeline that produced no state still owes an answer, so it is rebuilt from what
    # was written down rather than skipped. Every encoder handles an absent diagnosis;
    # none of them is reached if nothing is submitted at all.
    state = state if state is not None else _recovered_state(deployment, start)
    try:
        if submit:
            submitted, error = await _submit_plan(adapter, start, state, journal)
            if adapter.holds_environment_for_grading():
                feedback = await adapter.fetch_feedback()
                journal.record(
                    EnvironmentEvent(
                        adapter=adapter.name,
                        action="graded",
                        detail=feedback.source if feedback else "no grades available",
                        ok=feedback is not None,
                    )
                )
        await _distil(deployment, start.session_id, feedback)
    finally:
        # Always, and before the run status is finalised. A slow or crashed distilling
        # pass must never strand a deployed application, and a poller that saw a
        # terminal status before the release would tear the environment down while the
        # release was still in flight.
        await adapter.release(state=state)
        _record_episode(
            journal, adapter, start, submitted, loop_ended_at, feedback, error, deployment
        )
        if not crashed:
            # A crash has already written its terminal status. Re-resolving one from a
            # state rebuilt to answer with would report the run that did not happen.
            _finalize(deployment, start.session_id, state)

    return EpisodeResult(
        session_id=start.session_id,
        submitted=tuple(submitted),
        graded=feedback is not None,
    )


def _recovered_state(deployment: Deployment, start: EpisodeStart) -> IncidentState:
    """Rebuild enough of an incident to answer with, from what was written down.

    A crashed pipeline returns nothing, and a storage failure must never forfeit
    grading, so the fallback is the signal the episode opened on rather than no
    submission at all.
    """
    try:
        record = deployment.storage.sessions.get_session(start.session_id)
    except Exception:
        _log.exception("could not read session %s back to answer with", start.session_id)
        record = None
    if record is not None:
        return stored_state(record, target_stage=start.context.target_stage)
    return IncidentState(
        session_id=start.session_id,
        alert=dict(start.signal),
        target_stage=start.context.target_stage,
        error=None,
    )


def _journal(deployment: Deployment, session_id: str) -> Journal:
    """The episode's own log, which belongs to the environment rather than to a role."""
    return Journal(
        deployment.sink,
        session_id=session_id,
        stage="environment",
        sequence=deployment.sequences.for_session(session_id),
    )


def _bind(deployment: Deployment, adapter: EnvironmentAdapter) -> Deployment:
    """Compose the deployment this episode's roles run under.

    Everything the adapter supplies is bound here, once, so a role receives an
    environment rather than consulting one. A transport of None keeps the deployment's
    configured default, which is what an adapter with nothing to supply wants.

    The adapter itself is bound too, so that a role run under this episode can be
    narrowed by it. It is this instance rather than a fresh one, because an overlay may
    depend on what :meth:`~kira.adapters.base.EnvironmentAdapter.acquire` learned.
    """
    transport = adapter.tool_transport()
    evidence = adapter.evidence_policy()
    changes: dict[str, Any] = {"tool_extras": adapter.extra_tools(), "adapter": adapter}
    if transport is not None:

        def transport_for(session_id: str) -> ToolTransport:
            return transport

        changes["transport_for"] = transport_for
    if evidence is not None:
        changes["evidence"] = evidence
    return replace(deployment, **changes)


async def _submit_plan(
    adapter: EnvironmentAdapter,
    start: EpisodeStart,
    state: IncidentState,
    journal: Journal,
) -> tuple[list[str], str | None]:
    """Hand over each answer in turn, once its stage will take it.

    Nothing is submitted mid-pipeline, and that is the ordering the whole design turns
    on: a harness that grades the first submission per stage would grade an early
    diagnosis rather than the final one, so the answers go over once, at the end, after
    the verification hand-backs have settled.
    """
    submitted: list[str] = []
    for kind in start.context.submission_kinds:
        if not await adapter.await_ready(kind):
            _log.error(
                "submission stage %r never became live; stopping after %d — grading "
                "will be incomplete",
                kind,
                len(submitted),
            )
            return submitted, f"stage {kind!r} never became live"
        record = await adapter.submit(kind, adapter.encode(kind, state, start.context))
        submitted.append(record.kind)
        journal.record(
            EnvironmentEvent(
                adapter=adapter.name,
                action="submitted",
                detail=record.error or record.kind,
                ok=record.error is None,
            )
        )
    return submitted, None


async def _distil(
    deployment: Deployment, session_id: str, feedback: ExternalFeedback | None
) -> None:
    """Run the distilling pass here, bounded, rather than leaving it to the scheduler.

    It has to finish while the environment is still up, because it may re-check the
    post-remediation cluster, and it has to be bounded because the release is waiting
    on it. Its failure is swallowed so it can never block that release, and logged
    loudly because a silent one means no memory record was written for this session and
    a memory arm quietly degrades to no carry-forward — which went unnoticed for a whole
    run once.
    """
    if not deployment.settings.memory.learning_enabled:
        return
    timeout = deployment.settings.memory.learning_timeout_s
    try:
        await asyncio.wait_for(
            run_learning(
                session_id,
                deployment,
                external_feedback=feedback.to_packet() if feedback else None,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        _log.error(
            "distilling session %s exceeded %ds — releasing the environment with no "
            "memory record written for it",
            session_id,
            timeout,
        )
    except Exception:
        _log.exception(
            "distilling session %s failed — no memory record was written for it and it "
            "will not carry forward",
            session_id,
        )


def _fail(deployment: Deployment, session_id: str, exc: BaseException) -> None:
    """Leave a crashed episode in a state its poller can tell is over.

    Without the run status a crashed pipeline stays ``running`` forever and the harness
    blocks until its own outer timeout, which costs the whole cell rather than one run.
    """
    detail = f"{type(exc).__name__}: {exc}"
    _log.exception("episode %s crashed", session_id)
    try:
        sessions = deployment.storage.sessions
        sessions.update_artefacts(session_id, notes=detail)
        sessions.set_state(session_id, LifecycleState.INVALIDATED)
        sessions.set_run_status(session_id, RunStatus.FAILED)
    except Exception:
        _log.exception("could not record the failure of episode %s", session_id)


def _finalize(deployment: Deployment, session_id: str, state: IncidentState | None) -> None:
    """Write the terminal run status, now that the environment has been handed back."""
    status = resolve_outcome(state).run_status if state is not None else None
    try:
        # Never left unresolved. A run that produced nothing decisive still has to stop
        # a poller, so an unresolved outcome finalises as a failure rather than hanging.
        deployment.storage.sessions.set_run_status(session_id, status or RunStatus.FAILED)
    except Exception:
        _log.exception("could not finalise the run status of episode %s", session_id)


def _record_episode(
    journal: Journal,
    adapter: EnvironmentAdapter,
    start: EpisodeStart,
    submitted: list[str],
    loop_ended_at: str,
    feedback: ExternalFeedback | None,
    error: str | None,
    deployment: Deployment,
) -> None:
    """Close the episode's log with what was handed over and when.

    Submitting only at the end collapses the harness's own per-stage submit times, so
    per-stage latency has to come from this side instead: when the last diagnosis run
    finished, and when the pipeline stopped.
    """
    detail = {
        "external_key": start.context.external_key,
        "stages": submitted,
        "rca_completed_at": _last_run_ended(deployment, start.session_id, "rca"),
        "loop_ended_at": loop_ended_at,
        "has_feedback": feedback is not None,
        **({"error": error} if error else {}),
    }
    journal.record(
        EnvironmentEvent(
            adapter=adapter.name,
            action="released",
            detail=json.dumps(detail, sort_keys=True, default=str),
            ok=error is None,
        )
    )


def _last_run_ended(deployment: Deployment, session_id: str, role: str) -> str | None:
    """When the latest run of one role finished, best effort."""
    try:
        runs = deployment.storage.runs.runs_for(session_id)
    except Exception:
        return None
    ended = [run.ended_at for run in runs if run.agent_name == role and run.ended_at]
    return ended[-1] if ended else None


def _utcnow() -> str:
    """The current time in the session store's own format, so the two compare."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
