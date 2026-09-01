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
"""

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from kira.adapters.base import EnvironmentAdapter, EpisodeContext, ExternalFeedback
from kira.audit.journal import Journal
from kira.contracts.events import EnvironmentEvent
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import IncidentState
from kira.middleware.authority import ApprovalPending
from kira.orchestration.learning import run_learning
from kira.orchestration.outcomes import resolve_outcome
from kira.orchestration.runner import Deployment, run_incident
from kira.tools.transport.base import ToolTransport

__all__ = ["EpisodeResult", "EpisodeStart", "open_episode", "run_episode"]

_log = logging.getLogger(__name__)

EXTERNAL_KEY = "_external_key"
"""Where the adapter's own episode identifier is stored on the signal.

Underscore-prefixed, and that prefix is the whole mechanism: for an evaluation harness
this value is the ground-truth fault label, and every prompt builder drops keys that
start with one. It exists so a session can be joined to the grade it earned, which
otherwise falls back to a time window — and did, mis-mapping five of six sessions in one
measured case.
"""


@dataclass(frozen=True, slots=True)
class EpisodeStart:
    """One episode, taken and bound to a session, before any role has run."""

    session_id: str
    started_at: str
    context: EpisodeContext
    signal: dict[str, Any]


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
        Exception: Whatever the adapter raised. Nothing has started yet, so a failure
            here is a clean refusal rather than an abandoned environment.
    """
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
        How the episode ended.
    """
    bound = _bind(deployment, adapter)
    journal = Journal(deployment.sink, session_id=start.session_id, stage="environment")
    journal.record(
        EnvironmentEvent(
            adapter=adapter.name, action="acquired", detail=start.context.namespace or ""
        )
    )

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
        )
    except ApprovalPending:
        # The environment belongs to the resume path. Releasing it now would tear down
        # the cluster the approved write is meant to act on.
        _log.info("episode %s is parked on an approval; not releasing", start.session_id)
        return EpisodeResult(session_id=start.session_id, paused=True)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _fail(deployment, start.session_id, exc)
        journal.record(
            EnvironmentEvent(
                adapter=adapter.name,
                action="failed",
                detail=f"{type(exc).__name__}: {exc}",
                ok=False,
            )
        )
        return EpisodeResult(session_id=start.session_id, failed=True)

    loop_ended_at = _utcnow()
    submitted: list[str] = []
    feedback: ExternalFeedback | None = None
    error: str | None = None
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
        await _distil(bound, start.session_id, feedback)
    finally:
        # Always, and before the run status is finalised. A slow or crashed distilling
        # pass must never strand a deployed application, and a poller that saw a
        # terminal status before the release would tear the environment down while the
        # release was still in flight.
        await adapter.release(state=state)
        _record_episode(journal, adapter, start, submitted, loop_ended_at, feedback, error, bound)
        _finalize(deployment, start.session_id, state)

    return EpisodeResult(
        session_id=start.session_id,
        submitted=tuple(submitted),
        graded=feedback is not None,
    )


def _bind(deployment: Deployment, adapter: EnvironmentAdapter) -> Deployment:
    """Compose the deployment this episode's roles run under.

    Everything the adapter supplies is bound here, once, so a role receives an
    environment rather than consulting one. A transport of None keeps the deployment's
    configured default, which is what an adapter with nothing to supply wants.
    """
    transport = adapter.tool_transport()
    evidence = adapter.evidence_policy()
    changes: dict[str, Any] = {"tool_extras": adapter.extra_tools()}
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
