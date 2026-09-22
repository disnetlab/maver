"""What happens after an incident closes.

One registry, matched on the outcome. The stage that distils an incident into a record
runs from here rather than from inside the pipeline, because the published description
of the system says it runs offline, outside the incident path — and because a
distillation taken the instant the pipeline stops would miss an outcome that arrives
seconds later.

A scheduling failure is logged and swallowed. The incident has already finished and its
result is already persisted; failing it now would turn a queue problem into a lost run.
"""

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Any

from maver.memory.ports import LearningIntent, SessionStore
from maver.orchestration.outcomes import Outcome
from maver.orchestration.scheduler import scheduler_for

__all__ = ["PostIncidentHook", "clear_hooks", "fire_hooks", "register_hook", "registered_hooks"]

_log = logging.getLogger(__name__)

_HOOKS: dict[str, "PostIncidentHook"] = {}


@dataclass(frozen=True, slots=True)
class PostIncidentHook:
    """Work to schedule once an incident has reached one of the outcomes it names."""

    name: str
    job_fn: Callable[..., Awaitable[Any]]
    """Must be a module-level function. The scheduler restores a job by dotted import
    path, so a closure or a lambda is a job that cannot survive a restart.

    Its return value is a short status for the scheduler's log, not a result anything
    acts on: by the time a job runs, the incident that scheduled it is long closed.
    """
    on_outcomes: frozenset[Outcome]
    delay_seconds: int = 0
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def register_hook(hook: PostIncidentHook) -> None:
    """Register one hook.

    Args:
        hook: The hook.

    Raises:
        ValueError: When the name is already taken. A silent overwrite would leave a
            deployment running work nobody registered.
    """
    if hook.name in _HOOKS:
        raise ValueError(f"post-incident hook {hook.name!r} is already registered")
    _HOOKS[hook.name] = hook


def registered_hooks() -> Mapping[str, PostIncidentHook]:
    """Every hook registered so far.

    Returns:
        A read-only view of the registry.
    """
    return dict(_HOOKS)


def clear_hooks() -> None:
    """Empty the registry. For tests, and for a composition root rebuilding it."""
    _HOOKS.clear()


async def fire_hooks(
    outcome: Outcome,
    session_id: str,
    *,
    workdir: Path,
    sessions: SessionStore | None = None,
    built_in_learning_enabled: bool = True,
    **extra: Any,
) -> None:
    """Schedule every hook that matches this incident's outcome.

    Args:
        outcome: How the incident ended.
        session_id: The incident.
        workdir: The deployment's workspace, which locates the job store.
        sessions: Session store for durable built-in Learning intents.
        built_in_learning_enabled: Whether the selected pipeline bound Learning.
        **extra: Passed to the job alongside its own configured arguments. This is how
            an adapter carries an external grade through the registry instead of
            calling the job directly and bypassing it.
    """
    for hook in _HOOKS.values():
        if outcome not in hook.on_outcomes:
            continue
        due_at = datetime.now(UTC) + timedelta(seconds=hook.delay_seconds)
        learning = import_module("maver.orchestration.learning")
        if hook.job_fn is learning.distil_incident and not built_in_learning_enabled:
            continue
        if hook.job_fn is learning.distil_incident and sessions is not None:
            feedback = {**dict(hook.kwargs), **extra}.get("external_feedback")
            try:
                sessions.put_learning_intent(
                    LearningIntent(
                        session_id=session_id,
                        due_at=due_at.isoformat(),
                        external_feedback=feedback if isinstance(feedback, dict) else None,
                    )
                )
            except Exception:
                _log.exception("could not persist Learning intent for %s", session_id)
                # Keep the original one-shot job as a best-effort fallback. Never
                # claim that an intent was durable when its transaction failed.
                try:
                    scheduler_for(workdir).add_job(
                        hook.job_fn,
                        "date",
                        run_date=due_at,
                        kwargs={"session_id": session_id, **dict(hook.kwargs), **extra},
                        id=f"{hook.name}:{session_id}",
                        replace_existing=True,
                    )
                except Exception:
                    _log.exception("could not schedule fallback Learning job for %s", session_id)
                continue
            # The persisted intent is the authority; a scheduler outage cannot erase it.
            try:
                learning.schedule_learning_intent(workdir, sessions.get_learning_intent(session_id))
            except Exception:
                _log.exception("could not schedule post-incident hook %s", hook.name)
            continue
        try:
            scheduler_for(workdir).add_job(
                hook.job_fn,
                "date",
                run_date=due_at,
                kwargs={"session_id": session_id, **dict(hook.kwargs), **extra},
                id=f"{hook.name}:{session_id}",
                replace_existing=True,
            )
        except Exception:
            # The incident is over and its result is already stored. A queue problem
            # must not become a lost run.
            _log.exception("could not schedule post-incident hook %s", hook.name)
