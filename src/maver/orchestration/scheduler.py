"""The one scheduler, and why its jobs outlive the process.

Distillation is deliberately delayed — minutes, so a late outcome is visible before an
incident is turned into a record — and an approval that nobody answers has to expire on
its own. Both are therefore work that exists between processes, and a job held only in
memory is a memory record or an approval quietly lost to a restart. The job store is
SQLite even under a PostgreSQL deployment, which is a simplification rather than an
oversight: the jobs are few, short-lived and local to one process, and a second database
to operate would buy nothing.

Jobs are restored by dotted import path, which is what forces every job function to be a
module-level one. See :mod:`maver.orchestration.jobs`.
"""

import logging
from pathlib import Path
from typing import Any

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler

__all__ = ["scheduler_for", "shutdown_scheduler"]

_log = logging.getLogger(__name__)

_SCHEDULER: AsyncIOScheduler | None = None

MISFIRE_GRACE_SECONDS = 86400
"""How late a job may still run. A day, because the alternative is dropping it.

A restart that outlasts the grace period discards the job silently, and the work these
jobs do — distilling an incident, expiring an approval — is worth doing late.
"""


def scheduler_for(workdir: Path) -> AsyncIOScheduler:
    """Return the running scheduler, starting it the first time.

    Args:
        workdir: The deployment's workspace. The job database lives beneath it, so two
            deployments on one machine do not share a queue.

    Returns:
        The scheduler.
    """
    global _SCHEDULER
    if _SCHEDULER is not None:
        return _SCHEDULER

    directory = workdir / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    store = SQLAlchemyJobStore(url=f"sqlite:///{directory / 'scheduler.sqlite'}")
    scheduler = AsyncIOScheduler(
        jobstores={"default": store},
        job_defaults={
            "misfire_grace_time": MISFIRE_GRACE_SECONDS,
            # One run, not one per missed firing: a job that missed six windows during a
            # restart should distil the incident once.
            "coalesce": True,
        },
    )
    scheduler.start()
    _SCHEDULER = scheduler
    _log.info("scheduler started with a persistent job store at %s", directory)
    return scheduler


def shutdown_scheduler(*, wait: bool = False) -> None:
    """Stop the scheduler if one is running.

    Args:
        wait: Whether to block until running jobs finish.
    """
    global _SCHEDULER
    if _SCHEDULER is None:
        return
    scheduler: Any = _SCHEDULER
    _SCHEDULER = None
    scheduler.shutdown(wait=wait)
