"""Liveness and readiness, which are not the same question.

``/healthz`` answers "is this process alive". It performs no I/O and holds no lock, so
it keeps answering while the model endpoint, the database and the cluster are all down.
Two requirements agree on that: a Kubernetes liveness probe that depends on a downstream
service turns a downstream outage into a restart loop, and the evaluation driver polls
this endpoint for thirty seconds before every episode and fails the episode if it does
not answer. Anyone tempted to "improve" it into a dependency check breaks both.

``/readyz`` answers "can this process take an incident right now", which is where the
dependency checks belong. A check is required when its failure means an incident
accepted now would certainly fail, and advisory when the failure is expected and
self-clearing.
"""

import asyncio
import os
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from maver import __version__
from maver.api.deps import Service, ServiceDep

__all__ = ["router"]

router = APIRouter()

_REVISION = os.environ.get("MAVER_REVISION", "unknown")
"""The commit this image was built from.

Threaded in from the image's ``org.opencontainers.image.revision`` label, and reported
so a deploy can assert that the container answering is the one that was just built. The
prototype had no way to tell a restarted service from a stale one, and a deploy that
missed the evaluation container graded old code without saying so.
"""


@dataclass(frozen=True, slots=True)
class Check:
    """One readiness probe's verdict."""

    name: str
    ok: bool
    required: bool
    detail: str = ""


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Report that the process is alive.

    Deliberately dependency-free, and deliberately still carrying ``status``: the
    evaluation driver accepts any 200 JSON, so adding keys is compatible and removing
    that one is not.

    Returns:
        The liveness record.
    """
    return {"status": "ok", "version": __version__, "revision": _REVISION}


@router.get("/readyz")
async def readyz(service: ServiceDep) -> JSONResponse:
    """Report whether this process can take an incident.

    Args:
        service: The process's resources.

    Returns:
        The same body either way, with 503 when a required check failed. The shape does
        not change with the verdict, so a reader parses one thing.
    """
    checks = [
        _startup(service),
        _draining(service),
        await asyncio.to_thread(_store, service),
        _queue(service),
    ]
    ready = all(check.ok for check in checks if check.required)
    body: dict[str, Any] = {
        "status": "ready" if ready else "not ready",
        "checks": [asdict(check) for check in checks],
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


def _startup(service: Service) -> Check:
    """Report what this process resolved at startup."""
    settings = service.deployment.settings
    return Check(
        name="startup",
        ok=True,
        required=True,
        detail=f"profile={service.deployment.profile.name} authority={settings.authority.mode}",
    )


def _draining(service: Service) -> Check:
    """Report a process on its way out as not ready.

    Required, because this is what removes the pod from a Service's endpoint list and
    stops requests arriving at something that will not finish them. ``/healthz`` stays
    200 throughout, so the kubelet does not restart the pod part-way through its drain.
    """
    draining = service.runs.draining
    return Check(
        name="draining",
        ok=not draining,
        required=True,
        detail="draining" if draining else "accepting",
    )


def _store(service: Service) -> Check:
    """Read from the store, because a process that cannot persist cannot be given work.

    No session record means no audit trail and no resume, so this is required rather
    than advisory: an incident accepted now would certainly be lost.
    """
    try:
        count = service.deployment.storage.index.count()
    except Exception as exc:
        # A probe that throws is a failed probe. The endpoint's job is to report.
        return Check(name="store", ok=False, required=True, detail=f"{type(exc).__name__}: {exc}")
    return Check(name="store", ok=True, required=True, detail=f"{count} records")


def _queue(service: Service) -> Check:
    """A configured queue intake must still be consuming."""
    if not service.deployment.settings.intake.kafka.enabled:
        return Check(name="queue", ok=True, required=False, detail="disabled")
    queue = service.queue.consumer
    ok = queue is not None and queue.task is not None and not queue.task.done()
    detail = (
        str(queue.error)
        if queue is not None and queue.error
        else "consumer stopped"
        if not ok
        else "consuming"
    )
    return Check(name="queue", ok=ok, required=True, detail=detail)
