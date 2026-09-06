"""What a request handler is given, and where it comes from.

One object, built once by the composition root and read off the application state, so a
handler reaches for nothing global. The prototype's routers called
``get_session_store()`` and imported the run registry as a module, which is why none of
them could be exercised without the whole process standing up around them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from kira.orchestration.control import Claim, RunRegistry
from kira.orchestration.runner import Deployment

__all__ = ["Service", "ServiceDep", "admit", "service"]


@dataclass(frozen=True, slots=True)
class Service:
    """Everything the HTTP surface shares across requests."""

    deployment: Deployment
    """The composition root: stores, model, transport factory and authority."""
    schedule_resume: Callable[[str], None]
    """Continues the incident a decided approval request was holding.

    Injected rather than reached for, for the same reason
    :class:`~kira.middleware.human_approval.HumanApproval` takes its timeout scheduler
    that way: the scheduler is a process-wide resource, and a route that reached for it
    directly could not be exercised without one. Required rather than defaulted, because
    a route that silently continued nothing would look exactly like one that worked.
    """
    runs: RunRegistry = field(default_factory=RunRegistry)
    """The incidents running in this process right now, and the tasks carrying them."""


def service(request: Request) -> Service:
    """Read the service off the application.

    Args:
        request: The incoming request.

    Returns:
        The process's service object.
    """
    return cast(Service, request.app.state.service)


ServiceDep = Annotated[Service, Depends(service)]
"""The dependency every handler declares. Annotated so the default stays a type."""


def admit(service: Service, *, fingerprint: str | None = None) -> Claim:
    """Take this process's permission to open one more incident.

    Both rules an intake obeys are process-wide rather than one route's. A process on
    its way out must not open an incident it cannot finish; and a private model endpoint
    serves a small number of concurrent sequences, so oversubscribing one degrades every
    run in flight rather than queueing. The adapter's episode route enforced neither, so
    a deploy or an eviction during a campaign accepted a problem into a process that had
    already begun shutting down, and the cell was recorded as an agent failure instead of
    being retried against a live one.

    Args:
        service: The process's resources.
        fingerprint: What identifies the alert, for an intake that deduplicates reposts.

    Returns:
        The claim to release once the run it starts is registered, or once it is certain
        none will be.

    Raises:
        HTTPException: 503 while draining, and 429 when this process is already carrying
            as many incidents as it should. Both name a Retry-After, because both are
            conditions the caller can come back from.
    """
    if service.runs.draining:
        # Refused before the deduplication a claim also does, because a repost during
        # shutdown is not a request to join a run this process is about to abandon.
        raise HTTPException(
            status_code=503,
            detail="shutting down; not accepting new incidents",
            headers={"Retry-After": "30"},
        )
    limit = service.deployment.settings.server.max_concurrent_incidents
    claim = service.runs.claim(fingerprint, limit)
    if claim is None:
        # Refused rather than queued. The queue would be invisible to the caller and
        # every run already in flight would slow down to make room for it; a refusal an
        # alerting source can retry is the honest answer.
        raise HTTPException(
            status_code=429,
            detail=f"already handling {limit} incidents",
            headers={"Retry-After": "60"},
        )
    return claim
