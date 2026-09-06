"""What a request handler is given, and where it comes from.

One object, built once by the composition root and read off the application state, so a
handler reaches for nothing global. The prototype's routers called
``get_session_store()`` and imported the run registry as a module, which is why none of
them could be exercised without the whole process standing up around them.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, cast

from fastapi import Depends, Request

from kira.orchestration.control import RunRegistry
from kira.orchestration.runner import Deployment

__all__ = ["Service", "ServiceDep", "service"]


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
