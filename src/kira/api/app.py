"""The HTTP application, and the composition root that stands behind it.

One deployment is built here, once, and handed to every request through the application
state. Nothing under ``api/`` reaches for a store, a model client or a transport: the
prototype's routers each imported their own singletons, which is why a route could not be
exercised without the whole process standing up around it, and why two of them ended up
reading configuration the composition root had already resolved.

The pieces are assembled before the application is returned rather than inside the
lifespan, because a handler must never see half a process. The lifespan owns what is
process-scoped rather than application-scoped: telling scheduled jobs which deployment
they belong to, and letting go of the stores on the way out.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from kira.adapters.registry import (
    get_adapter,
    legacy_selection,
    load_entry_point_adapters,
)
from kira.api.deps import Service
from kira.api.health import router as health_router
from kira.api.hitl import router as hitl_router
from kira.api.incidents import router as incidents_router
from kira.audit.store_sink import StoreSink
from kira.llm.client import LLMClient
from kira.memory.factory import build_storage
from kira.memory.ports import ApprovalStore
from kira.middleware.authority import Authority, AutoApprove, DenyAll
from kira.middleware.human_approval import HumanApproval
from kira.observability.tracing import configure_tracing
from kira.orchestration.bootstrap import register_post_incident_hooks
from kira.orchestration.control import RunRegistry
from kira.orchestration.jobs import (
    bind_deployment,
    schedule_approval_resume,
    schedule_approval_timeout,
    unbind_deployment,
)
from kira.orchestration.runner import Deployment
from kira.orchestration.scheduler import shutdown_scheduler
from kira.roles.registry import validate_roles
from kira.settings.profiles import load_profile
from kira.settings.root import Settings, load_settings
from kira.tools.transport.base import ToolTransport
from kira.tools.transport.direct import direct_transport

__all__ = ["API_PREFIX", "build_deployment", "create_app", "enabled_adapters"]

_log = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
"""Where the incident and decision routes live. The health routes sit above it, because
a Kubernetes probe should not have to know the API's version."""


def build_deployment(settings: Settings) -> Deployment:
    """Assemble everything a run needs from one settings object.

    Args:
        settings: The resolved configuration.

    Returns:
        The process's deployment.

    Raises:
        FileNotFoundError: When the configured model profile does not exist.
        RoleRegistryError: When a role cannot terminate or cannot be prompted. Checked
            here, at startup, because the alternative is a run that burns its whole turn
            budget and reports what looks like a model failure.
        ValueError: When the configured storage backend is not installed.
    """
    profile = load_profile(settings.model.profile, settings.model.profile_dir)
    validate_roles()
    storage = build_storage(settings)

    def transport_for(session_id: str) -> ToolTransport:
        return direct_transport(settings.transport)

    # What this process is actually running, in one line an operator reads in the pod's
    # log. A campaign is only reconstructible if that is recorded beside its results.
    _log.info(
        "serving profile %s under authority %s, workspace %s",
        profile.name,
        settings.authority.mode,
        settings.server.workdir,
    )
    return Deployment(
        settings=settings,
        storage=storage,
        completion=LLMClient(profile, settings),
        profile=profile,
        transport_for=transport_for,
        authority=_authority(settings, storage.approvals),
        # Not the discarding default. Every model call, tool call, submission and
        # authority decision is meant to reach the ordered per-incident log, and a run
        # that built its events and dropped them is a run nobody can audit afterwards.
        sink=StoreSink(storage.runs),
    )


async def _drain(service: Service, timeout_s: float) -> None:
    """Let the incidents already in flight finish before the process goes.

    An incident is minutes to hours of model calls against a cluster it may already
    have changed, so killing one mid-remediation leaves the cluster half-modified with
    nobody recording what was done. Bounded, because a shutdown that never completes is
    its own failure: what is still running when the bound expires is named in the log
    so a reader knows which incidents to look at rather than discovering the gap later.

    Args:
        service: The process's resources.
        timeout_s: How long to wait. Zero exits immediately.
    """
    deadline = time.monotonic() + timeout_s
    while service.runs.live() and time.monotonic() < deadline:
        _log.info("draining %d incident(s) before shutdown", len(service.runs.live()))
        await asyncio.sleep(1.0)
    remaining = service.runs.live()
    if remaining:
        _log.warning("shut down with %d incident(s) unfinished: %s", len(remaining), remaining)


def create_app(
    *,
    deployment: Deployment | None = None,
    schedule_resume: Callable[[str], None] | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        deployment: An already-built deployment. When given, it is used as-is and left
            open on shutdown, since whoever built it owns it. Defaults to building one
            from the environment.
        schedule_resume: How a decided approval continues its incident. Defaults to a
            scheduled job, which survives a restart between the decision and the
            continuation.

    Returns:
        The application, with its service already on ``app.state``.
    """
    owned = deployment is None
    active = build_deployment(load_settings()) if deployment is None else deployment

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # A restored job arrives with nothing but its keyword arguments, so the process
        # has to tell it which deployment it belongs to.
        bind_deployment(active)
        register_post_incident_hooks(active.settings)
        # Instrumented either way; only export depends on a collector being configured.
        # Code that runs only when someone is watching is code the measured runs never
        # exercised.
        configure_tracing(active.settings)
        try:
            yield
        finally:
            await _drain(app.state.service, active.settings.server.drain_timeout_s)
            shutdown_scheduler()
            unbind_deployment()
            if owned:
                active.storage.close()

    docs = active.settings.server.environment != "production"
    app = FastAPI(
        title="kira",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs" if docs else None,
        redoc_url=f"{API_PREFIX}/redoc" if docs else None,
        openapi_url=f"{API_PREFIX}/openapi.json" if docs else None,
    )
    app.state.service = Service(
        deployment=active,
        schedule_resume=schedule_resume or schedule_approval_resume,
        runs=RunRegistry(),
    )
    # No prefix: a probe URL that moved with the API version would break every manifest
    # that already names it.
    app.include_router(health_router)
    app.include_router(incidents_router, prefix=API_PREFIX)
    app.include_router(hitl_router, prefix=API_PREFIX)
    for router in _adapter_routers(active):
        app.include_router(router, prefix=API_PREFIX)
    return app


def enabled_adapters(settings: Settings) -> frozenset[str]:
    """Which environment adapters this deployment runs under.

    Discovery is not selection: every installed adapter is imported so that one which
    could not be is *reported* rather than silently missing, and only the configured
    names are enabled. A deployment file written for the prototype selects through the
    switch its adapter declares, which is why the core never has to name one.

    Args:
        settings: The resolved configuration.

    Returns:
        The names to enable.
    """
    load_entry_point_adapters()
    return frozenset(settings.intake.adapters) | legacy_selection()


def _adapter_routers(deployment: Deployment) -> list[APIRouter]:
    """Build each enabled adapter once, to collect the routes it mounts.

    An adapter instance is per episode, so this one is used for nothing else and
    discarded. Building it here is also the moment its own configuration slice is
    validated, which turns a typo inside that slice into a startup failure rather than
    a default that quietly changes what an arm means.
    """
    routers: list[APIRouter] = []
    for name in sorted(enabled_adapters(deployment.settings)):
        router = get_adapter(name)(deployment.settings).routes()
        _log.info("environment adapter %s enabled", name)
        if isinstance(router, APIRouter):
            routers.append(router)
    return routers


def _authority(settings: Settings, approvals: ApprovalStore) -> Authority:
    """Choose who rules on a cluster write.

    Args:
        settings: The resolved configuration.
        approvals: Where pending requests are stored, used only in prompt mode.

    Returns:
        The configured authority.
    """
    if settings.authority.mode == "auto":
        return AutoApprove()
    if settings.authority.mode == "deny":
        return DenyAll()
    return HumanApproval(
        store=approvals,
        settings=settings.authority,
        arm_timeout=schedule_approval_timeout,
    )
