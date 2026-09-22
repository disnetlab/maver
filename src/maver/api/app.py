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
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI

from maver.adapters.base import EnvironmentAdapter
from maver.adapters.registry import (
    get_adapter,
    legacy_selection,
    load_entry_point_adapters,
)
from maver.api.deps import Service
from maver.api.health import router as health_router
from maver.api.hitl import router as hitl_router
from maver.api.incidents import router as incidents_router
from maver.api.queue import QueueConsumer
from maver.api.reconciliation import router as reconciliation_router
from maver.api.recover import recover_active
from maver.audit.durable_sink import DurableAuditSink
from maver.audit.journal import EventSink
from maver.audit.store_sink import StoreSink
from maver.configuration import Configuration, load_configuration
from maver.llm.client import LLMClient
from maver.memory.factory import build_storage
from maver.memory.ports import ApprovalStore
from maver.middleware.authority import Authority
from maver.middleware.human_approval import authorities
from maver.observability.tracing import configure_tracing
from maver.orchestration.bootstrap import register_post_incident_hooks
from maver.orchestration.control import RunRegistry
from maver.orchestration.jobs import (
    bind_deployment,
    schedule_approval_resume,
    schedule_approval_sweep,
    schedule_approval_timeout,
    unbind_deployment,
)
from maver.orchestration.runner import Deployment, TransportFactory
from maver.orchestration.scheduler import shutdown_scheduler
from maver.roles.registry import load_plugin_roles, validate_roles
from maver.secondpass.strategies import policy_for
from maver.settings.authority import Mode
from maver.settings.profiles import load_profile
from maver.settings.root import Settings, load_settings
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.registry import load_descriptions, load_entry_point_tools, validate_registry
from maver.tools.sandbox import fs_sandbox_available
from maver.tools.transport.base import ToolTransport
from maver.tools.transport.direct import direct_transport

__all__ = [
    "API_PREFIX",
    "build_deployment",
    "create_app",
    "enabled_adapters",
    "validate_deployment",
]

_log = logging.getLogger(__name__)

API_PREFIX = "/api/v1"
"""Where the incident and decision routes live. The health routes sit above it, because
a Kubernetes probe should not have to know the API's version."""


def validate_deployment(settings: Settings) -> Configuration:
    """Refuse a configuration this process cannot serve, before it serves anything.

    The one place these checks live, so that ``maver config check`` answers for the
    configuration ``serve`` builds rather than for a similar one. Every check the two
    did not share was a green pre-flight on a deployment that then crash-looped: an
    override declaring a marker nothing fills, a ranker nothing registered, or a
    gate-and-delivery pair naming no composition, which reached only the first incident
    because it was resolved inside the graph.

    The stores are opened and closed again, because a store that cannot be opened is a
    startup failure and there is no cheaper way to find out.

    Args:
        settings: The resolved configuration.

    Raises:
        FileNotFoundError: When the configured model profile does not exist.
        RoleRegistryError: When a role cannot terminate or cannot be prompted.
        SecondPassConfigError: When the configured gate and delivery name no
            composition.
        ToolRegistryError: When a tool and its description disagree.
        ValueError: When the configured storage backend or ranker is not installed.
    """
    load_profile(settings.model.profile, settings.model.profile_dir)
    # An out-of-tree role or tool is registered by importing the distribution that
    # declares it, and nothing else imports it. Loaded here rather than at package
    # import so that the checks below see the whole deployment: a plugin was invisible
    # to every one of them, and naming it in an arm file failed as an unknown role.
    load_plugin_roles()
    load_entry_point_tools()
    validate_registry(descriptions=load_descriptions(), allowlists=ROLE_TOOLSETS)
    validate_roles(override_dir=settings.prompts.override_dir)
    policy_for(settings.memory.second_pass, settings.memory)
    configuration = load_configuration(settings)
    build_storage(settings).close()
    return configuration


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
        SecondPassConfigError: When the configured gate and delivery name no
            composition.
        ValueError: When the configured storage backend is not installed.
    """
    configuration = validate_deployment(settings)
    # Asked here so that a host with no usable filesystem sandbox says so in the pod's
    # log at startup, rather than at the first tool call of the first incident.
    if settings.tools.fs_sandbox:
        fs_sandbox_available()
    profile = load_profile(settings.model.profile, settings.model.profile_dir)
    storage = build_storage(settings)

    def direct_for(session_id: str) -> ToolTransport:
        return direct_transport(settings.transport)

    transport_for: TransportFactory = direct_for
    environment = _process_bound_adapter(settings)
    bound: dict[str, Any] = {}
    if environment is not None:
        transport = environment.tool_transport()
        if transport is not None:

            def bound_for(session_id: str) -> ToolTransport:
                return transport

            transport_for = bound_for
        evidence = environment.evidence_policy()
        if evidence is not None:
            bound["evidence"] = evidence
        bound["tool_extras"] = environment.extra_tools()
        bound["adapter"] = environment
        _log.info(
            "environment adapter %s is bound for this process: every incident runs "
            "under its transport, its tools and its role overlays",
            environment.name,
        )

    # What this process is actually running, in one line an operator reads in the pod's
    # log. A campaign is only reconstructible if that is recorded beside its results.
    _log.info(
        "serving arm %s, profile %s under authority %s, workspace %s",
        settings.arm or "none",
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
        pipelines=configuration.pipelines,
        pipeline_selectors=configuration.selectors,
        completion_for=LLMClient,
        authorities=_authorities(settings, storage.approvals),
        # Not the discarding default. Every model call, tool call, submission and
        # authority decision is meant to reach the ordered per-incident log, and a run
        # that built its events and dropped them is a run nobody can audit afterwards.
        sink=DurableAuditSink(StoreSink(storage.runs), audit_dir=settings.server.workdir / "audit"),
        **bound,
    )


def _flush_sink(sink: EventSink) -> None:
    """Let the audit sink write its backlog, if it has one.

    Asked for by name rather than declared on :class:`~maver.audit.journal.EventSink`,
    because holding a backlog is one sink's implementation detail and the protocol is
    what a role has to satisfy to be written to at all.
    """
    closer = getattr(sink, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        _log.exception("the audit sink did not shut down cleanly")


_CANCELLATION_GRACE_S = 5.0
"""How long a cancelled incident gets to write that it is over.

Short, and separate from the drain budget it follows: by this point the work is not
being waited for any more, only the record of its end."""


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
    if not remaining:
        return

    # Cancellation lets each handler settle its record. With opt-in recovery the stage
    # checkpoint remains running for the next process; otherwise cancellation is final.
    _log.warning("cancelling %d incident(s) that did not drain: %s", len(remaining), remaining)
    carrying = service.runs.tasks()
    for task in carrying:
        task.cancel()
    if carrying:
        await asyncio.wait(carrying, timeout=_CANCELLATION_GRACE_S)
    still_live = service.runs.live()
    if still_live:
        _log.error("shut down with %d incident(s) unrecorded: %s", len(still_live), still_live)


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
        queue = QueueConsumer(app.state.service) if active.settings.intake.kafka.enabled else None
        # Register cleanup before initialization: a failed startup still owns resources,
        # and a failed queue stop must not skip draining, unbinding, or closing storage.
        async with AsyncExitStack() as cleanup:
            if owned:
                cleanup.callback(active.storage.close)
                cleanup.callback(_flush_sink, active.sink)
            cleanup.callback(unbind_deployment)
            cleanup.callback(shutdown_scheduler)
            cleanup.push_async_callback(
                _drain, app.state.service, active.settings.server.drain_timeout_s
            )
            if queue is not None:
                cleanup.push_async_callback(queue.stop)
            cleanup.callback(app.state.service.runs.begin_draining)

            bind_deployment(active, runs=app.state.service.runs)
            register_post_incident_hooks(active.settings)
            schedule_approval_sweep(active.settings.authority.poll_interval_s)
            configure_tracing(active.settings)
            if active.settings.server.resume_interrupted_incidents:
                await recover_active(app.state.service)
            if queue is not None:
                await queue.start()
                app.state.service.queue.consumer = queue
            yield

    docs = active.settings.server.environment != "production"
    app = FastAPI(
        title="maver",
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
    app.include_router(reconciliation_router, prefix=API_PREFIX)
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


def _process_bound_adapter(settings: Settings) -> EnvironmentAdapter | None:
    """The environment every incident in this process runs under, if there is one.

    An adapter is one of two shapes. It either mounts an episode route, in which case
    an outside driver posts one episode at a time and the episode driver binds it for
    the length of that episode; or it mounts none, in which case there is no episode to
    bind and its incidents arrive through the generic intake like any other. The second
    shape is bound here instead, once, for the process — which is what gives those
    incidents its transport, its tools and its role overlays.

    Its lifecycle half runs in whatever holds the environment, which for a benchmark
    that drives us is its own driver process. This process never calls it, and the
    methods say so rather than returning something plausible.

    Args:
        settings: The resolved configuration.

    Returns:
        The adapter to bind, or None when nothing is enabled or everything enabled
        drives its own episodes.

    Raises:
        ValueError: When two such adapters are enabled. Only one can supply the
            transport, and picking one would make the run's record a guess.
    """
    bound = [
        built
        for built in (get_adapter(name)(settings) for name in sorted(enabled_adapters(settings)))
        if built.routes() is None
    ]
    if len(bound) > 1:
        raise ValueError(
            "two environment adapters mount no episode route and would both be bound "
            f"for this process: {', '.join(sorted(a.name for a in bound))}. Only one can "
            "supply the transport its tools speak; enable one."
        )
    return bound[0] if bound else None


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


def _authorities(settings: Settings, approvals: ApprovalStore) -> dict[Mode, Authority]:
    """Build who rules on a cluster write, one decider per mode.

    All three, not the one the deployment's mode names, because a policy rule names a
    mode of its own and the gate resolves the governing mode per call.

    Args:
        settings: The resolved configuration.
        approvals: Where pending requests are stored, used by the human authority.

    Returns:
        A decider for each mode.
    """
    return authorities(settings.authority, approvals, schedule_approval_timeout)
