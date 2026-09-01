"""Running one incident, from the alert to the record of what happened.

This is the composition root for an incident. Everything a role touches — its tools,
its transport, its workspace, its stores, its audit journal — is built here and handed
down, so a role reads no global and a test substitutes any of it by passing a different
object. The prototype reached most of these through module-level singletons and four
context variables, which is why its roles could not be run without a fixture that set
all four, and why one of those variables leaked between incidents.

Everything scoped to the incident is bound and unbound by a context manager rather than
set and forgotten. The prototype bound the incident scope and discarded the token that
would have released it, so the last incident's namespace stayed visible to whatever ran
next on that task. Under one incident per process that is harmless, which is why it
survived for as long as it did.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from kira.audit.journal import EventSink, Journal, NullSink
from kira.contracts.events import StageEvent
from kira.contracts.lifecycle import RunStatus
from kira.contracts.routing import HandBack
from kira.contracts.state import IncidentState, TargetStage
from kira.llm.types import Completion
from kira.memory.factory import StorageStack
from kira.memory.ports import AgentRunRecord, AgentRunStatus
from kira.middleware.authority import ApprovalPending, Authority, AutoApprove
from kira.middleware.base import AgentHook, MiddlewareStack
from kira.middleware.write_gate import WriteGate
from kira.orchestration.budget import RetryBudget
from kira.orchestration.engine import Engine, LangGraphEngine, SequentialEngine, StageHook
from kira.orchestration.graph import build_graph
from kira.orchestration.hooks import fire_hooks
from kira.orchestration.modes import MODES, ModeContext, resolve_mode
from kira.orchestration.nodes import ContextBuilder, IncidentDeps, make_node
from kira.orchestration.outcomes import OutcomeRecord, persist_outcome, resolve_outcome
from kira.orchestration.resume import save_checkpoint
from kira.orchestration.routing import RoutingPolicy
from kira.orchestration.state import initial_state
from kira.roles.base import RoleContext, RoleSpec, StateDelta
from kira.roles.registry import get_role, registered_roles
from kira.settings.profiles import ModelProfile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.scope import IncidentScope
from kira.tools.toolsets import build_toolset
from kira.tools.transport.base import ToolTransport
from kira.tools.workspace import SessionWorkspace

__all__ = ["Deployment", "IncidentResources", "incident_context", "run_incident"]

_log = logging.getLogger(__name__)

TransportFactory = Callable[[str], ToolTransport]
"""Builds this incident's transport. Per run, because one may hold a correlation id."""

GRAPH_ROLES = ("rca", "planning", "execution", "verify")


@dataclass(frozen=True)
class Deployment:
    """What every incident in this process shares."""

    settings: Settings
    storage: StorageStack
    completion: Completion
    profile: ModelProfile
    transport_for: TransportFactory
    authority: Authority = field(default_factory=AutoApprove)
    """Who rules on a cluster write. The gate that consults it is built per role run,
    because it records into that run's journal."""
    hooks: Sequence[AgentHook] = ()
    """Extra middleware, composed after the write gate."""
    sink: EventSink = field(default_factory=NullSink)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """One cluster writer per process. Held here rather than in a module global so two
    deployments in one process are genuinely independent, which is what makes the
    serialisation testable."""


@dataclass(frozen=True, slots=True)
class IncidentResources:
    """Everything bound for the length of one incident, and released with it."""

    workspace: SessionWorkspace
    scope: IncidentScope
    transport: ToolTransport
    incident_dir: Path
    shared_dir: Path


@asynccontextmanager
async def incident_context(
    session_id: str, alert: dict[str, Any], deployment: Deployment
) -> AsyncIterator[IncidentResources]:
    """Bind, and always unbind, everything scoped to one incident.

    Args:
        session_id: The incident.
        alert: The payload that raised it.
        deployment: The process's shared resources.

    Yields:
        The incident's own resources.
    """
    workdir = deployment.settings.server.workdir
    incident_dir = workdir / "incidents" / session_id
    incident_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = workdir / "memory"

    transport = deployment.transport_for(session_id)
    resources = IncidentResources(
        workspace=SessionWorkspace(incident_dir),
        # Namespace is trusted alert context, not a model-supplied tool argument.
        # Telemetry wrappers use it to prevent cross-application evidence leakage.
        scope=IncidentScope.from_alert(alert),
        transport=transport,
        incident_dir=incident_dir,
        shared_dir=shared_dir,
    )
    try:
        yield resources
    finally:
        closer = getattr(transport, "close", None)
        if closer is not None:
            try:
                await closer() if asyncio.iscoroutinefunction(closer) else closer()
            except Exception:
                _log.exception("closing the transport for %s failed", session_id)


class _RunRecorder:
    """Opens and closes the persisted row for each role run."""

    def __init__(self, storage: StorageStack) -> None:
        """Bind to this deployment's stores."""
        self._storage = storage
        self._counts: dict[str, int] = {}

    def start(self, *, session_id: str, role: str, after: str | None) -> str:
        """Open a run row.

        Args:
            session_id: The incident.
            role: The role starting.
            after: The previous run along this path.

        Returns:
            The new run's identifier.
        """
        self._counts[role] = self._counts.get(role, 0) + 1
        run_id = f"{session_id}:{role}:{self._counts[role]}"
        self._storage.runs.start_run(
            AgentRunRecord(
                agent_run_id=run_id,
                pipeline_id=session_id,
                agent_name=role,
                attempt=self._counts[role],
                status="running",
                started_at="",
                metadata={"after": after} if after else {},
            )
        )
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        output: Mapping[str, Any] | None,
        trace_summary: Mapping[str, Any] | None,
        error: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        """Close a run row.

        Args:
            run_id: Which run.
            status: How it ended.
            output: The artifact it produced.
            trace_summary: Its bounded account of itself.
            error: Why it failed, when it did.
            metadata: Anything the role recorded about its own run.
        """
        self._storage.runs.finish_run(
            run_id,
            status=_RUN_STATUS.get(status, "failed"),
            output=dict(output) if output is not None else None,
            trace_summary=dict(trace_summary) if trace_summary is not None else None,
            error=error,
        )


_RUN_STATUS: Mapping[str, AgentRunStatus] = {
    "succeeded": "succeeded",
    "failed": "failed",
    "halted": "halted",
}
"""A halt is a role doing its job, so its row records the halt rather than a failure.
The prototype's row said `failed`, which made a deliberate stop indistinguishable from a
crash in every count taken downstream. See :mod:`kira.orchestration.outcomes`."""


def _engine(deployment: Deployment, deps: IncidentDeps) -> Engine:
    """Build the configured executor over the same nodes either one would run."""
    nodes = {name: make_node(get_role(name), deps) for name in GRAPH_ROLES}
    if deployment.settings.pipeline.engine == "sequential":
        return SequentialEngine(nodes, deps.policy)
    return LangGraphEngine(build_graph(nodes, deps.policy))


def _context_builder(
    deployment: Deployment,
    resources: IncidentResources,
    on_stage: StageHook | None,
) -> ContextBuilder:
    """Assemble a role's context from the incident's own resources."""

    def build(
        spec: RoleSpec, state: IncidentState, handback: HandBack | None, run_id: str
    ) -> RoleContext:
        journal = Journal(
            deployment.sink,
            session_id=state.get("session_id", ""),
            stage=spec.name,
            run_id=run_id,
        )
        tools = ToolContext(
            transport=resources.transport,
            workspace=resources.workspace,
            role=spec.name,
            session_id=state.get("session_id", ""),
            journal=journal,
            settings=deployment.settings.tools,
            scope=resources.scope,
            memory=deployment.storage.retriever if deployment.settings.memory.enabled else None,
            sessions=deployment.storage.sessions,
            runs=deployment.storage.runs,
        )

        async def milestone(stage: str, delta: StateDelta) -> None:
            # A role that answers more than one graded question announces each as it
            # lands. The adapter above sees the same stage names either configuration
            # produces, which is what lets one driver grade both.
            if on_stage is not None:
                await on_stage(stage, cast(IncidentState, {**state, **delta}))

        return RoleContext(
            spec=spec,
            state=state,
            handback=handback,
            settings=deployment.settings,
            completion=deployment.completion,
            profile=deployment.profile,
            middleware=MiddlewareStack(
                [
                    WriteGate(
                        authority=deployment.authority,
                        settings=deployment.settings.authority,
                        journal=journal,
                    ),
                    *deployment.hooks,
                ]
            ),
            write_lock=deployment.write_lock,
            toolset=build_toolset(spec.name, tools),
            tool_context=tools,
            journal=journal,
            run_id=run_id,
            session_id=state.get("session_id", ""),
            memory=tools.memory,
            milestone=milestone,
        )

    return build


async def run_incident(
    alert: dict[str, Any],
    session_id: str,
    deployment: Deployment,
    *,
    on_stage: StageHook | None = None,
    fire_post_hooks: bool = True,
    finalize_run_status: bool = True,
    target_stage: TargetStage = "full",
) -> IncidentState:
    """Run one incident to a terminal state and record what happened.

    Args:
        alert: The payload that raised the incident.
        session_id: The persisted session this run belongs to.
        deployment: The process's shared resources.
        on_stage: Called with ``(stage_name, state)`` after each role completes. When
            provided the graph is streamed, so a caller such as a benchmark episode can
            react as artifacts become available rather than waiting for the end. It must
            not raise.
        fire_post_hooks: When False the post-incident hooks are not scheduled, and the
            caller takes over that lifecycle — for an adapter that runs the distilling
            pass itself before releasing its environment.
        finalize_run_status: When False the terminal run status is not written and the
            run stays running, so a poller keeps waiting through post-pipeline work. The
            caller then finalises it. The lifecycle state is recorded immediately either
            way.
        target_stage: How far to run.

    Returns:
        The terminal incident state.

    Raises:
        ValueError: When ``target_stage`` is not one this system runs.
    """
    if target_stage not in ("full", "diagnose"):
        raise ValueError(f"unsupported target_stage: {target_stage!r}")

    async with incident_context(session_id, alert, deployment) as resources:
        state = initial_state(
            session_id=session_id,
            alert=alert,
            target_stage=target_stage,
            incident_dir=resources.incident_dir,
            shared_dir=resources.shared_dir,
        )
        journal = Journal(deployment.sink, session_id=session_id, stage="pipeline")
        # Written before any role runs, and first. A campaign is reconstructible only if
        # the configuration it ran under is in the record beside its result.
        journal.record(
            StageEvent(
                stage="pipeline",
                event="entered",
                detail=json.dumps(deployment.settings.snapshot(), sort_keys=True),
            )
        )

        deps = IncidentDeps(
            policy=RoutingPolicy.from_settings(deployment.settings),
            runs=_RunRecorder(deployment.storage),
            journal=journal,
            build_context=_context_builder(deployment, resources, on_stage),
        )
        mode = resolve_mode(target_stage=target_stage, settings=deployment.settings)
        try:
            final_state = await MODES[mode](
                state,
                ModeContext(deps=deps, engine=_engine(deployment, deps), on_stage=on_stage),
            )
        except ApprovalPending as pending:
            # Not an error. Park the incident where it stopped and let the caller unwind
            # cleanly; nothing is persisted as terminal and no hook fires, because the
            # run has not finished — it is waiting.
            save_checkpoint(
                deployment.storage.sessions, cast(IncidentState, pending.state), pending
            )
            raise

        record = resolve_outcome(final_state)
        try:
            persist_outcome(
                final_state,
                record,
                deployment.storage.sessions,
                finalize_run_status=finalize_run_status,
            )
        except Exception:
            # A storage failure must never kill a finished pipeline: the caller still
            # reports its outcome from the in-memory state, and an unsanitised byte in
            # an artifact once cost a fully graded run here.
            _log.exception("persisting the outcome failed; continuing from memory")

        if fire_post_hooks and target_stage == "full":
            # A diagnosis-only task has no remediation outcome to learn from. Never
            # distil it as a successful incident memory entry.
            await fire_hooks(record.outcome, session_id, workdir=deployment.settings.server.workdir)

        _record_completion(journal, final_state, record, deps.policy)
        return final_state


def _record_completion(
    journal: Journal,
    state: IncidentState,
    record: OutcomeRecord,
    policy: RoutingPolicy,
) -> None:
    """Close the incident's log with what it cost and what it came to."""
    budget = RetryBudget.of(state, policy)
    turns = {
        role: sum(
            1 for m in state.get("messages", {}).get(role, []) if m.get("role") == "assistant"
        )
        for role in registered_roles()
        if state.get("messages", {}).get(role)
    }
    journal.record(
        StageEvent(
            stage="pipeline",
            event="exited",
            detail=json.dumps(
                {
                    "outcome": record.outcome.value,
                    "lifecycle": record.lifecycle.value if record.lifecycle else None,
                    "run_status": (
                        record.run_status.value if record.run_status else RunStatus.RUNNING.value
                    ),
                    "remediation_retries": budget.remediation_spent,
                    "reassessments": budget.reassessments,
                    "verify_rechecks": budget.verify_rechecks,
                    "assistant_turns": turns,
                },
                sort_keys=True,
            ),
        )
    )
