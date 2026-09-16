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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from maver.adapters.base import AdapterTool, EnvironmentAdapter
from maver.agents import AgentDefinition
from maver.audit.journal import EventSink, Journal, NullSink, SequenceRegistry
from maver.configuration import behavior_digest, load_configuration
from maver.contracts import SCHEMA_VERSION
from maver.contracts.events import InterventionEvent, StageEvent
from maver.contracts.lifecycle import RunStatus
from maver.contracts.routing import HandBack, Route
from maver.contracts.runtime import AgentState, RunState
from maver.contracts.state import IncidentState, TargetStage
from maver.llm.sampling import (
    SamplingResolution,
    assert_sampling_is_pinned,
    resolve_sampling,
    split_sampling,
)
from maver.llm.types import Completion, ToolResult
from maver.memory.factory import StorageStack
from maver.memory.ports import AgentRunRecord, AgentRunStatus, ApprovalRecord
from maver.middleware.authority import ApprovalPending, Authority, AutoApprove
from maver.middleware.base import AgentHook, MiddlewareStack, ToolRequest
from maver.middleware.human_approval import approval_granted
from maver.middleware.write_gate import WriteGate
from maver.observability.tracing import span
from maver.observability.usage import RecordingCompletion
from maver.orchestration.budget import RetryBudget
from maver.orchestration.engine import Engine, LangGraphEngine, SequentialEngine
from maver.orchestration.graph import build_graph
from maver.orchestration.hooks import fire_hooks
from maver.orchestration.modes import MODES, ModeContext, resolve_mode
from maver.orchestration.nodes import ContextBuilder, IncidentDeps, NodeFn, make_node
from maver.orchestration.outcomes import OutcomeRecord, persist_outcome, resolve_outcome
from maver.orchestration.resume import (
    ApprovalCheckpoint,
    denial_result,
    load_checkpoint,
    restored_state,
    resume_events,
    save_checkpoint,
)
from maver.orchestration.routing import RoutingPolicy, pending_handback
from maver.orchestration.runtime import save_runtime
from maver.orchestration.state import initial_state
from maver.pipelines import (
    PipelineCatalog,
    PipelineChangedError,
    PipelineSelector,
    ResolvedPipeline,
    SelectedPipeline,
    restore_selection,
    select_pipeline,
)
from maver.roles.base import RoleContext, RoleSpec, StateDelta, stage_role
from maver.roles.policy import close_dangling_tool_calls
from maver.roles.registry import get_role, registered_roles
from maver.secondpass.audit import JournalEmitter
from maver.secondpass.runner import SecondPass
from maver.secondpass.strategies import policy_for
from maver.secondpass.types import (
    DiagnosisRun,
    SecondPassContext,
    SecondPassPolicy,
    SecondPassResult,
)
from maver.settings.authority import Mode
from maver.settings.profiles import ModelProfile
from maver.settings.root import Settings
from maver.tools.context import EvidencePolicy, NullEvidencePolicy, ToolContext
from maver.tools.scope import IncidentScope
from maver.tools.snapshot import namespace_snapshot
from maver.tools.toolsets import build_toolset
from maver.tools.transport.base import ToolTransport
from maver.tools.workspace import SessionWorkspace

__all__ = [
    "Deployment",
    "IncidentResources",
    "incident_context",
    "record_hook_failures",
    "resume_after_decision",
    "run_incident",
    "tool_context_for",
    "withheld_records",
]

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
    pipelines: PipelineCatalog | None = None
    pipeline_selectors: tuple[PipelineSelector, ...] = ()
    completion_for: Callable[[ModelProfile, Settings], Completion] | None = None
    authorities: Mapping[Mode, Authority] = field(default_factory=lambda: {"auto": AutoApprove()})
    """Who rules on a cluster write, per mode. The gate that consults them is built per
    role run, because it records into that run's journal.

    One per mode rather than one for the deployment, because a policy rule names a mode
    of its own. The default carries only the unattended decider, which is the posture a
    deployment assembled in code without an approval store is in; the served process
    supplies all three."""
    hooks: Sequence[AgentHook] = ()
    """Extra middleware, composed after the write gate."""
    evidence: EvidencePolicy = field(default_factory=NullEvidencePolicy)
    """A last-mile filter over what tools return, supplied by an environment adapter.

    An environment can leak its own fault injection into the telemetry it serves, which
    turns a diagnosis into a lookup. Filtering belongs to whatever knows that about its
    environment, so it arrives here rather than being a rule the tools carry."""
    tool_extras: tuple[AdapterTool, ...] = ()
    """Tools an environment adapter supplies, each carrying the roles allowed them."""
    adapter: EnvironmentAdapter | None = None
    """The environment bound to every incident in this composition, or None.

    Held so that a role can be narrowed for the environment it is running under. It is
    the same instance the rest of the composition came from, because an overlay may read
    what its episode acquired, and it is None whenever nothing is bound — which is what
    makes an unbound run identical to one taken before this field existed."""
    sink: EventSink = field(default_factory=NullSink)
    sequences: SequenceRegistry = field(default_factory=SequenceRegistry)
    """The event position each session's journals share. Per deployment for the same
    reason the write lock is: a session is written by a journal per stage, and an order
    that restarts with each of them orders nothing."""
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """One cluster writer per process. Held here rather than in a module global so two
    deployments in one process are genuinely independent, which is what makes the
    serialisation testable."""
    runtime_digest: str = field(init=False)
    sampling: SamplingResolution = field(init=False)
    """The profile's set resolved against settings, once, for the whole process."""

    def __post_init__(self) -> None:
        """Resolve the sampling set once, and refuse a deployment that cannot send it."""
        # Resolved and checked here because this is the composition root: every entry
        # point builds a Deployment, so a set that cannot be sent fails at startup
        # rather than at the first call, or — the case that actually happened — not at
        # all, with the run completing on a configuration nobody could reconstruct.
        if self.pipelines is None:
            configuration = load_configuration(self.settings, default_profile=self.profile)
            object.__setattr__(self, "pipelines", configuration.pipelines)
            object.__setattr__(
                self, "pipeline_selectors", (*configuration.selectors, *self.pipeline_selectors)
            )
        assert self.pipelines is not None
        if self.completion_for is None and any(
            agent.profile is not None and agent.profile != self.profile
            for pipeline in self.pipelines.pipelines.values()
            for agent in (*pipeline.agents.values(), pipeline.learning_agent)
            if agent is not None
        ):
            raise ValueError("configured agent model profiles require a completion_for factory")
        for pipeline in self.pipelines.pipelines.values():
            definitions = [*pipeline.agents.values()]
            if pipeline.learning_agent is not None:
                definitions.append(pipeline.learning_agent)
            for definition in definitions:
                effective = definition.settings_for(self.settings)
                profile = definition.profile or self.profile
                assert_sampling_is_pinned(profile, resolve_sampling(profile, effective))
        object.__setattr__(self, "runtime_digest", behavior_digest(self.settings, self.profile))
        resolution = resolve_sampling(self.profile, self.settings)
        assert_sampling_is_pinned(self.profile, resolution)
        # Both sets, because both are sent. The gate, the nominator and the selector run
        # on the auxiliary one, and it was checked nowhere: a set that cannot be sent
        # failed at the first helper call, in the middle of an incident.
        aux = self.profile.aux_sampling or self.profile.sampling
        assert_sampling_is_pinned(self.profile, split_sampling(aux, self.profile))
        object.__setattr__(self, "sampling", resolution)


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

    def start(self, *, session_id: str, role: str, after: str | None) -> str:
        """Open a run row.

        Args:
            session_id: The incident.
            role: The role starting.
            after: The previous run along this path.

        Returns:
            The new run's identifier.
        """
        # Counted from what is stored rather than from this object, because a
        # continuation after an approval is a second recorder over the same incident and
        # would otherwise reuse the identifiers the first one issued.
        attempt = 1 + sum(
            1 for run in self._storage.runs.runs_for(session_id) if run.agent_name == role
        )
        run_id = f"{session_id}:{role}:{attempt}"
        self._storage.runs.start_run(
            AgentRunRecord(
                agent_run_id=run_id,
                pipeline_id=session_id,
                agent_name=role,
                attempt=attempt,
                status="running",
                # A real instant, not a placeholder the store is expected to fill in.
                # One backend accepted the empty string and ordered runs by insertion
                # instead; the other refused it, and refusing it at the first role of
                # every incident released the environment with nothing submitted, which
                # grades as an agent that never answered.
                started_at=datetime.now(UTC).isoformat(),
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
            metadata=dict(metadata),
            error=error,
        )


_RUN_STATUS: Mapping[str, AgentRunStatus] = {
    "succeeded": "succeeded",
    "failed": "failed",
    "halted": "halted",
}
"""A halt is a role doing its job, so its row records the halt rather than a failure.
The prototype's row said `failed`, which made a deliberate stop indistinguishable from a
crash in every count taken downstream. See :mod:`maver.orchestration.outcomes`."""


def record_hook_failures(journal: Journal) -> Callable[[str, str, BaseException], None]:
    """Send a hook that raised to the incident's own log as well as to the process log.

    A shaping or reviewing hook that fails is skipped, and skipped silently it is a
    cross-cutting concern that was configured, was reached, and did not run — invisible
    to any reconstruction of the incident, which is the record the auditability claim
    rests on. The stack's default reports to a Python logger no run reconstruction will
    ever see.

    Args:
        journal: The incident's log.

    Returns:
        The callback a :class:`MiddlewareStack` reports hook failures through.
    """

    def report(hook: str, phase: str, exc: BaseException) -> None:
        _log.warning("hook %s failed in %s and was skipped: %s", hook, phase, exc, exc_info=exc)
        journal.record(
            InterventionEvent(
                control="hook_failure",
                action="bypassed",
                detail=f"{hook} raised in {phase}: {type(exc).__name__}: {exc}",
            )
        )

    return report


def _engine(
    deployment: Deployment,
    deps: IncidentDeps,
    nodes: Mapping[str, NodeFn],
    pipeline: ResolvedPipeline | None = None,
) -> Engine:
    """Build the configured executor over the nodes it is given.

    Given rather than built here, so the executor and every mode run the same objects.
    Two compositions of one incident are equivalent today and one refactor away from
    not being.
    """
    if deployment.settings.pipeline.engine == "sequential":
        return SequentialEngine(nodes, deps.policy, pipeline=pipeline)
    return LangGraphEngine(build_graph(nodes, deps.policy, pipeline=pipeline), pipeline=pipeline)


def _nodes(
    deployment: Deployment,
    deps: IncidentDeps,
    resources: IncidentResources,
    pipeline: ResolvedPipeline | None = None,
) -> dict[str, NodeFn]:
    """The graph's roles, with the diagnosing one composed if a second pass is configured.

    The mechanism wraps the diagnosis rather than living inside it. Both of its runs are
    ordinary role runs — same wrapper, same run rows, same audit — and what differs is
    only what they are told and whether they may retrieve. A role that knew it was being
    second-guessed would be a different role in the arm that does it.
    """
    nodes = (
        {
            stage.name: make_node(pipeline.agents[stage.name].spec, deps, stage=stage.name)
            for stage in pipeline.workflow.stages
        }
        if pipeline is not None
        else {name: make_node(get_role(name), deps) for name in GRAPH_ROLES}
    )
    policy = policy_for(deployment.settings.memory.second_pass, deployment.settings.memory)
    if (
        policy is not None
        and "rca" in nodes
        and (pipeline is None or pipeline.workflow.stage("rca").role == "rca")
    ):
        nodes["rca"] = _second_pass_node(
            deployment, deps, resources, policy, nodes["rca"], pipeline
        )
    return nodes


def _second_pass_node(
    deployment: Deployment,
    deps: IncidentDeps,
    resources: IncidentResources,
    policy: SecondPassPolicy,
    plain: NodeFn,
    pipeline: ResolvedPipeline | None = None,
) -> NodeFn:
    """Run the gated second opinion in place of a first diagnosis.

    A later re-diagnosis runs the plain node instead. Repeating the pass would weigh
    retrieved precedent against the verification this incident just performed, which is
    the stronger signal, and would make precedent exposure a function of how the first
    repair turned out — the confound the mechanism is measured against. It also renders
    the assignment slot on top of the reassessment block, a combination no prompt is
    written for.
    """

    async def node(state: IncidentState) -> StateDelta:
        handback = pending_handback(state)
        if handback is not None and handback.target is Route.RCA:
            return await plain(state)

        # What the inner runs' own wrappers produced, kept per entry. The mechanism
        # reduces a run to the four fields it arbitrates over; everything else the
        # wrapper decided — the hand-back accounting above all — belongs to the incident
        # and is merged back below.
        inner: list[Mapping[str, Any]] = []

        async def diagnose(
            state: IncidentState, *, brief: str | None, retrieval: bool, run_label: str
        ) -> DiagnosisRun:
            rca = make_node(
                pipeline.agents["rca"].spec if pipeline is not None else get_role("rca"),
                replace(
                    deps,
                    build_context=_context_builder(
                        deployment,
                        resources,
                        None,
                        assignment=brief or "",
                        retrieval=retrieval,
                        run_label=run_label,
                        pipeline=pipeline,
                    ),
                ),
            )
            delta = dict(await rca(state))
            inner.append(delta)
            run_ids = delta.get("run_ids") or []
            return DiagnosisRun(
                artifact=delta.get("root_cause"),
                run_id=str(run_ids[-1]) if run_ids else None,
                messages=tuple(delta.get("messages", {}).get("rca", ())),
                trace_summaries=tuple(delta.get("trace_summaries", ())),
            )

        journal = Journal(
            deployment.sink,
            session_id=state.get("session_id", ""),
            stage="rca",
            sequence=deployment.sequences.for_session(state.get("session_id", "")),
        )
        tools = tool_context_for(
            deployment,
            resources,
            "rca",
            state.get("session_id", ""),
            journal,
            withheld=state.get("withheld_record_ids", ()),
        )

        async def snapshot() -> str:
            namespace = str((state.get("alert") or {}).get("namespace") or "").strip()
            if not namespace:
                return ""
            try:
                return str(await namespace_snapshot(tools, namespace))
            except Exception:  # a missing snapshot narrows the query, it does not fail it
                _log.warning("snapshot for the pre-anchor query could not be collected")
                return ""

        result = await SecondPass(policy, rca=diagnose).run(
            SecondPassContext(
                session_id=state.get("session_id", ""),
                alert=dict(state.get("alert") or {}),
                state=state,
                completion=RecordingCompletion(
                    deployment.completion, journal, deployment.profile, deployment.sampling
                ),
                sampling=deployment.profile.aux_sampling or deployment.profile.sampling,
                memory=deployment.storage.retriever if deployment.settings.memory.enabled else None,
                snapshot=snapshot,
                withheld_record_ids=withheld_records(deployment, state),
                emit=JournalEmitter(journal, policy=policy.name),
            )
        )
        return _second_pass_delta(state, result, inner)

    return node


def _second_pass_delta(
    state: IncidentState, result: SecondPassResult, inner: Sequence[Mapping[str, Any]]
) -> StateDelta:
    """Merge both diagnosis runs into the state, not only the winner.

    The prototype returned one of the two states, so the losing run's identifier never
    entered the ordered path: both rows existed in the database and the chain that
    linked them did not.

    Built on what the inner wrappers returned rather than assembled from scratch, so a
    composed diagnosis node returns everything an ordinary one does. Picking four keys
    off it dropped the rest — the hand-back accounting most of all, which left a spent
    verdict live in the state and a retry budget that never fell.

    Args:
        state: The state the node was entered with.
        result: What the mechanism decided.
        inner: Each inner diagnosis run's own delta, in the order they ran.

    Returns:
        The state delta for the whole pass.
    """
    runs = [run for run in (result.first, result.second) if run is not None]
    delta: dict[str, Any] = {}
    for produced in inner:
        delta.update(produced)
    delta.update(
        {
            "root_cause": result.chosen,
            "messages": {**state.get("messages", {}), "rca": list(runs[-1].messages)},
            "trace_summaries": [
                *state.get("trace_summaries", []),
                *(dict(summary) for run in runs for summary in run.trace_summaries),
            ],
            "run_ids": [
                *state.get("run_ids", []),
                *(run.run_id for run in runs if run.run_id),
            ],
        }
    )
    if "runtime" in state:
        state["runtime"].artifacts["root_cause"] = (
            result.chosen.model_dump(mode="json") if result.chosen is not None else None
        )
    if result.chosen is None:
        delta["error"] = f"rca failed: {result.skipped or 'no diagnosis'}"
    return delta


def _context_builder(
    deployment: Deployment,
    resources: IncidentResources,
    resume_for: dict[str, list[dict[str, Any]]] | None = None,
    resume_stage: str = "",
    assignment: str = "",
    retrieval: bool | None = None,
    run_label: str = "",
    pipeline: ResolvedPipeline | None = None,
    definition_override: AgentDefinition | None = None,
) -> ContextBuilder:
    """Assemble a role's context from the incident's own resources.

    ``resume_for`` carries the transcript a paused role should open on, keyed by that
    role, and ``resume_stage`` the label it was paused under. It is consumed on the
    first entry: a role re-entered later in the same run is starting a fresh cycle, not
    continuing the one that was interrupted.
    """
    pending_resume = dict(resume_for or {})
    # A run told not to retrieve is given no store at all, so the toolset drops the
    # retrieval tools and the prompt composer selects the memory-off block. Gating in
    # one place is what keeps the two from disagreeing.
    retrieving = deployment.settings.memory.enabled and deployment.settings.memory.role_retrieval
    if retrieval is not None:
        retrieving = retrieval

    def build(
        spec: RoleSpec, state: IncidentState, handback: HandBack | None, run_id: str
    ) -> RoleContext:
        runtime = state.setdefault("runtime", RunState())
        stage = runtime.active_stage or spec.name
        definition = definition_override or (
            pipeline.agents.get(stage) if pipeline is not None else None
        )
        settings = (
            definition.settings_for(deployment.settings) if definition else deployment.settings
        )
        profile = (definition.profile if definition else None) or deployment.profile
        sampling = resolve_sampling(profile, settings)
        assert_sampling_is_pinned(profile, sampling)
        completion = (
            deployment.completion_for(profile, settings)
            if definition is not None
            and definition.profile is not None
            and deployment.completion_for is not None
            else deployment.completion
        )
        resumed = pending_resume.pop(stage, None)
        agent = runtime.agents.get(runtime.active_agent or "") if resumed is not None else None
        if agent is None:
            agent = AgentState(
                id=run_id,
                definition=definition.name if definition else spec.name,
                stage=stage,
                role=spec.name,
            )
            runtime.agents[agent.id] = agent
        runtime.active_agent = agent.id
        journal = Journal(
            deployment.sink,
            session_id=state.get("session_id", ""),
            stage=spec.name,
            run_id=run_id,
            sequence=deployment.sequences.for_session(state.get("session_id", "")),
        )
        tools = tool_context_for(
            deployment,
            resources,
            spec.name,
            state.get("session_id", ""),
            journal,
            retrieving=retrieving and settings.memory.enabled,
            withheld=state.get("withheld_record_ids", ()),
        )

        # Asked once, before the role builds anything, so every application of it is
        # downstream of one call. An adapter that overlays nothing answers None and the
        # role is composed exactly as it would be with no environment at all.
        overlay = (
            deployment.adapter.role_overlay(spec.name, state)
            if deployment.adapter is not None
            else None
        )

        context = RoleContext(
            spec=spec,
            state=state,
            handback=handback,
            settings=settings,
            completion=RecordingCompletion(completion, journal, profile, sampling),
            profile=profile,
            middleware=MiddlewareStack(
                [
                    WriteGate(
                        authorities=deployment.authorities,
                        settings=deployment.settings.authority,
                        tools=deployment.settings.tools,
                        journal=journal,
                    ),
                    *deployment.hooks,
                ],
                on_error=record_hook_failures(journal),
                run_state=runtime,
                agent_state=agent,
            ),
            write_lock=deployment.write_lock,
            toolset=build_toolset(
                spec.name,
                tools,
                extras=deployment.tool_extras,
                drop=overlay.drop_tools if overlay is not None else frozenset(),
                names=definition.tool_names if definition is not None else None,
            ),
            tool_context=tools,
            journal=journal,
            run_id=run_id,
            session_id=state.get("session_id", ""),
            memory=tools.memory,
            run_state=runtime,
            agent_state=agent,
            overlay=overlay,
            assignment=assignment,
            run_label=run_label,
            resume=resumed,
            resume_stage=resume_stage if resumed is not None else "",
        )
        if definition is not None:
            for hook in definition.make_middleware(context):
                context = replace(context, middleware=context.middleware.with_hook(hook))
        return context

    return build


async def run_incident(
    alert: dict[str, Any],
    session_id: str,
    deployment: Deployment,
    *,
    fire_post_hooks: bool = True,
    finalize_run_status: bool = True,
    target_stage: TargetStage = "full",
    withheld_record_ids: Sequence[str] = (),
    selection: SelectedPipeline | None = None,
    metadata: dict[str, Any] | None = None,
) -> IncidentState:
    """Run one incident to a terminal state and record what happened.

    Args:
        alert: The payload that raised the incident.
        session_id: The persisted session this run belongs to.
        deployment: The process's shared resources.
        fire_post_hooks: When False the post-incident hooks are not scheduled, and the
            caller takes over that lifecycle — for an adapter that runs the distilling
            pass itself before releasing its environment.
        finalize_run_status: When False the terminal run status is not written and the
            run stays running, so a poller keeps waiting through post-pipeline work. The
            caller then finalises it. The lifecycle state is recorded immediately either
            way.
        target_stage: How far to run.
        selection: A selection already resolved at admission, or None to select here.
        metadata: Private JSON metadata available to middleware, never injected implicitly.
        withheld_record_ids: Corpus records this incident may not retrieve, unioned with
            the process-wide floor in settings rather than replacing it. An environment
            adapter supplies one per episode; a plain deployment supplies none.

    Returns:
        The terminal incident state.

    Raises:
        ValueError: When ``target_stage`` is not one this system runs.
    """
    if target_stage not in ("full", "diagnose"):
        raise ValueError(f"unsupported target_stage: {target_stage!r}")

    assert deployment.pipelines is not None
    if selection is None:
        selection = await select_pipeline(
            deployment.pipelines,
            alert=alert,
            metadata=metadata,
            explicit="diagnose" if target_stage == "diagnose" else None,
            selectors=deployment.pipeline_selectors,
        )
    else:
        # Do not accept a caller binding from a different deployment.
        selection = restore_selection(deployment.pipelines, selection.manifest())
    pipeline = selection.pipeline
    if target_stage == "diagnose" and not (
        len(pipeline.workflow.stages) == 1 and pipeline.workflow.stages[0].role == "rca"
    ):
        raise ValueError("target_stage='diagnose' requires a one-stage RCA workflow")
    async with incident_context(session_id, alert, deployment) as resources:
        state = initial_state(
            session_id=session_id,
            alert=alert,
            target_stage=target_stage,
            incident_dir=resources.incident_dir,
            shared_dir=resources.shared_dir,
            withheld_record_ids=withheld_record_ids,
        )
        state["runtime"] = RunState(
            selection=selection.manifest(),
            metadata=metadata or {},
            deployment_digest=deployment.runtime_digest,
        )
        save_runtime(deployment.storage.sessions, session_id, state["runtime"])
        journal = Journal(
            deployment.sink,
            session_id=session_id,
            stage="pipeline",
            sequence=deployment.sequences.for_session(session_id),
        )
        # Written before any role runs, and first. A campaign is reconstructible only if
        # the configuration it ran under is in the record beside its result — including
        # which schemas produced it, which is otherwise inferred from the date.
        journal.record(
            StageEvent(
                stage="pipeline",
                event="entered",
                detail=json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "selected_pipeline": selection.manifest(),
                        # Which environment supplied this run's transport, tools and
                        # overlays. The settings say which adapters are enabled; this
                        # says which one was actually bound, and the two differ for an
                        # adapter that drives its own episodes.
                        **(
                            {"environment": deployment.adapter.name}
                            if deployment.adapter is not None
                            else {}
                        ),
                        **deployment.settings.snapshot(),
                    },
                    sort_keys=True,
                ),
            )
        )

        deps = IncidentDeps(
            policy=RoutingPolicy.from_settings(deployment.settings),
            runs=_RunRecorder(deployment.storage),
            journal=journal,
            build_context=_context_builder(deployment, resources, pipeline=pipeline),
        )
        mode = resolve_mode(target_stage=target_stage, settings=deployment.settings)
        with span(
            "maver.incident",
            {
                "session.id": session_id,
                "alert.name": str(alert.get("alertname") or ""),
                "maver.mode": mode,
                "maver.target_stage": target_stage,
            },
        ) as incident_span:
            return await _run_mode(
                mode,
                state,
                deployment,
                deps,
                resources,
                session_id=session_id,
                journal=journal,
                target_stage=target_stage,
                fire_post_hooks=fire_post_hooks,
                finalize_run_status=finalize_run_status,
                incident_span=incident_span,
                pipeline=pipeline,
            )


async def _run_mode(
    mode: str,
    state: IncidentState,
    deployment: Deployment,
    deps: IncidentDeps,
    resources: IncidentResources,
    *,
    session_id: str,
    journal: Journal,
    target_stage: TargetStage,
    fire_post_hooks: bool,
    finalize_run_status: bool,
    incident_span: Any,
    pipeline: ResolvedPipeline | None = None,
) -> IncidentState:
    """Run the chosen mode and record what it came to."""
    nodes = _nodes(deployment, deps, resources, pipeline)
    try:
        final_state = (
            await _engine(deployment, deps, nodes, pipeline).run(state)
            if pipeline is not None
            else await MODES[mode](
                state,
                ModeContext(
                    deps=deps,
                    engine=_engine(deployment, deps, nodes),
                    nodes=nodes,
                ),
            )
        )
    except ApprovalPending as pending:
        # Not an error. Park the incident where it stopped and let the caller unwind
        # cleanly; nothing is persisted as terminal and no hook fires, because the
        # run has not finished — it is waiting.
        _park(
            deployment,
            pending,
            mode=mode,
            target_stage=target_stage,
            fire_post_hooks=fire_post_hooks,
            finalize_run_status=finalize_run_status,
        )
        raise

    record = (
        pipeline.resolve_outcome(final_state)
        if pipeline is not None
        else resolve_outcome(final_state)
    )
    # The span is opened by the caller and handed down for this. Each role annotates its
    # own with what it submitted; the incident's carried how it started and nothing about
    # how it ended, so a trace showed every incident alike and the one question a trace is
    # opened to answer had to be taken from the database instead.
    incident_span.set_attribute("maver.outcome", record.outcome.value)
    if record.lifecycle is not None:
        incident_span.set_attribute("maver.lifecycle", record.lifecycle.value)
    try:
        save_runtime(deployment.storage.sessions, session_id, final_state["runtime"])
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

    if (
        fire_post_hooks
        and target_stage == "full"
        and (pipeline is None or pipeline.workflow.post_hooks)
    ):
        # A diagnosis-only task has no remediation outcome to learn from. Never
        # distil it as a successful incident memory entry.
        await fire_hooks(record.outcome, session_id, workdir=deployment.settings.server.workdir)

    _record_completion(journal, final_state, record, deps.policy)
    return final_state


def _park(
    deployment: Deployment,
    pending: ApprovalPending,
    *,
    mode: str,
    target_stage: str,
    fire_post_hooks: bool,
    finalize_run_status: bool,
) -> None:
    """Record an incident that stopped for a decision: close its row, then checkpoint it.

    A row left running is indistinguishable from a role that hung, and the continuation
    opens its own row: what happened is one attempt that stopped for a decision and a
    second that carried on. Both entries to the pipeline park, and while only one of
    them closed the row, every incident needing a second decision left one open for good.

    Args:
        deployment: This process's resources.
        pending: The pause raised by the gate.
        mode: The shape the incident is running.
        target_stage: How far the caller asked the incident to go.
        fire_post_hooks: Whether the caller wants the post-incident hooks.
        finalize_run_status: Whether the caller wants the run status closed here.
    """
    if pending.state.get("runtime") is not None:
        save_runtime(deployment.storage.sessions, pending.session_id, pending.state["runtime"])
    if pending.run_id:
        deployment.storage.runs.finish_run(
            pending.run_id,
            status="awaiting_human",
            error=f"awaiting approval request {pending.request_id}",
        )
    save_checkpoint(
        deployment.storage.sessions,
        cast(IncidentState, pending.state),
        pending,
        sequence=deployment.sequences.for_session(pending.session_id).last,
        mode=mode,
        target_stage=target_stage,
        fire_post_hooks=fire_post_hooks,
        finalize_run_status=finalize_run_status,
    )


async def resume_after_decision(request_id: str, deployment: Deployment) -> str:
    """Continue an incident once the human it was waiting on has answered.

    The continuation lives here rather than in the role that raised. The prototype's
    remediating role imported the graph, the outcome resolver and the post-incident
    hooks so it could drive the pipeline itself, and that inversion is why the approval
    path was the least exercised part of the system.

    It runs on the sequential executor, starting at the role that was interrupted. The
    graph executor cannot be entered part-way without a checkpointer, and keeping
    persistence and resume in our own code is the point of not having one.

    Args:
        request_id: The request that was decided.
        deployment: This process's resources.

    Returns:
        A short status for the job log: why it could not continue, ``awaiting_human``
        when the continuation paused again, or the outcome the incident reached.
    """
    sessions = deployment.storage.sessions
    loaded = load_checkpoint(sessions, deployment.storage.approvals, request_id)
    if isinstance(loaded, str):
        return loaded
    checkpoint, record = loaded

    session = sessions.get_session(checkpoint.session_id)
    if session is None or session.run_status not in ("awaiting_human", "resuming"):
        # Answered twice, or the session moved on. Continuing now would re-dispatch a
        # write against a state nobody approved it for. Note the limitation honestly:
        # this is a read followed by a write, not a compare-and-set, so two processes
        # racing on one request are ordered by the request's own single decision rather
        # than by this check.
        return "already_resumed"
    state = restored_state(checkpoint)
    runtime = state["runtime"]
    if runtime.deployment_digest and runtime.deployment_digest != deployment.runtime_digest:
        raise PipelineChangedError(
            "deployment behavior changed since admission; restore its original configuration before resuming"
        )
    assert deployment.pipelines is not None
    selected = (
        restore_selection(deployment.pipelines, runtime.selection) if runtime.selection else None
    )
    pipeline = selected.pipeline if selected is not None else None
    workflow_stage = runtime.active_stage or stage_role(checkpoint.stage)
    sessions.set_run_status(
        checkpoint.session_id,
        RunStatus.RESUMING,
        current_stage=checkpoint.stage,
        checkpoint=checkpoint.model_dump(mode="json"),
        awaiting_request_id=request_id,
    )

    alert = dict(state.get("alert") or {})
    # The stage label may name a phase of a role; everything keyed by role needs the
    # role. The gate keeps the full label, which is what its write policy is written in.
    role = stage_role(checkpoint.stage)
    async with incident_context(checkpoint.session_id, alert, deployment) as resources:
        journal = Journal(
            deployment.sink,
            session_id=checkpoint.session_id,
            stage=checkpoint.stage,
            # Where the run that parked left off. A process that did not raise the pause
            # holds no allocator for this session, and one starting at 1 again would
            # give the continuation's events the positions the first pass already used.
            sequence=deployment.sequences.for_session(
                checkpoint.session_id, start=checkpoint.sequence
            ),
        )
        tools = tool_context_for(
            deployment,
            resources,
            role,
            checkpoint.session_id,
            journal,
            # Carried across the pause. A continuation that read the floor alone would
            # finish the run able to retrieve what the first half was denied.
            withheld=state.get("withheld_record_ids", ()),
        )
        replay_context = None
        if pipeline is not None:
            definition = pipeline.agents[workflow_stage]
            replay_context = _context_builder(
                deployment,
                resources,
                {workflow_stage: checkpoint.messages},
                resume_stage=checkpoint.stage,
                pipeline=pipeline,
            )(definition.spec, state, None, runtime.active_agent or checkpoint.run_id)
            tools = replay_context.tool_context
        content = await _replay(checkpoint, record, tools, deployment, journal, replay_context)
        for event in resume_events(checkpoint, record, result_chars=len(content)):
            journal.record(event)

        # Everything but the decided call, which the replay above answers. A turn that
        # paused part-way through its calls leaves the ones it never reached issued and
        # unanswered, and a server rejects a request carrying a tool_call_id with no
        # reply, so the continuation would die on the transcript rather than on
        # anything the operator did.
        transcript = close_dangling_tool_calls(
            [
                *checkpoint.messages,
                {
                    "role": "tool",
                    "tool_call_id": checkpoint.tool_call_id,
                    "name": checkpoint.tool_name,
                    "content": content,
                },
            ]
        )
        deps = IncidentDeps(
            policy=RoutingPolicy.from_settings(deployment.settings),
            runs=_RunRecorder(deployment.storage),
            journal=journal,
            build_context=_context_builder(
                deployment,
                resources,
                {workflow_stage: transcript},
                resume_stage=checkpoint.stage,
                pipeline=pipeline,
            ),
        )
        # The same composition the first pass ran under. Building the roles a second
        # way here would let a continuation re-enter a diagnosis without the second
        # opinion the arm is defined by, and the run would silently be two arms.
        nodes = _nodes(deployment, deps, resources, pipeline)
        try:
            # The shape the incident was running, not the graph by assumption. Always on
            # the sequential executor: the graph executor cannot be entered part-way
            # without a checkpointer, and not having one is the point.
            final_state = (
                await SequentialEngine(nodes, deps.policy, pipeline=pipeline).run(
                    state, start=workflow_stage
                )
                if pipeline is not None
                else await MODES[checkpoint.mode](
                    state,
                    ModeContext(
                        deps=deps,
                        engine=SequentialEngine(nodes, deps.policy),
                        nodes=nodes,
                        start=role,
                    ),
                )
            )
        except ApprovalPending as pending:
            # A second decision is needed. Re-park and leave the hooks alone: the
            # incident still has not finished.
            _park(
                deployment,
                pending,
                mode=checkpoint.mode,
                target_stage=checkpoint.target_stage,
                fire_post_hooks=checkpoint.fire_post_hooks,
                finalize_run_status=checkpoint.finalize_run_status,
            )
            return "awaiting_human"

        outcome = (
            pipeline.resolve_outcome(final_state)
            if pipeline is not None
            else resolve_outcome(final_state)
        )
        try:
            save_runtime(sessions, checkpoint.session_id, final_state["runtime"])
            persist_outcome(
                final_state,
                outcome,
                sessions,
                finalize_run_status=checkpoint.finalize_run_status,
            )
        except Exception:
            _log.exception("persisting the resumed outcome failed; continuing from memory")
        # Once, at the end, and only if the caller that parked wanted them. An interim
        # result recorded mid-continuation would be read as terminal, and a driver that
        # owns the run's ending must not have it closed from underneath.
        if (
            checkpoint.fire_post_hooks
            and checkpoint.target_stage == "full"
            and (pipeline is None or pipeline.workflow.post_hooks)
        ):
            await fire_hooks(
                outcome.outcome,
                checkpoint.session_id,
                workdir=deployment.settings.server.workdir,
            )
        _record_completion(journal, final_state, outcome, deps.policy)
        return outcome.outcome.value


async def _replay(
    checkpoint: ApprovalCheckpoint,
    record: ApprovalRecord,
    tools: ToolContext,
    deployment: Deployment,
    journal: Journal,
    context: RoleContext | None = None,
) -> str:
    """Produce the result of the call the incident was paused on.

    Through the same middleware the live path dispatches through, because this is the
    one tool call in the system that an operator has personally approved and therefore
    the one that least deserves a shortcut around the gate. Calling the bound tool
    directly left the grant with nothing to present itself to: the gate's check that the
    arguments approved are the arguments executed could not run, so an edited checkpoint
    would have replayed whatever it now said, under a transcript showing an approval.
    """
    if record.status != "approved":
        return denial_result(record)

    toolset = (
        context.toolset
        if context is not None
        else build_toolset(stage_role(checkpoint.stage), tools, extras=deployment.tool_extras)
    )
    bound = toolset.dispatch.get(checkpoint.tool_name)
    if bound is None:
        # Approved, but the tool is not available in this configuration any more.
        # Reported as a refusal rather than raised: the role can say so and move on.
        return f"error: {checkpoint.tool_name} is not available in this configuration"

    async def dispatch(request: ToolRequest) -> ToolResult:
        return ToolResult(
            tool_call_id=request.call_id,
            name=request.name,
            content=str(await bound(**request.arguments)),
        )

    stack = (
        context.middleware
        if context is not None
        else MiddlewareStack(
            [
                WriteGate(
                    authorities=deployment.authorities,
                    settings=deployment.settings.authority,
                    tools=deployment.settings.tools,
                    journal=journal,
                ),
                *deployment.hooks,
            ],
            on_error=record_hook_failures(journal),
        )
    )
    # The grant is what the operator approved, read from the request they were shown;
    # the call is what the checkpoint says to run. Sourcing both from the checkpoint
    # made them agree by construction, so the gate's comparison could not fail and the
    # check meant nothing.
    with approval_granted(
        record.request_id,
        tool=str(record.tool_name or checkpoint.tool_name),
        arguments=dict(record.payload.get("arguments") or {}),
    ):
        result = await stack.run_tool(
            ToolRequest(
                call_id=checkpoint.tool_call_id,
                name=checkpoint.tool_name,
                arguments=dict(checkpoint.tool_arguments),
                # The label, not the role: the write policy names the phase that may
                # write, and resolving it here would offer the gate a stage it does not
                # recognise.
                role=checkpoint.stage,
                session_id=checkpoint.session_id,
            ),
            dispatch,
        )
    return result.content


def tool_context_for(
    deployment: Deployment,
    resources: IncidentResources,
    role: str,
    session_id: str,
    journal: Journal,
    *,
    retrieving: bool | None = None,
    withheld: Sequence[str] = (),
) -> ToolContext:
    """One role's view of the environment. Shared by the runner and the continuation.

    ``withheld`` is the incident's own exclusion set, which is unioned with the
    process-wide floor rather than replacing it.
    """
    return ToolContext(
        transport=resources.transport,
        workspace=resources.workspace,
        role=role,
        session_id=session_id,
        journal=journal,
        settings=deployment.settings.tools,
        # The gate's own mode, not a second knob: the proxy route puts a real kubectl on
        # a role's PATH, where a write never reaches the gate, so the shim there refuses
        # one unless this deployment approves writes unattended.
        authority_mode=deployment.settings.authority.mode,
        kubectl_binary=deployment.settings.transport.kubectl_binary,
        widened_write_stages=deployment.settings.authority.additional_write_stages,
        scope=resources.scope,
        evidence=deployment.evidence,
        # Two conditions, not one. A configuration with the store enabled and roles kept
        # retrieval-free is real: the gated second pass reads precedents in the harness
        # while both of its diagnosis runs hold no retrieval tool and receive no memory
        # prompt block. Withholding the store here is what makes the toolset drop those
        # tools, which in turn is what selects the memory-off prompt.
        memory=deployment.storage.retriever if _may_retrieve(deployment, retrieving) else None,
        sessions=deployment.storage.sessions,
        runs=deployment.storage.runs,
        # The floor the setting promises, plus whatever this incident was opened with.
        # The floor reached the gated second pass and no role, so a generalisation arm
        # withheld the evaluated incident's own record from the harness query while
        # every role could still search it up and read it back.
        memory_exclude=tuple(
            sorted(frozenset(deployment.settings.memory.exclude_record_ids) | frozenset(withheld))
        ),
    )


def withheld_records(deployment: Deployment, state: IncidentState) -> frozenset[str]:
    """Every corpus record this incident may not retrieve.

    Two sources, unioned. The process-wide floor is a deployment's own promise about the
    corpus; the incident's set is what an environment adapter withheld for this episode,
    and neither is the other's to override.

    Args:
        deployment: The process's shared resources.
        state: The incident state.

    Returns:
        The session ids to withhold.
    """
    return frozenset(deployment.settings.memory.exclude_record_ids) | frozenset(
        state.get("withheld_record_ids") or ()
    )


def _may_retrieve(deployment: Deployment, override: bool | None) -> bool:
    """Whether this run may reach the experience store.

    Two conditions, not one. A configuration with the store enabled and roles kept
    retrieval-free is real: the gated second pass reads precedents in the harness while
    both of its diagnosis runs hold no retrieval tool and receive no memory prompt
    block. A caller may narrow it further, never widen it.
    """
    configured = deployment.settings.memory.enabled and deployment.settings.memory.role_retrieval
    return configured if override is None else (configured and override)


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
