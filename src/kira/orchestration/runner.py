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

from kira.adapters.base import AdapterTool
from kira.audit.journal import EventSink, Journal, NullSink
from kira.contracts.events import StageEvent
from kira.contracts.lifecycle import RunStatus
from kira.contracts.routing import HandBack
from kira.contracts.state import IncidentState, TargetStage
from kira.llm.sampling import (
    SamplingResolution,
    assert_sampling_is_pinned,
    resolve_sampling,
)
from kira.llm.types import Completion, ToolResult
from kira.memory.factory import StorageStack
from kira.memory.ports import AgentRunRecord, AgentRunStatus, ApprovalRecord
from kira.middleware.authority import ApprovalPending, Authority, AutoApprove
from kira.middleware.base import AgentHook, MiddlewareStack, ToolRequest
from kira.middleware.human_approval import approval_granted
from kira.middleware.write_gate import WriteGate
from kira.observability.tracing import span
from kira.observability.usage import RecordingCompletion
from kira.orchestration.budget import RetryBudget
from kira.orchestration.engine import Engine, LangGraphEngine, SequentialEngine, StageHook
from kira.orchestration.graph import build_graph
from kira.orchestration.hooks import fire_hooks
from kira.orchestration.modes import MODES, ModeContext, resolve_mode
from kira.orchestration.nodes import ContextBuilder, IncidentDeps, NodeFn, make_node
from kira.orchestration.outcomes import OutcomeRecord, persist_outcome, resolve_outcome
from kira.orchestration.resume import (
    ApprovalCheckpoint,
    denial_result,
    load_checkpoint,
    restored_state,
    resume_events,
    save_checkpoint,
)
from kira.orchestration.routing import RoutingPolicy
from kira.orchestration.state import initial_state
from kira.roles.base import RoleContext, RoleSpec, StateDelta, stage_role
from kira.roles.registry import get_role, registered_roles
from kira.secondpass.audit import JournalEmitter
from kira.secondpass.runner import SecondPass
from kira.secondpass.strategies import policy_for
from kira.secondpass.types import (
    DiagnosisRun,
    SecondPassContext,
    SecondPassPolicy,
    SecondPassResult,
)
from kira.settings.authority import Mode
from kira.settings.profiles import ModelProfile
from kira.settings.root import Settings
from kira.tools.context import EvidencePolicy, NullEvidencePolicy, ToolContext
from kira.tools.scope import IncidentScope
from kira.tools.snapshot import namespace_snapshot
from kira.tools.toolsets import build_toolset
from kira.tools.transport.base import ToolTransport
from kira.tools.workspace import SessionWorkspace

__all__ = [
    "Deployment",
    "IncidentResources",
    "incident_context",
    "resume_after_decision",
    "run_incident",
    "tool_context_for",
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
    sink: EventSink = field(default_factory=NullSink)
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    """One cluster writer per process. Held here rather than in a module global so two
    deployments in one process are genuinely independent, which is what makes the
    serialisation testable."""
    sampling: SamplingResolution = field(init=False)
    """The profile's set resolved against settings, once, for the whole process."""

    def __post_init__(self) -> None:
        """Resolve the sampling set once, and refuse a deployment that cannot send it."""
        # Resolved and checked here because this is the composition root: every entry
        # point builds a Deployment, so a set that cannot be sent fails at startup
        # rather than at the first call, or — the case that actually happened — not at
        # all, with the run completing on a configuration nobody could reconstruct.
        resolution = resolve_sampling(self.profile, self.settings)
        assert_sampling_is_pinned(self.profile, resolution)
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
crash in every count taken downstream. See :mod:`kira.orchestration.outcomes`."""


def _engine(deployment: Deployment, deps: IncidentDeps, nodes: Mapping[str, NodeFn]) -> Engine:
    """Build the configured executor over the nodes it is given.

    Given rather than built here, so the executor and every mode run the same objects.
    Two compositions of one incident are equivalent today and one refactor away from
    not being.
    """
    if deployment.settings.pipeline.engine == "sequential":
        return SequentialEngine(nodes, deps.policy)
    return LangGraphEngine(build_graph(nodes, deps.policy))


def _nodes(
    deployment: Deployment, deps: IncidentDeps, resources: IncidentResources
) -> dict[str, NodeFn]:
    """The graph's roles, with the diagnosing one composed if a second pass is configured.

    The mechanism wraps the diagnosis rather than living inside it. Both of its runs are
    ordinary role runs — same wrapper, same run rows, same audit — and what differs is
    only what they are told and whether they may retrieve. A role that knew it was being
    second-guessed would be a different role in the arm that does it.
    """
    nodes = {name: make_node(get_role(name), deps) for name in GRAPH_ROLES}
    policy = policy_for(deployment.settings.secondpass, deployment.settings.memory)
    if policy is not None:
        nodes["rca"] = _second_pass_node(deployment, deps, resources, policy)
    return nodes


def _second_pass_node(
    deployment: Deployment,
    deps: IncidentDeps,
    resources: IncidentResources,
    policy: SecondPassPolicy,
) -> NodeFn:
    """Run the gated second opinion in place of a single diagnosis."""

    async def diagnose(
        state: IncidentState, *, brief: str | None, retrieval: bool, run_label: str
    ) -> DiagnosisRun:
        node = make_node(
            get_role("rca"),
            replace(
                deps,
                build_context=_context_builder(
                    deployment,
                    resources,
                    None,
                    assignment=brief or "",
                    retrieval=retrieval,
                    run_label=run_label,
                ),
            ),
        )
        delta = dict(await node(state))
        run_ids = delta.get("run_ids") or []
        return DiagnosisRun(
            artifact=delta.get("root_cause"),
            run_id=str(run_ids[-1]) if run_ids else None,
            messages=tuple(delta.get("messages", {}).get("rca", ())),
            trace_summaries=tuple(delta.get("trace_summaries", ())),
        )

    async def node(state: IncidentState) -> StateDelta:
        journal = Journal(deployment.sink, session_id=state.get("session_id", ""), stage="rca")
        tools = tool_context_for(deployment, resources, "rca", state.get("session_id", ""), journal)

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
                withheld_record_ids=frozenset(deployment.settings.memory.exclude_record_ids),
                emit=JournalEmitter(journal, policy=policy.name),
            )
        )
        return _second_pass_delta(state, result)

    return node


def _second_pass_delta(state: IncidentState, result: SecondPassResult) -> StateDelta:
    """Merge both diagnosis runs into the state, not only the winner.

    The prototype returned one of the two states, so the losing run's identifier never
    entered the ordered path: both rows existed in the database and the chain that
    linked them did not.
    """
    runs = [run for run in (result.first, result.second) if run is not None]
    delta: dict[str, Any] = {
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
    if result.chosen is None:
        delta["error"] = f"rca failed: {result.skipped or 'no diagnosis'}"
    return delta


def _context_builder(
    deployment: Deployment,
    resources: IncidentResources,
    on_stage: StageHook | None,
    resume_for: dict[str, list[dict[str, Any]]] | None = None,
    assignment: str = "",
    retrieval: bool | None = None,
    run_label: str = "",
) -> ContextBuilder:
    """Assemble a role's context from the incident's own resources.

    ``resume_for`` carries the transcript a paused role should open on, keyed by that
    role. It is consumed on the first entry: a role re-entered later in the same run is
    starting a fresh cycle, not continuing the one that was interrupted.
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
        journal = Journal(
            deployment.sink,
            session_id=state.get("session_id", ""),
            stage=spec.name,
            run_id=run_id,
        )
        tools = tool_context_for(
            deployment,
            resources,
            spec.name,
            state.get("session_id", ""),
            journal,
            retrieving=retrieving,
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
            completion=RecordingCompletion(
                deployment.completion, journal, deployment.profile, deployment.sampling
            ),
            profile=deployment.profile,
            middleware=MiddlewareStack(
                [
                    WriteGate(
                        authorities=deployment.authorities,
                        settings=deployment.settings.authority,
                        journal=journal,
                    ),
                    *deployment.hooks,
                ]
            ),
            write_lock=deployment.write_lock,
            toolset=build_toolset(spec.name, tools, extras=deployment.tool_extras),
            tool_context=tools,
            journal=journal,
            run_id=run_id,
            session_id=state.get("session_id", ""),
            memory=tools.memory,
            milestone=milestone,
            assignment=assignment,
            run_label=run_label,
            resume=pending_resume.pop(spec.name, None),
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
        with span(
            "kira.incident",
            {
                "session.id": session_id,
                "alert.name": str(alert.get("alertname") or ""),
                "kira.mode": mode,
                "kira.target_stage": target_stage,
            },
        ) as incident_span:
            return await _run_mode(
                mode,
                state,
                deployment,
                deps,
                resources,
                on_stage=on_stage,
                session_id=session_id,
                journal=journal,
                target_stage=target_stage,
                fire_post_hooks=fire_post_hooks,
                finalize_run_status=finalize_run_status,
                incident_span=incident_span,
            )


async def _run_mode(
    mode: str,
    state: IncidentState,
    deployment: Deployment,
    deps: IncidentDeps,
    resources: IncidentResources,
    *,
    on_stage: StageHook | None,
    session_id: str,
    journal: Journal,
    target_stage: TargetStage,
    fire_post_hooks: bool,
    finalize_run_status: bool,
    incident_span: Any,
) -> IncidentState:
    """Run the chosen mode and record what it came to."""
    nodes = _nodes(deployment, deps, resources)
    try:
        final_state = await MODES[mode](
            state,
            ModeContext(
                deps=deps,
                engine=_engine(deployment, deps, nodes),
                nodes=nodes,
                on_stage=on_stage,
            ),
        )
    except ApprovalPending as pending:
        # Not an error. Park the incident where it stopped and let the caller unwind
        # cleanly; nothing is persisted as terminal and no hook fires, because the
        # run has not finished — it is waiting.
        _park_run(deployment, pending)
        save_checkpoint(
            deployment.storage.sessions,
            cast(IncidentState, pending.state),
            pending,
            mode=mode,
            target_stage=target_stage,
            fire_post_hooks=fire_post_hooks,
            finalize_run_status=finalize_run_status,
        )
        raise

    record = resolve_outcome(final_state)
    # The span is opened by the caller and handed down for this. Each role annotates its
    # own with what it submitted; the incident's carried how it started and nothing about
    # how it ended, so a trace showed every incident alike and the one question a trace is
    # opened to answer had to be taken from the database instead.
    incident_span.set_attribute("kira.outcome", record.outcome.value)
    if record.lifecycle is not None:
        incident_span.set_attribute("kira.lifecycle", record.lifecycle.value)
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


def _park_run(deployment: Deployment, pending: ApprovalPending) -> None:
    """Close the paused role's row as waiting, rather than leaving it open.

    A row left running is indistinguishable from a role that hung, and the continuation
    opens its own row: what happened is one attempt that stopped for a decision and a
    second that carried on.
    """
    if not pending.run_id:
        return
    deployment.storage.runs.finish_run(
        pending.run_id,
        status="awaiting_human",
        error=f"awaiting approval request {pending.request_id}",
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
    sessions.set_run_status(
        checkpoint.session_id,
        RunStatus.RESUMING,
        current_stage=checkpoint.stage,
        checkpoint=checkpoint.model_dump(mode="json"),
        awaiting_request_id=request_id,
    )

    state = restored_state(checkpoint)
    alert = dict(state.get("alert") or {})
    # The stage label may name a phase of a role; everything keyed by role needs the
    # role. The gate keeps the full label, which is what its write policy is written in.
    role = stage_role(checkpoint.stage)
    async with incident_context(checkpoint.session_id, alert, deployment) as resources:
        journal = Journal(deployment.sink, session_id=checkpoint.session_id, stage=checkpoint.stage)
        tools = tool_context_for(deployment, resources, role, checkpoint.session_id, journal)
        content = await _replay(checkpoint, record, tools, deployment, journal)
        for event in resume_events(checkpoint, record, result_chars=len(content)):
            journal.record(event)

        transcript = [
            *checkpoint.messages,
            {
                "role": "tool",
                "tool_call_id": checkpoint.tool_call_id,
                "name": checkpoint.tool_name,
                "content": content,
            },
        ]
        deps = IncidentDeps(
            policy=RoutingPolicy.from_settings(deployment.settings),
            runs=_RunRecorder(deployment.storage),
            journal=journal,
            build_context=_context_builder(deployment, resources, None, {role: transcript}),
        )
        # The same composition the first pass ran under. Building the roles a second
        # way here would let a continuation re-enter a diagnosis without the second
        # opinion the arm is defined by, and the run would silently be two arms.
        nodes = _nodes(deployment, deps, resources)
        try:
            # The shape the incident was running, not the graph by assumption. Always on
            # the sequential executor: the graph executor cannot be entered part-way
            # without a checkpointer, and not having one is the point.
            final_state = await MODES[checkpoint.mode](
                state,
                ModeContext(
                    deps=deps,
                    engine=SequentialEngine(nodes, deps.policy),
                    nodes=nodes,
                    start=role,
                ),
            )
        except ApprovalPending as pending:
            # A second decision is needed. Re-park and leave the hooks alone: the
            # incident still has not finished.
            save_checkpoint(
                sessions,
                cast(IncidentState, pending.state),
                pending,
                mode=checkpoint.mode,
                target_stage=checkpoint.target_stage,
                fire_post_hooks=checkpoint.fire_post_hooks,
                finalize_run_status=checkpoint.finalize_run_status,
            )
            return "awaiting_human"

        outcome = resolve_outcome(final_state)
        try:
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
        if checkpoint.fire_post_hooks and checkpoint.target_stage == "full":
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

    toolset = build_toolset(stage_role(checkpoint.stage), tools, extras=deployment.tool_extras)
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

    stack = MiddlewareStack(
        [
            WriteGate(
                authorities=deployment.authorities,
                settings=deployment.settings.authority,
                journal=journal,
            ),
            *deployment.hooks,
        ]
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
) -> ToolContext:
    """One role's view of the environment. Shared by the runner and the continuation."""
    return ToolContext(
        transport=resources.transport,
        workspace=resources.workspace,
        role=role,
        session_id=session_id,
        journal=journal,
        settings=deployment.settings.tools,
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
        # The floor the setting promises. It reached the gated second pass and no role,
        # so a generalisation arm withheld the evaluated incident's own record from the
        # harness query while every role could still search it up and read it back.
        memory_exclude=tuple(sorted(deployment.settings.memory.exclude_record_ids)),
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
