"""What the system stores, and the interfaces behind which it is stored.

Four ports rather than one, because four different things consume them and a single
store interface would force every backend to implement all of it. A deployment that
never uses human approval should not have to implement approval persistence to run, and
whether it does is something the startup check can now actually ask.

The records are plain dataclasses. They cross the port boundary in both directions and
are the only shape a backend has to agree on.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from maver.contracts.lifecycle import LifecycleState, RunStatus

__all__ = [
    "AgentRunEventDraft",
    "AgentRunEventRecord",
    "AgentRunRecord",
    "AgentRunStatus",
    "ApprovalRecord",
    "ApprovalStatus",
    "ApprovalStore",
    "DecisionStatus",
    "QuarantineRecord",
    "RunStore",
    "SearchHit",
    "SearchScope",
    "SessionMemory",
    "SessionRecord",
    "SessionStore",
    "TextSearch",
    "VectorSearch",
]

SearchScope = Literal["evidence", "lesson"]
"""Which projection a query is matched against.

Evidence is what the incident looked like; the lesson is what was concluded. They are
retrieved separately because a later incident resembles a past one by its symptoms long
before it resembles it by its cause.
"""

ApprovalStatus = Literal["pending", "approved", "denied", "answered", "timed_out", "cancelled"]

DecisionStatus = Literal["approved", "denied", "answered", "timed_out", "cancelled"]
"""What a pending request can become.

:data:`ApprovalStatus` minus ``pending``, so that a decision *into* the state a request
is already in cannot be expressed. There is no such thing as deciding to keep waiting.
"""

AgentRunStatus = Literal["running", "succeeded", "failed", "halted", "awaiting_human", "cancelled"]


@dataclass
class SessionRecord:
    """One incident, from the alert that raised it to the artefacts it produced.

    The prototype also carried ``transcript`` and ``agent_trace_summaries``. Nothing
    ever wrote the first and no method here can write the second, so both are dropped:
    a field that is always empty is worse than absent, because it reads as "this run
    produced no conversation" rather than "nobody recorded one". What each role did is
    on :class:`AgentRunRecord.trace_summary`, reachable through :meth:`RunStore.runs_for`.
    """

    session_id: str
    started_at: str
    ended_at: str | None = None
    state: str = LifecycleState.PENDING.value
    run_status: str = RunStatus.PENDING.value
    current_stage: str | None = None
    checkpoint: dict[str, Any] | None = None
    """Enough state to resume. Written before a run parks on a human."""
    awaiting_request_id: str | None = None
    alert: dict[str, Any] = field(default_factory=dict)
    root_cause: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    verify_result: dict[str, Any] | None = None
    """The recovery verdict, when verification ran.

    Here as well as on the run row because this is the artefact that decided the
    incident's outcome, and a reader holding only the session record would otherwise be
    able to see the claim that a fix was applied but not the check that disproved it.
    """
    distilled: bool = False
    notes: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    """Private pipeline state, isolated from model-visible artifacts and alert input."""


@dataclass
class QuarantineRecord:
    """A Kafka record that could not be admitted, preserved for explicit retry."""

    record_id: str
    source: str
    key: str
    raw_base64: str
    reason: str
    created_at: str
    retry_session_id: str | None = None


@dataclass
class ApprovalRecord:
    """One request put to a human, and what they said."""

    request_id: str
    session_id: str
    kind: str
    stage: str
    status: str
    summary: str
    payload: dict[str, Any]
    tool_call_id: str | None = None
    tool_name: str | None = None
    prompt: str | None = None
    response: dict[str, Any] | None = None
    created_at: str = ""
    expires_at: str | None = None
    decided_at: str | None = None
    decided_by: str = "generic human"
    reason: str | None = None


@dataclass
class AgentRunRecord:
    """One role's execution within one incident."""

    agent_run_id: str
    pipeline_id: str | None
    agent_name: str
    attempt: int
    status: str
    started_at: str
    ended_at: str | None = None
    output: dict[str, Any] | None = None
    trace_summary: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunEventDraft:
    """An event on its way in, before the store has given it a place in the order.

    Identical to :class:`AgentRunEventRecord` without ``seq``, because the sequence is
    the store's to allocate and a caller that could set it could also collide with
    another caller writing to the same run.
    """

    event_id: str
    agent_run_id: str
    pipeline_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: str
    role: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class AgentRunEventRecord:
    """One ordered event inside a role's execution, as stored."""

    event_id: str
    agent_run_id: str
    pipeline_id: str | None
    seq: int
    event_type: str
    payload: dict[str, Any]
    created_at: str
    role: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class SearchHit:
    """One retrieved record.

    ``score`` is set by whatever ranked it. In the prototype no backend ever assigned
    it: the typed hit was converted to an untyped dict at the fusion boundary and the
    score lived there instead, so everything downstream of retrieval consumed dicts and
    the dataclass documented a field that was always None.
    """

    session_id: str
    summary: str
    """The projection this hit was matched on, evidence or lesson."""
    metadata: dict[str, Any] = field(default_factory=dict)
    incident_summary: str | None = None
    scope: SearchScope = "evidence"
    evidence_text: str = ""
    lesson_text: str = ""
    score: float | None = None


@dataclass
class SessionMemory:
    """Everything stored about one past incident, read whole."""

    session_id: str
    evidence_text: str
    lesson_text: str
    incident_summary: str
    lifecycle_state: str
    fault_category: str
    alert_name: str
    namespace: str
    affected_resource: str
    verification_verdict: str = ""
    evaluation_label: str = ""
    evaluation_source: str = ""
    evaluation_diagnosis_success: bool | None = None
    evaluation_diagnosis_score: float | None = None
    evaluation_mitigation_success: bool | None = None
    corrected_cause: str = ""
    """Deliberately absent from the lesson projection.

    It reaches a reader only by opening the record directly. A corrected cause states
    the answer, so putting it in the text that retrieval returns would let a later
    incident read a conclusion it had not reached.
    """
    labels: dict[str, str] = field(default_factory=dict)
    """The environment's own keys for this record, for the environment's own use.

    Written from the underscore-prefixed keys of the session's stored signal, which are
    the ones every prompt builder drops. They are not part of what a record *says*:
    nothing renders them into a projection, a prompt or a tool result, and no backend
    indexes them, so a value that is a ground-truth label under an evaluation harness
    cannot reach a model through them.

    They exist so that an adapter can ask which records came from one of its own
    episodes — :meth:`TextSearch.session_ids_with_label` is the only read — without the
    core having to know what the key means.
    """


@runtime_checkable
class SessionStore(Protocol):
    """Persistence for incidents and their artefacts."""

    def create_session(self, alert: dict[str, Any]) -> SessionRecord:
        """Open a session for an incoming alert.

        Args:
            alert: The alert payload.

        Returns:
            The created record.
        """
        ...

    def open_idempotent_session(
        self,
        source: str,
        key: str,
        request_hash: str,
        alert: dict[str, Any],
        runtime: dict[str, Any],
    ) -> tuple[SessionRecord, bool]:
        """Atomically claim an intake key and initialize its running session.

        Return the stored session and whether this call created it. A reused key with
        different request content raises ValueError. Source namespaces HTTP and queues.
        """
        ...

    def get_intake_receipt(
        self,
        source: str,
        key: str,
        request_hash: str,
    ) -> SessionRecord | None:
        """Find a prior admission, rejecting reuse with different request content."""
        ...

    def bind_intake_receipt(
        self,
        source: str,
        key: str,
        request_hash: str,
        session_id: str,
    ) -> SessionRecord:
        """Attach another delivery key to an already running session atomically."""
        ...

    def quarantine(self, source: str, key: str, raw_base64: str, reason: str) -> QuarantineRecord:
        """Persist or return a rejected delivery before acknowledging it."""
        ...

    def list_quarantine(self) -> list[QuarantineRecord]:
        """List quarantined deliveries, oldest first."""
        ...

    def get_quarantine(self, record_id: str) -> QuarantineRecord | None:
        """Read a quarantined delivery by id."""
        ...

    def mark_quarantine_retried(self, record_id: str, session_id: str) -> None:
        """Keep the admission reached by an explicit retry."""
        ...

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Read one session.

        Args:
            session_id: Which session.

        Returns:
            The record, or None when it does not exist.
        """
        ...

    def set_runtime(self, session_id: str, runtime: dict[str, Any]) -> None:
        """Replace the private run state without changing alert, artifacts or status.

        Args:
            session_id: The incident whose state is saved.
            runtime: JSON-safe selection, metadata and isolated agent state.
        """
        ...

    def awaiting_sessions(self) -> list[SessionRecord]:
        """Read every session parked on a human decision.

        The one question the approval sweep has to ask, from the side that is bounded.
        Nothing prunes decided requests, so scanning them grew a little longer every
        time somebody answered one; the sessions parked right now are bounded by the
        incidents in flight.

        Returns:
            The parked sessions, oldest first.
        """
        ...

    def running_sessions(self) -> list[SessionRecord]:
        """Read sessions left running by a prior process, oldest first."""
        ...

    def reconciliation_sessions(self) -> list[SessionRecord]:
        """Read sessions awaiting an operator's interrupted-write decision."""
        ...

    def get_states(self, session_ids: Sequence[str]) -> dict[str, str]:
        """Read the lifecycle state of many sessions at once.

        Retrieval needs the *current* state of every candidate, and the state stored
        beside a record goes stale the moment the run that produced it is re-graded.

        Args:
            session_ids: Which sessions.

        Returns:
            Session id to lifecycle state, omitting sessions that do not exist.
        """
        ...

    def update_artefacts(
        self,
        session_id: str,
        *,
        root_cause: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        verify_result: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Attach what a role produced to its session.

        Partial. A ``None`` leaves that artefact unchanged, so Planning attaching a
        plan cannot erase the diagnosis it was given. Contrast :meth:`set_run_status`,
        where a ``None`` clears.

        Args:
            session_id: Which session.
            root_cause: The diagnosis, when one was submitted.
            plan: The remediation plan, when one was submitted.
            execution_result: What execution did, when it ran.
            verify_result: The recovery verdict, when verification ran.
            notes: Free text for an operator.
        """
        ...

    def set_state(self, session_id: str, state: LifecycleState) -> None:
        """Record how far a session's claim has been confirmed.

        Checked against :func:`maver.memory.lifecycle.check_transition` rather than
        written unconditionally, and checked here rather than in any one caller, because
        which verdict may follow which is a property of the stored record: an operator
        route overturning a verdict has to obey it as much as the pipeline does, and
        neither should have to remember to. Writing the state a session already holds is
        a no-op, so a job retried after a crash is safe to run again.

        Args:
            session_id: Which session.
            state: The new lifecycle state.

        Raises:
            IllegalTransition: When the record's current state cannot become this one.
                Nothing is written. A session that does not exist is not an error, for
                the same reason it is not one anywhere else here.
        """
        ...

    def set_run_status(
        self,
        session_id: str,
        status: RunStatus,
        *,
        current_stage: str | None = None,
        checkpoint: dict[str, Any] | None = None,
        awaiting_request_id: str | None = None,
    ) -> None:
        """Record where a run is, and enough to resume it.

        All four values move together, in one write. A ``None`` therefore clears rather
        than leaving alone, which is the opposite of :meth:`update_artefacts` and is the
        point: a reader must never see ``awaiting_human`` without the checkpoint that
        belongs to it, and a run that has resumed must not still advertise the
        checkpoint it has already consumed.

        ``ended_at`` is stamped the first time a terminal status (``completed``,
        ``failed``, ``cancelled``) is recorded, and never moves again.

        Args:
            session_id: Which session.
            status: The operational status.
            current_stage: The role that was running, cleared when None.
            checkpoint: State to resume from, cleared when None.
            awaiting_request_id: The approval being waited on, cleared when None.
        """
        ...

    def decide_reconciliation(self, session_id: str, *, notes: str) -> bool:
        """Atomically close a reconciliation case, retaining its checkpoint and evidence."""
        ...

    def reopen_reconciliation(self, session_id: str, *, notes: str) -> bool:
        """Repark a claimed case when opening its replacement incident failed."""
        ...

    def mark_distilled(self, session_id: str) -> None:
        """Record that a session's lesson has been written to the corpus.

        Args:
            session_id: Which session.
        """
        ...


@runtime_checkable
class ApprovalStore(Protocol):
    """Persistence for requests put to a human.

    Separate from :class:`SessionStore` so that a deployment running unattended does
    not have to implement it, and so the startup check for ``authority.mode == prompt``
    can ask a precise question instead of assuming.
    """

    def create_request(self, record: ApprovalRecord) -> ApprovalRecord:
        """Persist a pending request.

        Args:
            record: The request.

        Returns:
            The stored record.
        """
        ...

    def get_request(self, request_id: str) -> ApprovalRecord | None:
        """Read one request.

        Args:
            request_id: Which request.

        Returns:
            The record, or None.
        """
        ...

    def list_requests(self, *, status: ApprovalStatus | None = None) -> list[ApprovalRecord]:
        """List requests, newest first.

        Args:
            status: Only requests in this status, or all of them.

        Returns:
            The matching records.
        """
        ...

    def decide(
        self,
        request_id: str,
        *,
        status: DecisionStatus,
        decided_by: str,
        reason: str | None = None,
    ) -> ApprovalRecord | None:
        """Record a decision, if the request is still pending.

        Returning None on a request that is already decided is what makes a late
        decision and a duplicate one harmless rather than a second dispatch.

        Args:
            request_id: Which request.
            status: The decision.
            decided_by: Who decided.
            reason: Why.

        Returns:
            The updated record, or None when it was not pending.
        """
        ...

    def expire_due(self, *, now: str) -> list[ApprovalRecord]:
        """Time out every request past its expiry.

        Timestamps are ISO-8601 UTC and must be compared as parsed instants, not as
        strings: ``"...00Z"`` and ``"...00+00:00"`` are the same moment and sort
        differently, because ``"+"`` orders before ``"Z"``. A request whose
        ``expires_at`` cannot be parsed stays pending, since a malformed expiry is not
        evidence that anything has expired.

        Only ``decided_at`` is set, to ``now``. Nobody decided, so ``decided_by`` and
        ``reason`` are left as they were.

        Args:
            now: The current timestamp.

        Returns:
            The records that expired.
        """
        ...


@runtime_checkable
class RunStore(Protocol):
    """Persistence for what each role did, in order."""

    def start_run(self, record: AgentRunRecord) -> None:
        """Record that a role started.

        Args:
            record: The run.
        """
        ...

    def finish_run(
        self,
        agent_run_id: str,
        *,
        status: AgentRunStatus,
        output: dict[str, Any] | None = None,
        trace_summary: dict[str, Any] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a role finished.

        Args:
            agent_run_id: Which run.
            status: How it ended.
            output: What it produced.
            trace_summary: A condensed account of what it did.
            error: Why it failed, when it did.
            metadata: What the role recorded about its own run. Merged key by key into
                what :meth:`start_run` wrote, alone among these arguments, because the
                two writes describe different things and the earlier one — the edge this
                run was reached by — is not the later one's to discard. Most of what a
                role can say about itself is only knowable at the end anyway: a run id
                has to exist before the role's context does, so nothing about what the
                model was actually offered can be recorded at the start.
        """
        ...

    def get_run(self, agent_run_id: str) -> AgentRunRecord | None:
        """Read one run.

        Args:
            agent_run_id: Which run.

        Returns:
            The record, or None when it does not exist.
        """
        ...

    def runs_for(self, session_id: str) -> list[AgentRunRecord]:
        """Read every role's run within one incident, oldest first.

        Matched on ``AgentRunRecord.pipeline_id``, which is the incident a run belongs
        to. This is the only way to reach what a role produced: ``output``,
        ``trace_summary``, ``error`` and ``metadata`` are otherwise write-only.

        Args:
            session_id: Which incident.

        Returns:
            The runs, ordered by start time.
        """
        ...

    def append_event(self, event: AgentRunEventDraft) -> AgentRunEventRecord:
        """Append one event, giving it the next place in the run's order.

        The sequence is allocated here, inside the write, and is monotonic per
        ``agent_run_id`` starting at 1. That is why the draft has no ``seq``: a caller
        that could choose one could also collide with another caller writing to the
        same run.

        Note that :mod:`maver.audit.journal` stamps a sequence of its own, into the
        payload, per ``session_id`` rather than per run. The two are different
        orderings and both are stored: this one orders a run, and the payload's orders
        the incident the run belongs to.

        Args:
            event: The event to append.

        Returns:
            The stored record, carrying the sequence it was given.
        """
        ...

    def events_for(self, agent_run_id: str) -> list[AgentRunEventRecord]:
        """Read a run's events in order.

        Args:
            agent_run_id: Which run.

        Returns:
            The events, ordered by sequence.
        """
        ...


@runtime_checkable
class TextSearch(Protocol):
    """Lexical retrieval over the distilled corpus.

    The required backend. It generates candidates and never ranks them, so every method
    here is a way of narrowing the corpus rather than of ordering it.
    """

    def upsert_summary(self, memory: SessionMemory) -> None:
        """Write or replace one record.

        Args:
            memory: The distilled record.
        """
        ...

    def candidates(
        self,
        text: str,
        *,
        limit: int = 500,
        scope: SearchScope = "evidence",
        exclude: Sequence[str] = (),
    ) -> list[SessionMemory]:
        """Return a pool of records that could plausibly match, **unranked**.

        A text backend generates candidates; it does not rank them. Ranking belongs to
        the retriever, because each database ranks differently — PostgreSQL's ``ts_rank``
        has no inverse-document-frequency term at all, and SQLite's FTS5 uses its own
        length normalisation — so a system that let the backend rank would be a
        different retriever on each backend while claiming to be one.

        Term semantics must be disjunctive. Requiring every term reduces the pool to
        empty on most queries, which leaves the ranker nothing to work with.

        Args:
            text: The query, used to select a pool.
            limit: Ceiling on the pool.
            scope: Which projection the query is aimed at.
            exclude: Session ids to withhold, supplied by an evaluation harness so a
                run cannot retrieve its own record.

        Returns:
            Candidate records, in no meaningful order.
        """
        ...

    def session_ids_with_label(self, name: str, value: str) -> list[str]:
        """Find every record carrying one of the environment's own labels.

        A read over :attr:`SessionMemory.labels`, and the only one. An evaluation adapter
        uses it to name the records an episode may not retrieve; nothing about what a
        record says is reachable through it.

        Args:
            name: The label, as :func:`maver.memory.distil.signal_labels` names it.
            value: The value it must hold. An empty value matches nothing rather than
                every record that lacks the label, because the answer decides what is
                withheld and a wildcard there fails open.

        Returns:
            The session ids, in the order the backend stores them.
        """
        ...

    def open_record(self, session_id: str) -> SessionMemory | None:
        """Read one stored record whole, including what retrieval withholds.

        Named apart from :meth:`SessionStore.get_session` because a deployment may put
        both ports on one object, and they answer with different types.

        Args:
            session_id: Which session.

        Returns:
            The record, or None.
        """
        ...

    def delete_summary(self, session_id: str) -> None:
        """Remove one record.

        Args:
            session_id: Which session.
        """
        ...

    def count(self) -> int:
        """Report how many records are stored.

        Returns:
            The record count.
        """
        ...


@runtime_checkable
class VectorSearch(Protocol):
    """Dense retrieval over the same corpus. Optional.

    Offline evaluation on this corpus found lexical ranking to be the ceiling and every
    neural method below it, so this is a swappable alternative rather than the default.
    """

    def upsert_summary(self, memory: SessionMemory) -> None:
        """Write or replace one record.

        Args:
            memory: The distilled record.
        """
        ...

    def query_similar(
        self,
        text: str,
        *,
        n_results: int = 5,
        scope: SearchScope = "evidence",
        exclude: Sequence[str] = (),
    ) -> list[SearchHit]:
        """Rank the corpus against a query.

        Args:
            text: The query.
            n_results: How many to return.
            scope: Which projection to match.
            exclude: Session ids to withhold.

        Returns:
            Hits, best first.
        """
        ...

    def open_record(self, session_id: str) -> SessionMemory | None:
        """Read one stored record whole, including what retrieval withholds.

        Args:
            session_id: Which session.

        Returns:
            The record, or None.
        """
        ...

    def delete_summary(self, session_id: str) -> None:
        """Remove one record.

        Args:
            session_id: Which session.
        """
        ...

    def count(self) -> int:
        """Report how many records are stored.

        Returns:
            The record count.
        """
        ...
