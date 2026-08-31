"""Embedded storage: the memory ports over one SQLite database.

Two classes rather than one. :class:`SqliteStore` holds the ports a running incident
touches and :class:`SqliteRecordIndex` the corpus retrieval reads, which are different
things with different lifetimes: an evaluation harness opens the corpus with no session
store behind it, and a deployment can run with no corpus at all. The index normally
comes from ``SqliteStore.records`` and shares the store's connection, which is what makes
``":memory:"`` a usable database rather than two unrelated ones.

**The index generates candidates and never ranks them.** FTS5 has its own ``bm25()``
with ``b=0.75``, a negated sign, and length normalisation computed across every indexed
column, so a long lesson dilutes an evidence match on the same row. Ordering by it here
would make the embedded deployment a different retriever from the one that was measured,
under the same name. See :mod:`kira.memory.bm25` and :mod:`kira.memory.retrieval`, which
own the ranking.

One connection, guarded by one lock. The alternative, a connection per call, is what the
prototype did; it cannot address an in-memory database, and the store has to be reachable
from the scheduler thread as well as the request thread.
"""

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.memory.bm25 import tokenize
from kira.memory.ports import (
    AgentRunEventDraft,
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStatus,
    ApprovalRecord,
    ApprovalStatus,
    DecisionStatus,
    SearchScope,
    SessionMemory,
    SessionRecord,
)

if TYPE_CHECKING:
    from kira.memory.ports import ApprovalStore, RunStore, SessionStore, TextSearch

__all__ = ["SqliteRecordIndex", "SqliteStore"]

_OPERATIONAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    state               TEXT NOT NULL DEFAULT 'pending',
    run_status          TEXT NOT NULL DEFAULT 'pending',
    current_stage       TEXT,
    checkpoint_json     TEXT,
    awaiting_request_id TEXT,
    alert_json          TEXT NOT NULL DEFAULT '{}',
    root_cause_json     TEXT,
    plan_json           TEXT,
    execution_json      TEXT,
    distilled           INTEGER NOT NULL DEFAULT 0,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state);
CREATE INDEX IF NOT EXISTS idx_sessions_distilled ON sessions(distilled, ended_at);

CREATE TABLE IF NOT EXISTS approvals (
    request_id    TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    summary       TEXT NOT NULL,
    payload_json  TEXT NOT NULL DEFAULT '{}',
    tool_call_id  TEXT,
    tool_name     TEXT,
    prompt        TEXT,
    response_json TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT,
    decided_at    TEXT,
    decided_by    TEXT NOT NULL DEFAULT 'generic human',
    reason        TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id       TEXT PRIMARY KEY,
    pipeline_id        TEXT,
    agent_name         TEXT NOT NULL,
    attempt            INTEGER NOT NULL DEFAULT 1,
    status             TEXT NOT NULL,
    started_at         TEXT NOT NULL,
    ended_at           TEXT,
    output_json        TEXT,
    trace_summary_json TEXT,
    error              TEXT,
    metadata_json      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_pipeline ON agent_runs(pipeline_id, started_at);

CREATE TABLE IF NOT EXISTS agent_run_events (
    event_id     TEXT PRIMARY KEY,
    agent_run_id TEXT NOT NULL,
    pipeline_id  TEXT,
    seq          INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    role         TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (agent_run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_agent_run_events_run ON agent_run_events(agent_run_id, seq);
"""

# Two indexed columns and one payload blob, which is the whole design: FTS5 decides
# which rows could match, and nothing else. Foreign keys are deliberately absent
# throughout: the four ports are separate so that a deployment can implement one
# without the others, and a constraint here would silently make that impossible.
_RECORD_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    session_id UNINDEXED,
    evidence_text,
    lesson_text,
    payload_json UNINDEXED
);
"""

_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)

_SESSION_MEMORY_FIELDS = frozenset(f.name for f in fields(SessionMemory))


def _now() -> str:
    """ISO-8601 UTC with microseconds, so two events in one millisecond differ."""
    return datetime.now(UTC).isoformat()


def _parse_time(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _scrub_text(text: str) -> str:
    """Strip NUL from a string bound to a TEXT column.

    A NUL is not merely ugly here. SQLite stores it but its own scalar functions treat
    TEXT as NUL-terminated, so a value containing one reports the wrong ``length()`` and
    stops matching ``LIKE`` past the byte; PostgreSQL cannot store it in ``text`` or
    ``jsonb`` at all, and in the prototype one such byte in tool output failed the write,
    which made the episode skip every submission and produced a stub result that read as
    an agent that never answered. Removing it at the write boundary is a deliberate,
    lossy choice: one unprintable byte is worth less than the record it would cost.
    """
    return text.replace("\x00", "") if "\x00" in text else text


def _scrub(value: object) -> object:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {_scrub(key): _scrub(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(item) for item in value]
    return value


def _dumps(value: object) -> str:
    """Serialise for a JSON column, scrubbing NUL anywhere inside it."""
    return json.dumps(_scrub(value))


def _scrub_dict(value: dict[str, Any]) -> dict[str, Any]:
    """What a dict looks like once stored, for a record handed straight back."""
    scrubbed: dict[str, Any] = json.loads(_dumps(value))
    return scrubbed


def _dumps_optional(value: object | None) -> str | None:
    return None if value is None else _dumps(value)


def _text(value: str | None) -> str | None:
    return None if value is None else _scrub_text(value)


def _loads(payload: str | None) -> dict[str, Any] | None:
    if not payload:
        return None
    loaded: dict[str, Any] = json.loads(payload)
    return loaded


def _loads_or_empty(payload: str | None) -> dict[str, Any]:
    return _loads(payload) or {}


class _Connection:
    """One connection and the lock that serialises access to it.

    Not an abstraction over sqlite3; every caller writes its own SQL. This exists only
    so the two classes can share a connection safely across threads.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the database, creating its parent directory for a file path."""
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.raw = sqlite3.connect(path, timeout=10, isolation_level=None, check_same_thread=False)
        self.raw.row_factory = sqlite3.Row
        self.raw.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            yield self.raw

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.raw.execute("BEGIN IMMEDIATE")
            try:
                yield self.raw
            except BaseException:
                self.raw.execute("ROLLBACK")
                raise
            self.raw.execute("COMMIT")

    def script(self, sql: str) -> None:
        # `executescript` commits whatever transaction is open before it runs, so DDL
        # cannot go inside `write()`. Every statement here is `IF NOT EXISTS`.
        with self.lock:
            self.raw.executescript(sql)

    def close(self) -> None:
        with self.lock:
            self.raw.close()


class SqliteStore:
    """:class:`~kira.memory.ports.SessionStore`, ``ApprovalStore`` and ``RunStore``.

    Attributes:
        records: The corpus index, sharing this store's database.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Open the database and create the schema if it is not there.

        Args:
            path: Database file, or ``":memory:"`` for an ephemeral one.
        """
        self._db = _Connection(path)
        self._db.script(_OPERATIONAL_SCHEMA)
        self.records = SqliteRecordIndex(self._db)

    def close(self) -> None:
        """Close the connection, and with it the record index sharing it."""
        self._db.close()

    # SessionStore ---------------------------------------------------------------

    def create_session(self, alert: dict[str, Any]) -> SessionRecord:
        """Open a session for an incoming alert.

        Args:
            alert: The alert payload.

        Returns:
            The created record, carrying the id and start time assigned here.
        """
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            started_at=_now(),
            alert=alert,
        )
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, started_at, state, run_status, alert_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    record.session_id,
                    record.started_at,
                    record.state,
                    record.run_status,
                    _dumps(alert),
                ),
            )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Read one session.

        Args:
            session_id: Which session.

        Returns:
            The record, or None when it does not exist.
        """
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else _to_session(row)

    def get_states(self, session_ids: Sequence[str]) -> dict[str, str]:
        """Read the lifecycle state of many sessions at once.

        Args:
            session_ids: Which sessions.

        Returns:
            Session id to lifecycle state, omitting sessions that do not exist.
        """
        ids = list(session_ids)
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with self._db.read() as conn:
            rows = conn.execute(
                f"SELECT session_id, state FROM sessions WHERE session_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row["session_id"]: row["state"] for row in rows}

    def update_artefacts(
        self,
        session_id: str,
        *,
        root_cause: dict[str, Any] | None = None,
        plan: dict[str, Any] | None = None,
        execution_result: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Attach what a role produced to its session.

        Partial: an argument left at None leaves that column alone, so Planning
        attaching a plan cannot erase the diagnosis that preceded it.

        Args:
            session_id: Which session.
            root_cause: The diagnosis, when one was submitted.
            plan: The remediation plan, when one was submitted.
            execution_result: What execution did, when it ran.
            notes: Free text for an operator.
        """
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("root_cause_json", _dumps_optional(root_cause)),
            ("plan_json", _dumps_optional(plan)),
            ("execution_json", _dumps_optional(execution_result)),
            ("notes", _text(notes)),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(session_id)
        with self._db.write() as conn:
            conn.execute(
                f"UPDATE sessions SET {', '.join(assignments)} WHERE session_id = ?", values
            )

    def set_state(self, session_id: str, state: LifecycleState) -> None:
        """Record how far a session's claim has been confirmed.

        Args:
            session_id: Which session.
            state: The new lifecycle state.
        """
        with self._db.write() as conn:
            conn.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?", (state.value, session_id)
            )

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

        All four values move in one statement, so no reader ever observes
        ``awaiting_human`` without the checkpoint that belongs to it. A None therefore
        clears rather than preserves: a run resuming passes its new status alone, and
        the checkpoint it has just consumed goes with it. ``ended_at`` is stamped the
        first time a terminal status is recorded and never moves again.

        Args:
            session_id: Which session.
            status: The operational status.
            current_stage: The role that was running.
            checkpoint: State to resume from.
            awaiting_request_id: The approval being waited on, when parked.
        """
        ended_at = _now() if status.value in _TERMINAL_RUN_STATUSES else None
        with self._db.write() as conn:
            conn.execute(
                "UPDATE sessions SET run_status = ?, current_stage = ?, checkpoint_json = ?, "
                "awaiting_request_id = ?, ended_at = COALESCE(ended_at, ?) WHERE session_id = ?",
                (
                    status.value,
                    _text(current_stage),
                    _dumps_optional(checkpoint),
                    awaiting_request_id,
                    ended_at,
                    session_id,
                ),
            )

    def mark_distilled(self, session_id: str) -> None:
        """Record that a session's lesson has been written to the corpus.

        Args:
            session_id: Which session.
        """
        with self._db.write() as conn:
            conn.execute("UPDATE sessions SET distilled = 1 WHERE session_id = ?", (session_id,))

    # ApprovalStore --------------------------------------------------------------

    def create_request(self, record: ApprovalRecord) -> ApprovalRecord:
        """Persist a pending request.

        The record is stored as given, apart from ``created_at``, which is stamped when
        it is blank because it is what orders the list a human reads.

        Args:
            record: The request.

        Returns:
            The stored record.
        """
        stored = replace(
            record,
            summary=_scrub_text(record.summary),
            prompt=_text(record.prompt),
            reason=_text(record.reason),
            created_at=record.created_at or _now(),
        )
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO approvals (request_id, session_id, kind, stage, status, summary, "
                "payload_json, tool_call_id, tool_name, prompt, response_json, created_at, "
                "expires_at, decided_at, decided_by, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stored.request_id,
                    stored.session_id,
                    stored.kind,
                    stored.stage,
                    stored.status,
                    stored.summary,
                    _dumps(stored.payload),
                    stored.tool_call_id,
                    stored.tool_name,
                    stored.prompt,
                    _dumps_optional(stored.response),
                    stored.created_at,
                    stored.expires_at,
                    stored.decided_at,
                    stored.decided_by,
                    stored.reason,
                ),
            )
        return stored

    def get_request(self, request_id: str) -> ApprovalRecord | None:
        """Read one request.

        Args:
            request_id: Which request.

        Returns:
            The record, or None.
        """
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
        return None if row is None else _to_approval(row)

    def list_requests(self, *, status: ApprovalStatus | None = None) -> list[ApprovalRecord]:
        """List requests, newest first.

        Args:
            status: Only requests in this status, or all of them.

        Returns:
            The matching records.
        """
        clause = "" if status is None else " WHERE status = ?"
        params: tuple[str, ...] = () if status is None else (status,)
        with self._db.read() as conn:
            rows = conn.execute(
                # rowid breaks ties, because two requests raised in the same microsecond
                # still have an order and it is the order they arrived in.
                f"SELECT * FROM approvals{clause} ORDER BY created_at DESC, rowid DESC",
                params,
            ).fetchall()
        return [_to_approval(row) for row in rows]

    def decide(
        self,
        request_id: str,
        *,
        status: DecisionStatus,
        decided_by: str,
        reason: str | None = None,
    ) -> ApprovalRecord | None:
        """Record a decision, if the request is still pending.

        The status test is part of the UPDATE rather than a read before it, so two
        decisions racing cannot both win.

        Args:
            request_id: Which request.
            status: The decision.
            decided_by: Who decided.
            reason: Why.

        Returns:
            The updated record, or None when it was not pending.
        """
        with self._db.write() as conn:
            cursor = conn.execute(
                "UPDATE approvals SET status = ?, decided_at = ?, decided_by = ?, reason = ? "
                "WHERE request_id = ? AND status = ?",
                (status, _now(), _scrub_text(decided_by), _text(reason), request_id, "pending"),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM approvals WHERE request_id = ?", (request_id,)
            ).fetchone()
        return _to_approval(row)

    def expire_due(self, *, now: str) -> list[ApprovalRecord]:
        """Time out every request past its expiry.

        Expiry is compared as parsed instants rather than as strings, so a caller whose
        timestamps end in ``Z`` and one whose timestamps end in ``+00:00`` get the same
        answer. A request whose ``expires_at`` cannot be parsed is left pending: a
        malformed expiry is not evidence that the request has expired.

        Args:
            now: The current timestamp.

        Returns:
            The records that expired.
        """
        cutoff = _parse_time(now)
        with self._db.write() as conn:
            pending = conn.execute(
                "SELECT request_id, expires_at FROM approvals "
                "WHERE status = 'pending' AND expires_at IS NOT NULL"
            ).fetchall()
            due: list[str] = []
            for row in pending:
                try:
                    expires_at = _parse_time(row["expires_at"])
                except ValueError:
                    continue
                if expires_at <= cutoff:
                    due.append(row["request_id"])
            if not due:
                return []
            placeholders = ", ".join("?" for _ in due)
            conn.execute(
                f"UPDATE approvals SET status = 'timed_out', decided_at = ? "
                f"WHERE request_id IN ({placeholders})",
                [now, *due],
            )
            rows = conn.execute(
                f"SELECT * FROM approvals WHERE request_id IN ({placeholders})", due
            ).fetchall()
        return [_to_approval(row) for row in rows]

    # RunStore -------------------------------------------------------------------

    def start_run(self, record: AgentRunRecord) -> None:
        """Record that a role started.

        Args:
            record: The run.
        """
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO agent_runs (agent_run_id, pipeline_id, agent_name, attempt, status, "
                "started_at, ended_at, output_json, trace_summary_json, error, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.agent_run_id,
                    record.pipeline_id,
                    record.agent_name,
                    record.attempt,
                    record.status,
                    record.started_at,
                    record.ended_at,
                    _dumps_optional(record.output),
                    _dumps_optional(record.trace_summary),
                    _text(record.error),
                    _dumps(record.metadata),
                ),
            )

    def finish_run(
        self,
        agent_run_id: str,
        *,
        status: AgentRunStatus,
        output: dict[str, Any] | None = None,
        trace_summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Record that a role finished.

        Terminal, and written whole: the arguments describe the run's final state, so a
        None here is "produced nothing" rather than "leave what was there".

        Args:
            agent_run_id: Which run.
            status: How it ended.
            output: What it produced.
            trace_summary: A condensed account of what it did.
            error: Why it failed, when it did.
        """
        with self._db.write() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = ?, ended_at = ?, output_json = ?, "
                "trace_summary_json = ?, error = ? WHERE agent_run_id = ?",
                (
                    status,
                    _now(),
                    _dumps_optional(output),
                    _dumps_optional(trace_summary),
                    _text(error),
                    agent_run_id,
                ),
            )

    def get_run(self, agent_run_id: str) -> AgentRunRecord | None:
        """Read one run.

        Args:
            agent_run_id: Which run.

        Returns:
            The record, or None when it does not exist.
        """
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE agent_run_id = ?", (agent_run_id,)
            ).fetchone()
        return None if row is None else _to_run(row)

    def runs_for(self, session_id: str) -> list[AgentRunRecord]:
        """Read every role's run within one incident, oldest first.

        Args:
            session_id: Which incident.

        Returns:
            The runs, ordered by start time.
        """
        with self._db.read() as conn:
            rows = conn.execute(
                # rowid breaks ties, because two roles starting in the same microsecond
                # still ran in the order they were written.
                "SELECT * FROM agent_runs WHERE pipeline_id = ? ORDER BY started_at, rowid",
                (session_id,),
            ).fetchall()
        return [_to_run(row) for row in rows]

    def append_event(self, event: AgentRunEventDraft) -> AgentRunEventRecord:
        """Append one event, giving it the next place in the run's order.

        The read of the current maximum and the insert that depends on it are one
        ``BEGIN IMMEDIATE`` transaction, so two writers cannot allocate the same
        sequence; ``UNIQUE (agent_run_id, seq)`` is the second line of defence rather
        than the first.

        Args:
            event: The event to append.

        Returns:
            The stored record, carrying the sequence it was given.
        """
        with self._db.write() as conn:
            seq = int(
                conn.execute(
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_run_events WHERE agent_run_id = ?",
                    (event.agent_run_id,),
                ).fetchone()[0]
            )
            conn.execute(
                "INSERT INTO agent_run_events (event_id, agent_run_id, pipeline_id, seq, "
                "event_type, role, tool_call_id, tool_name, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.agent_run_id,
                    event.pipeline_id,
                    seq,
                    event.event_type,
                    event.role,
                    event.tool_call_id,
                    event.tool_name,
                    _dumps(event.payload),
                    event.created_at,
                ),
            )
        return AgentRunEventRecord(
            event_id=event.event_id,
            agent_run_id=event.agent_run_id,
            pipeline_id=event.pipeline_id,
            seq=seq,
            event_type=event.event_type,
            payload=_scrub_dict(event.payload),
            created_at=event.created_at,
            role=event.role,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
        )

    def events_for(self, agent_run_id: str) -> list[AgentRunEventRecord]:
        """Read a run's events in order.

        Args:
            agent_run_id: Which run.

        Returns:
            The events, ordered by sequence.
        """
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_run_events WHERE agent_run_id = ? ORDER BY seq",
                (agent_run_id,),
            ).fetchall()
        return [_to_event(row) for row in rows]


class SqliteRecordIndex:
    """:class:`~kira.memory.ports.TextSearch` over an FTS5 index.

    A candidate generator. Nothing here orders by relevance; see the module docstring.
    """

    def __init__(self, connection: _Connection | str | Path = ":memory:") -> None:
        """Open or adopt a database and create the index if it is not there.

        Args:
            connection: A path, or the connection of the :class:`SqliteStore` this
                index belongs to.
        """
        self._db = connection if isinstance(connection, _Connection) else _Connection(connection)
        self._db.script(_RECORD_SCHEMA)

    def close(self) -> None:
        """Close the connection. A no-op worth calling only on a standalone index."""
        self._db.close()

    def upsert_summary(self, memory: SessionMemory) -> None:
        """Write or replace one record.

        Args:
            memory: The distilled record.
        """
        payload = {
            field.name: getattr(memory, field.name)
            for field in fields(memory)
            if field.name not in ("session_id", "evidence_text", "lesson_text")
        }
        with self._db.write() as conn:
            # FTS5 has no upsert, and a session must not end up indexed twice.
            conn.execute("DELETE FROM records_fts WHERE session_id = ?", (memory.session_id,))
            conn.execute(
                "INSERT INTO records_fts (session_id, evidence_text, lesson_text, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    memory.session_id,
                    _scrub_text(memory.evidence_text),
                    _scrub_text(memory.lesson_text),
                    _dumps(payload),
                ),
            )

    def candidates(
        self,
        text: str,
        *,
        limit: int = 500,
        scope: SearchScope = "evidence",
        exclude: Sequence[str] = (),
    ) -> list[SessionMemory]:
        """Return a pool of records that could plausibly match, unranked.

        Terms are ORed. Quoting each one keeps a token that happens to spell an FTS5
        operator a search term, and there is no cap on how many are used: the prototype
        truncated the query to 32 tokens, which silently decided which documents the
        ranker was allowed to see.

        Args:
            text: The query, used to select a pool.
            limit: Ceiling on the pool.
            scope: Which projection the query is aimed at.
            exclude: Session ids to withhold.

        Returns:
            Candidate records, in insertion order, which carries no relevance.
        """
        terms = list(dict.fromkeys(tokenize(text)))
        if not terms or limit <= 0:
            return []
        disjunction = " OR ".join(f'"{term}"' for term in terms)
        expression = f"{_column(scope)} : ({disjunction})"

        withheld = list(exclude)
        clause = ""
        if withheld:
            clause = f" AND session_id NOT IN ({', '.join('?' for _ in withheld)})"
        with self._db.read() as conn:
            rows = conn.execute(
                # ORDER BY rowid, never BY rank: FTS5's own bm25() is a different scorer
                # (b=0.75, negated, normalised across both indexed columns), and letting
                # it decide the pool would make this a different retriever.
                f"SELECT session_id, evidence_text, lesson_text, payload_json FROM records_fts "
                f"WHERE records_fts MATCH ?{clause} ORDER BY rowid LIMIT ?",
                [expression, *withheld, limit],
            ).fetchall()
        return [_to_memory(row) for row in rows]

    def scan_summaries(
        self,
        pattern: str,
        *,
        n_results: int = 5,
        scope: SearchScope = "evidence",
    ) -> list[SessionMemory]:
        """Return only records matching a regular expression.

        SQLite has no REGEXP operator by default and the corpus is small, so rows are
        filtered here rather than through a function registered on the connection. An
        invalid pattern returns nothing instead of raising, which costs a role one turn.

        Args:
            pattern: A case-insensitive regular expression.
            n_results: How many to return.
            scope: Which projection to match.

        Returns:
            Matching records, possibly none.
        """
        if not pattern.strip() or n_results <= 0:
            return []
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT session_id, evidence_text, lesson_text, payload_json FROM records_fts "
                "ORDER BY rowid"
            ).fetchall()
        found: list[SessionMemory] = []
        for row in rows:
            memory = _to_memory(row)
            body = _projection(memory, scope)
            if body and expression.search(body):
                found.append(memory)
            if len(found) >= n_results:
                break
        return found

    def open_record(self, session_id: str) -> SessionMemory | None:
        """Read one stored record whole, including what retrieval withholds.

        Args:
            session_id: Which session.

        Returns:
            The record, or None.
        """
        with self._db.read() as conn:
            row = conn.execute(
                # session_id is UNINDEXED, so this is a scan. The corpus is hundreds of
                # records; an FTS5 index on an identifier would cost more than it saves.
                "SELECT session_id, evidence_text, lesson_text, payload_json FROM records_fts "
                "WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return None if row is None else _to_memory(row)

    def delete_summary(self, session_id: str) -> None:
        """Remove one record.

        Args:
            session_id: Which session.
        """
        with self._db.write() as conn:
            conn.execute("DELETE FROM records_fts WHERE session_id = ?", (session_id,))

    def count(self) -> int:
        """Report how many records are stored.

        Returns:
            The record count.
        """
        with self._db.read() as conn:
            total: int = conn.execute("SELECT COUNT(*) FROM records_fts").fetchone()[0]
        return total


def _column(scope: SearchScope) -> str:
    return "lesson_text" if scope == "lesson" else "evidence_text"


def _projection(memory: SessionMemory, scope: SearchScope) -> str:
    return memory.lesson_text if scope == "lesson" else memory.evidence_text


def _to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        state=row["state"],
        run_status=row["run_status"],
        current_stage=row["current_stage"],
        checkpoint=_loads(row["checkpoint_json"]),
        awaiting_request_id=row["awaiting_request_id"],
        alert=_loads_or_empty(row["alert_json"]),
        root_cause=_loads(row["root_cause_json"]),
        plan=_loads(row["plan_json"]),
        execution_result=_loads(row["execution_json"]),
        distilled=bool(row["distilled"]),
        notes=row["notes"],
    )


def _to_approval(row: sqlite3.Row) -> ApprovalRecord:
    return ApprovalRecord(
        request_id=row["request_id"],
        session_id=row["session_id"],
        kind=row["kind"],
        stage=row["stage"],
        status=row["status"],
        summary=row["summary"],
        payload=_loads_or_empty(row["payload_json"]),
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        prompt=row["prompt"],
        response=_loads(row["response_json"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
        reason=row["reason"],
    )


def _to_run(row: sqlite3.Row) -> AgentRunRecord:
    return AgentRunRecord(
        agent_run_id=row["agent_run_id"],
        pipeline_id=row["pipeline_id"],
        agent_name=row["agent_name"],
        attempt=row["attempt"],
        status=row["status"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        output=_loads(row["output_json"]),
        trace_summary=_loads(row["trace_summary_json"]),
        error=row["error"],
        metadata=_loads_or_empty(row["metadata_json"]),
    )


def _to_event(row: sqlite3.Row) -> AgentRunEventRecord:
    return AgentRunEventRecord(
        event_id=row["event_id"],
        agent_run_id=row["agent_run_id"],
        pipeline_id=row["pipeline_id"],
        seq=row["seq"],
        event_type=row["event_type"],
        payload=_loads_or_empty(row["payload_json"]),
        created_at=row["created_at"],
        role=row["role"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
    )


def _to_memory(row: sqlite3.Row) -> SessionMemory:
    # Filtered against the dataclass so a field this build does not know about is
    # dropped rather than failing the read of every record beside it.
    payload = {
        key: value
        for key, value in _loads_or_empty(row["payload_json"]).items()
        if key in _SESSION_MEMORY_FIELDS
    }
    return SessionMemory(
        session_id=row["session_id"],
        evidence_text=row["evidence_text"] or "",
        lesson_text=row["lesson_text"] or "",
        **payload,
    )


if TYPE_CHECKING:

    def _ports_are_satisfied(
        store: SqliteStore, index: SqliteRecordIndex
    ) -> tuple["SessionStore", "ApprovalStore", "RunStore", "TextSearch"]:
        """Static proof of conformance. Type-checked, never called."""
        return store, store, store, index
