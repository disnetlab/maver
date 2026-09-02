"""Server storage: the memory ports over one PostgreSQL schema.

The same two classes as the embedded backend, for the same reason.
:class:`PostgresStore` holds the ports a running incident touches and
:class:`PostgresRecordIndex` the corpus retrieval reads; the index normally comes from
``PostgresStore.records`` and shares the store's connections, so the two see one
transaction boundary rather than two unrelated databases.

**The index generates candidates and never ranks them.** PostgreSQL's ``ts_rank`` has no
inverse-document-frequency term at all, so ordering by it here would make a deployment on
this backend a different retriever from the one that was measured, under the same name.
:mod:`kira.memory.bm25` and :mod:`kira.memory.retrieval` own the ranking, and the query
below has no ``ORDER BY`` that means anything.

Three things differ from SQLite and each is a decision rather than a translation.

**A pool of connections, not one guarded by a lock.** The embedded store serialises every
call through a single connection because SQLite has a single writer anyway. Here that
would be the wrong shape twice over: a server exists to be written to concurrently, and
the event sequence is allocated by letting two writers collide and retrying the loser, so
a store that could not produce a collision would be hiding the case it exists to handle.

**Timestamps are ``TIMESTAMPTZ``, and come back canonical.** The ports carry ISO-8601
strings, which are parsed on the way in and rendered on the way out with the session
pinned to UTC. A caller that writes ``"...00Z"`` therefore reads back ``"...00+00:00"``:
the same instant, in the spelling this system's own writes use. The cost is that a
timestamp PostgreSQL cannot parse is refused at the write rather than stored and ignored,
which is the one place this backend cannot mirror the embedded one, because ``TIMESTAMPTZ``
has no representation for "not a time".

**Rows carry their corpus.** The embedded backend partitions campaigns by database file;
one server database holds many, and the identifiers that reach it are not all generated
here — a run id or an approval id is supplied by the caller — so the corpus belongs in
the key rather than in a convention about how unique those identifiers happen to be.

**Every wait is bounded.** The embedded backend has had a lock-wait bound since it was
written, ``sqlite3.connect(timeout=10)``; this one had none, so a connection that never
established and a statement blocked on a lock both waited forever, on whatever thread
called them. That thread is normally the event loop running the incident, which has no
other clock: a wrapper timeout cannot cancel a call that never yields. Both waits are
bounded below, and a bound the deployment arranged for itself always wins, because an
operator who tuned their DSN or their role knows something this file does not.
"""

import logging
import re
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.memory.backends.sanitise import scrub, scrub_mapping, scrub_optional, scrub_text
from kira.memory.bm25 import tokenize
from kira.memory.lifecycle import check_transition
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

__all__ = ["PostgresRecordIndex", "PostgresStore"]

_log = logging.getLogger(__name__)

_Row = dict[str, Any]
_Connection = psycopg.Connection[_Row]

# Foreign keys are deliberately absent, as in the embedded schema: the four ports are
# separate so that a deployment can implement one without the others, and a constraint
# here would silently make that impossible.
_OPERATIONAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    corpus_id           TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    state               TEXT NOT NULL DEFAULT 'pending',
    run_status          TEXT NOT NULL DEFAULT 'pending',
    current_stage       TEXT,
    checkpoint          JSONB,
    awaiting_request_id TEXT,
    alert               JSONB NOT NULL DEFAULT '{}'::jsonb,
    root_cause          JSONB,
    plan                JSONB,
    execution_result    JSONB,
    verify_result       JSONB,
    distilled           BOOLEAN NOT NULL DEFAULT FALSE,
    notes               TEXT,
    PRIMARY KEY (corpus_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions (corpus_id, state);
CREATE INDEX IF NOT EXISTS idx_sessions_distilled ON sessions (corpus_id, distilled, ended_at);

CREATE TABLE IF NOT EXISTS approvals (
    corpus_id    TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,
    summary      TEXT NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    tool_call_id TEXT,
    tool_name    TEXT,
    prompt       TEXT,
    response     JSONB,
    created_at   TIMESTAMPTZ NOT NULL,
    expires_at   TIMESTAMPTZ,
    decided_at   TIMESTAMPTZ,
    decided_by   TEXT NOT NULL DEFAULT 'generic human',
    reason       TEXT,
    PRIMARY KEY (corpus_id, request_id)
);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals (corpus_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals (corpus_id, session_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    corpus_id     TEXT NOT NULL,
    agent_run_id  TEXT NOT NULL,
    pipeline_id   TEXT,
    agent_name    TEXT NOT NULL,
    attempt       INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,
    ended_at      TIMESTAMPTZ,
    output        JSONB,
    trace_summary JSONB,
    error         TEXT,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (corpus_id, agent_run_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_pipeline
    ON agent_runs (corpus_id, pipeline_id, started_at);

CREATE TABLE IF NOT EXISTS agent_run_events (
    corpus_id    TEXT NOT NULL,
    event_id     TEXT NOT NULL,
    agent_run_id TEXT NOT NULL,
    pipeline_id  TEXT,
    seq          INTEGER NOT NULL,
    event_type   TEXT NOT NULL,
    role         TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (corpus_id, event_id),
    CONSTRAINT agent_run_events_seq_key UNIQUE (corpus_id, agent_run_id, seq)
);
"""

# Two generated tsvectors and their GIN indexes, which is the whole design of the index:
# PostgreSQL decides which rows could match, and nothing else. There are no vector columns
# and no `CREATE EXTENSION vector` here on purpose. The default deployment needs neither,
# and shipping them would install an embedder nobody asked for.
_RECORD_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    corpus_id                     TEXT NOT NULL,
    session_id                    TEXT NOT NULL,
    evidence_text                 TEXT NOT NULL DEFAULT '',
    lesson_text                   TEXT NOT NULL DEFAULT '',
    incident_summary              TEXT NOT NULL DEFAULT '',
    lifecycle_state               TEXT NOT NULL DEFAULT 'pending',
    fault_category                TEXT NOT NULL DEFAULT '',
    alert_name                    TEXT NOT NULL DEFAULT '',
    namespace                     TEXT NOT NULL DEFAULT '',
    affected_resource             TEXT NOT NULL DEFAULT '',
    verification_verdict          TEXT NOT NULL DEFAULT '',
    evaluation_label              TEXT NOT NULL DEFAULT '',
    evaluation_source             TEXT NOT NULL DEFAULT '',
    evaluation_diagnosis_success  BOOLEAN,
    evaluation_diagnosis_score    DOUBLE PRECISION,
    evaluation_mitigation_success BOOLEAN,
    corrected_cause               TEXT NOT NULL DEFAULT '',
    written_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    evidence_tsv TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(evidence_text, ''))) STORED,
    lesson_tsv   TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(lesson_text, ''))) STORED,
    PRIMARY KEY (corpus_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_records_evidence_tsv ON records USING GIN (evidence_tsv);
CREATE INDEX IF NOT EXISTS idx_records_lesson_tsv ON records USING GIN (lesson_tsv);
"""

# Every lexeme the query text yields, ORed. The disjunction is the point: requiring every
# term empties the pool on most real queries and leaves the ranker nothing to work with.
# The `websearch_to_tsquery` form the prototype could be configured into is not offered
# at all rather than merely not defaulted to, because it is the mechanism behind a 91%
# no-match failure mode and leaving it selectable would leave that reachable.
#
# The inner normalisation is 'english' and the outer one is 'simple', which is not a
# typo. The inner call turns the query text into the same lexemes the stored vectors
# hold; the outer call would stem them a second time, and a stem is not always its own
# fixed point — "raise" becomes "rais", which stems again to "rai" and then matches
# nothing. Under 'simple' the already-normalised lexemes are taken as they are.
_CANDIDATE_QUERY = """
WITH q AS (
    SELECT to_tsquery('simple',
        (SELECT string_agg(quote_literal(lexeme), ' | ')
         FROM unnest(to_tsvector('english', %s)))) AS tsq
)
SELECT {columns}
FROM records, q
WHERE q.tsq @@ {vector}
  AND corpus_id = %s
  AND session_id <> ALL(%s)
ORDER BY written_at, session_id
LIMIT %s
"""

_SEQUENCE_CONSTRAINT = "agent_run_events_seq_key"
_SEQUENCE_ATTEMPTS = 5

CONNECT_TIMEOUT_S = 10
"""How long to wait for a connection to establish.

The server is on the same host or the same cluster network, so a connection that has not
been made in ten seconds is not being made. libpq reads anything below two as two.
"""

STATEMENT_TIMEOUT_MS = 30_000
"""How long one statement may run before the server cancels it.

Above the slowest wait this backend is built to survive — an event insert blocked on a
rival's uncommitted sequence, which the retry loop exists for and which the race test
allows fifteen seconds — and far below the hours an evaluation gives one incident. A
statement over this is not slow, it is stuck.
"""

_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)

# Derived from the dataclass so that the record table is described in one place, the DDL.
# A field added to SessionMemory without a column beside it fails the next write loudly,
# which is the failure worth having: the alternative is a payload blob that accepts the
# field and a corpus nobody can query with SQL.
_RECORD_FIELDS = tuple(field.name for field in fields(SessionMemory))
_RECORD_COLUMNS = ", ".join(_RECORD_FIELDS)
_RECORD_PLACEHOLDERS = ", ".join("%s" for _ in _RECORD_FIELDS)
_RECORD_UPDATES = ", ".join(
    f"{name} = EXCLUDED.{name}" for name in _RECORD_FIELDS if name != "session_id"
)


def _now() -> datetime:
    """The current instant. ISO-8601 UTC with microseconds once it is rendered back."""
    return datetime.now(UTC)


def _instant(text: str) -> datetime:
    """Parse a timestamp for a TIMESTAMPTZ column, assuming UTC when none is given."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _instant_or_none(text: str | None) -> datetime | None:
    return None if text is None else _instant(text)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _json(value: dict[str, Any] | None) -> Jsonb | None:
    """Bind a payload to a JSONB column, scrubbing NUL anywhere inside it."""
    return None if value is None else Jsonb(scrub(value))


def _connect_params(dsn: str, connect_timeout: int) -> dict[str, Any]:
    """The connection keywords to add to what the operator wrote.

    ``conninfo_to_dict`` reads both spellings a DSN comes in, a ``postgresql://`` URI and
    a keyword string, and reports only what was written rather than what libpq would
    default. So an operator's own value is visible here, and is left alone.
    """
    if "connect_timeout" in conninfo_to_dict(dsn):
        return {}
    return {"connect_timeout": connect_timeout}


def _statement_timeout(dsn: str, milliseconds: int) -> int | None:
    """The statement bound to install, or None when the DSN already carries one.

    A DSN sets it through libpq's ``options``, which is also where an operator writes
    ``statement_timeout=0`` to mean no bound at all. That reads back from the session as
    "no timeout", indistinguishable from never having set one, so the intent has to be
    read from the DSN itself.
    """
    options = str(conninfo_to_dict(dsn).get("options") or "")
    return None if "statement_timeout" in options else milliseconds


class _Pool:
    """Connections handed out one at a time, because a transaction belongs to one.

    Not an abstraction over psycopg; every caller writes its own SQL. This exists so that
    two threads can write at once, which is what the sequence allocation is built to
    survive, and so that closing the store closes every connection it opened rather than
    leaving one for the garbage collector to complain about.

    Connections are opened on demand and reused, one per concurrent caller. There is no
    ceiling and therefore no queue to wait in: a deployment is one process with a handful
    of threads, and a limit here would be a second thing to tune beside the server's own.
    """

    def __init__(
        self,
        dsn: str,
        schema: str,
        *,
        connect_timeout: int = CONNECT_TIMEOUT_S,
        statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
    ) -> None:
        """Open the first connection and create the schema if it is not there.

        Args:
            dsn: Connection string.
            schema: Schema holding the tables.
            connect_timeout: Seconds to wait for a connection, unless the DSN says.
            statement_timeout_ms: Milliseconds one statement may run, unless the
                deployment already says.
        """
        self._dsn = dsn
        self._schema = schema
        self._connect_params = _connect_params(dsn, connect_timeout)
        self._statement_timeout_ms = _statement_timeout(dsn, statement_timeout_ms)
        self._lock = threading.Lock()
        self._idle: list[_Connection] = []
        self._open: list[_Connection] = []
        self._closed = False
        with self._checkout() as conn:
            conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self._schema))
            )

    @contextmanager
    def read(self) -> Iterator[_Connection]:
        with self._checkout() as conn:
            yield conn

    @contextmanager
    def write(self) -> Iterator[_Connection]:
        with self._checkout() as conn, conn.transaction():
            yield conn

    def script(self, statements: str) -> None:
        """Run DDL. Every statement in it is ``IF NOT EXISTS``."""
        with self.write() as conn:
            # Sent without parameters, which is what lets one call carry several
            # statements.
            conn.execute(statements)

    def close(self) -> None:
        """Close every connection this pool opened."""
        with self._lock:
            self._closed = True
            open_connections, self._open, self._idle = self._open, [], []
        for conn in open_connections:
            conn.close()

    @contextmanager
    def _checkout(self) -> Iterator[_Connection]:
        conn = self._acquire()
        try:
            yield conn
        finally:
            self._release(conn)

    def _acquire(self) -> _Connection:
        with self._lock:
            if self._closed:
                raise RuntimeError("the storage backend is closed")
            while self._idle:
                conn = self._idle.pop()
                if not conn.closed:
                    return conn
        return self._connect()

    def _release(self, conn: _Connection) -> None:
        # A connection whose server-side state is unknown is dropped rather than reused:
        # the next caller would inherit an aborted transaction it had no part in.
        if conn.closed or conn.broken:
            conn.close()
            with self._lock:
                if conn in self._open:
                    self._open.remove(conn)
            return
        with self._lock:
            if self._closed:
                conn.close()
            else:
                self._idle.append(conn)

    def _connect(self) -> _Connection:
        conn: _Connection = psycopg.connect(
            self._dsn, autocommit=True, row_factory=dict_row, **self._connect_params
        )
        try:
            # Unqualified table names throughout, so the schema is named once rather than
            # in every statement. The session is pinned to UTC because a TIMESTAMPTZ is
            # rendered in the session's zone, and a record's timestamps are strings that
            # have to read back the way they were written.
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self._schema)))
            conn.execute("SET TIME ZONE 'UTC'")
            self._cap_statements(conn)
        except BaseException:
            conn.close()
            raise
        with self._lock:
            if self._closed:
                conn.close()
                raise RuntimeError("the storage backend is closed")
            self._open.append(conn)
        return conn

    def _cap_statements(self, conn: _Connection) -> None:
        """Bound this session's statements, unless the deployment already bounds them.

        Asked of the session rather than assumed, so that a timeout set on the role or
        the database — the usual place an operator puts one — is left alone as surely as
        one written into the DSN. Only a session reporting no timeout at all gets this.
        """
        if self._statement_timeout_ms is None:
            return
        row = conn.execute("SHOW statement_timeout").fetchone()
        if row is not None and row["statement_timeout"] != "0":
            return
        conn.execute(
            sql.SQL("SET statement_timeout = {}").format(sql.Literal(self._statement_timeout_ms))
        )


class PostgresStore:
    """:class:`~kira.memory.ports.SessionStore`, ``ApprovalStore`` and ``RunStore``.

    Attributes:
        corpus_id: The campaign whose rows this store can see.
        records: The corpus index, sharing this store's connections.
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "kira_memory",
        corpus_id: str = "default",
        connect_timeout: int = CONNECT_TIMEOUT_S,
        statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
    ) -> None:
        """Open the database and create the schema if it is not there.

        Args:
            dsn: Connection string.
            schema: Schema holding the tables.
            corpus_id: The campaign this deployment reads and writes.
            connect_timeout: Seconds to wait for a connection. A ``connect_timeout``
                in the DSN wins over this.
            statement_timeout_ms: Milliseconds one statement may run. A timeout the
                deployment already sets — in the DSN, on the role, on the database —
                wins over this.
        """
        self.corpus_id = corpus_id
        self._db = _Pool(
            dsn,
            schema,
            connect_timeout=connect_timeout,
            statement_timeout_ms=statement_timeout_ms,
        )
        self._db.script(_OPERATIONAL_SCHEMA)
        self.records = PostgresRecordIndex(self._db, corpus_id=corpus_id)

    def close(self) -> None:
        """Close every connection, and with them the record index sharing them."""
        self._db.close()

    # SessionStore ---------------------------------------------------------------

    def create_session(self, alert: dict[str, Any]) -> SessionRecord:
        """Open a session for an incoming alert.

        Args:
            alert: The alert payload.

        Returns:
            The created record, carrying the id and start time assigned here.
        """
        started_at = _now()
        record = SessionRecord(
            session_id=str(uuid.uuid4()),
            started_at=started_at.isoformat(),
            alert=alert,
        )
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO sessions (corpus_id, session_id, started_at, state, run_status, "
                "alert) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    self.corpus_id,
                    record.session_id,
                    started_at,
                    record.state,
                    record.run_status,
                    _json(alert),
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
                "SELECT * FROM sessions WHERE corpus_id = %s AND session_id = %s",
                (self.corpus_id, session_id),
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
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT session_id, state FROM sessions "
                "WHERE corpus_id = %s AND session_id = ANY(%s)",
                (self.corpus_id, ids),
            ).fetchall()
        return {row["session_id"]: row["state"] for row in rows}

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

        Partial: an argument left at None leaves that column alone, so Planning
        attaching a plan cannot erase the diagnosis that preceded it.

        Args:
            session_id: Which session.
            root_cause: The diagnosis, when one was submitted.
            plan: The remediation plan, when one was submitted.
            execution_result: What execution did, when it ran.
            verify_result: The recovery verdict, when verification ran.
            notes: Free text for an operator.
        """
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("root_cause", _json(root_cause)),
            ("plan", _json(plan)),
            ("execution_result", _json(execution_result)),
            ("verify_result", _json(verify_result)),
            ("notes", scrub_optional(notes)),
        ):
            if value is not None:
                assignments.append(f"{column} = %s")
                values.append(value)
        if not assignments:
            return
        values.extend((self.corpus_id, session_id))
        with self._db.write() as conn:
            conn.execute(
                f"UPDATE sessions SET {', '.join(assignments)} "
                "WHERE corpus_id = %s AND session_id = %s",
                values,
            )

    def set_state(self, session_id: str, state: LifecycleState) -> None:
        """Record how far a session's claim has been confirmed.

        The row is read ``FOR UPDATE`` so that the standing checked is the standing
        written against: under READ COMMITTED a plain read is a snapshot two writers can
        both act on.

        It locks where the sequence allocation deliberately does not, because the
        pressure is the opposite one. A run takes hundreds of events from whatever thread
        is running it; a session takes one verdict, at the end of the one pipeline that
        owns it. There is nothing here for a collide-and-retry to buy, and a refusal has
        to be able to say what the record's standing actually is.

        Args:
            session_id: Which session.
            state: The new lifecycle state.

        Raises:
            IllegalTransition: When the record's current state cannot become this one.
        """
        with self._db.write() as conn:
            row = conn.execute(
                "SELECT state FROM sessions WHERE corpus_id = %s AND session_id = %s FOR UPDATE",
                (self.corpus_id, session_id),
            ).fetchone()
            if row is None or not check_transition(LifecycleState(row["state"]), state):
                return
            conn.execute(
                "UPDATE sessions SET state = %s WHERE corpus_id = %s AND session_id = %s",
                (state.value, self.corpus_id, session_id),
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
                "UPDATE sessions SET run_status = %s, current_stage = %s, checkpoint = %s, "
                "awaiting_request_id = %s, ended_at = COALESCE(ended_at, %s) "
                "WHERE corpus_id = %s AND session_id = %s",
                (
                    status.value,
                    scrub_optional(current_stage),
                    _json(checkpoint),
                    awaiting_request_id,
                    ended_at,
                    self.corpus_id,
                    session_id,
                ),
            )

    def mark_distilled(self, session_id: str) -> None:
        """Record that a session's lesson has been written to the corpus.

        Args:
            session_id: Which session.
        """
        with self._db.write() as conn:
            conn.execute(
                "UPDATE sessions SET distilled = TRUE WHERE corpus_id = %s AND session_id = %s",
                (self.corpus_id, session_id),
            )

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
            summary=scrub_text(record.summary),
            prompt=scrub_optional(record.prompt),
            reason=scrub_optional(record.reason),
            created_at=record.created_at or _now().isoformat(),
        )
        with self._db.write() as conn:
            conn.execute(
                "INSERT INTO approvals (corpus_id, request_id, session_id, kind, stage, status, "
                "summary, payload, tool_call_id, tool_name, prompt, response, created_at, "
                "expires_at, decided_at, decided_by, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.corpus_id,
                    stored.request_id,
                    stored.session_id,
                    stored.kind,
                    stored.stage,
                    stored.status,
                    stored.summary,
                    _json(stored.payload),
                    stored.tool_call_id,
                    stored.tool_name,
                    stored.prompt,
                    _json(stored.response),
                    _instant(stored.created_at),
                    _instant_or_none(stored.expires_at),
                    _instant_or_none(stored.decided_at),
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
                "SELECT * FROM approvals WHERE corpus_id = %s AND request_id = %s",
                (self.corpus_id, request_id),
            ).fetchone()
        return None if row is None else _to_approval(row)

    def list_requests(self, *, status: ApprovalStatus | None = None) -> list[ApprovalRecord]:
        """List requests, newest first.

        Args:
            status: Only requests in this status, or all of them.

        Returns:
            The matching records.
        """
        clause = "" if status is None else " AND status = %s"
        params: tuple[str, ...] = (self.corpus_id,) if status is None else (self.corpus_id, status)
        with self._db.read() as conn:
            rows = conn.execute(
                # The request id breaks ties, because two requests raised in the same
                # microsecond still need an order and PostgreSQL has no insertion
                # counter to appeal to.
                f"SELECT * FROM approvals WHERE corpus_id = %s{clause} "
                "ORDER BY created_at DESC, request_id DESC",
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
            row = conn.execute(
                "UPDATE approvals SET status = %s, decided_at = %s, decided_by = %s, reason = %s "
                "WHERE corpus_id = %s AND request_id = %s AND status = 'pending' RETURNING *",
                (
                    status,
                    _now(),
                    scrub_text(decided_by),
                    scrub_optional(reason),
                    self.corpus_id,
                    request_id,
                ),
            ).fetchone()
        return None if row is None else _to_approval(row)

    def expire_due(self, *, now: str) -> list[ApprovalRecord]:
        """Time out every request past its expiry.

        Expiry is compared as instants rather than as strings, so a caller whose
        timestamps end in ``Z`` and one whose timestamps end in ``+00:00`` get the same
        answer. The comparison is the database's because the column is a
        ``TIMESTAMPTZ``, which is also why a malformed expiry cannot be encountered
        here: it was refused when the request was created.

        Args:
            now: The current timestamp.

        Returns:
            The records that expired.
        """
        cutoff = _instant(now)
        with self._db.write() as conn:
            rows = conn.execute(
                "UPDATE approvals SET status = 'timed_out', decided_at = %s "
                "WHERE corpus_id = %s AND status = 'pending' "
                "AND expires_at IS NOT NULL AND expires_at <= %s RETURNING *",
                (cutoff, self.corpus_id, cutoff),
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
                "INSERT INTO agent_runs (corpus_id, agent_run_id, pipeline_id, agent_name, "
                "attempt, status, started_at, ended_at, output, trace_summary, error, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.corpus_id,
                    record.agent_run_id,
                    record.pipeline_id,
                    record.agent_name,
                    record.attempt,
                    record.status,
                    _instant(record.started_at),
                    _instant_or_none(record.ended_at),
                    _json(record.output),
                    _json(record.trace_summary),
                    scrub_optional(record.error),
                    _json(record.metadata),
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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record that a role finished.

        Terminal, and written whole: the arguments describe the run's final state, so a
        None here is "produced nothing" rather than "leave what was there". ``metadata``
        is the exception and merges, because the row already carries the edge the run was
        reached by and a role describing itself is not overruling that.

        ``||`` merges it in the same statement, which is shallow — a key the role names
        replaces that key alone. The embedded backend does the merge in Python to reach
        the same answer, since ``json_patch`` would descend into nested objects.

        Args:
            agent_run_id: Which run.
            status: How it ended.
            output: What it produced.
            trace_summary: A condensed account of what it did.
            error: Why it failed, when it did.
            metadata: What the role recorded about its own run.
        """
        with self._db.write() as conn:
            conn.execute(
                "UPDATE agent_runs SET status = %s, ended_at = %s, output = %s, "
                "trace_summary = %s, error = %s, metadata = metadata || %s "
                "WHERE corpus_id = %s AND agent_run_id = %s",
                (
                    status,
                    _now(),
                    _json(output),
                    _json(trace_summary),
                    scrub_optional(error),
                    _json(metadata or {}),
                    self.corpus_id,
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
                "SELECT * FROM agent_runs WHERE corpus_id = %s AND agent_run_id = %s",
                (self.corpus_id, agent_run_id),
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
                # The run id breaks ties, for the same reason it does among requests.
                "SELECT * FROM agent_runs WHERE corpus_id = %s AND pipeline_id = %s "
                "ORDER BY started_at, agent_run_id",
                (self.corpus_id, session_id),
            ).fetchall()
        return [_to_run(row) for row in rows]

    def append_event(self, event: AgentRunEventDraft) -> AgentRunEventRecord:
        """Append one event, giving it the next place in the run's order.

        The sequence is read and used inside a single statement, and
        ``UNIQUE (corpus_id, agent_run_id, seq)`` decides who wins: under READ COMMITTED
        two concurrent appends to one run can see the same maximum, and the loser is
        retried rather than lost. Retrying is correct because an audit append is
        idempotent in its content and its sequence is assigned here rather than carried
        by the caller, so a second attempt writes the same event one place later.

        A counter row updated with ``RETURNING`` was considered and rejected: it
        serialises appends across the whole table under contention and adds a row whose
        absence is a silent failure. An advisory lock was rejected for the same reason,
        plus being an implementation only one of the two backends could have.

        Args:
            event: The event to append.

        Returns:
            The stored record, carrying the sequence it was given.

        Raises:
            RuntimeError: When the sequence could not be allocated within the retry
                budget, which means sustained contention on one run rather than a race.
        """
        clash: psycopg.errors.UniqueViolation | None = None
        for _ in range(_SEQUENCE_ATTEMPTS):
            try:
                return self._insert_event(event)
            except psycopg.errors.UniqueViolation as violation:
                if violation.diag.constraint_name != _SEQUENCE_CONSTRAINT:
                    # A duplicate event id is the caller's error, not a race.
                    raise
                _log.debug("sequence collision on run %s, retrying", event.agent_run_id)
                clash = violation
        raise RuntimeError(
            f"could not allocate an event sequence for run {event.agent_run_id} in "
            f"{_SEQUENCE_ATTEMPTS} attempts"
        ) from clash

    def events_for(self, agent_run_id: str) -> list[AgentRunEventRecord]:
        """Read a run's events in order.

        Args:
            agent_run_id: Which run.

        Returns:
            The events, ordered by sequence.
        """
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_run_events WHERE corpus_id = %s AND agent_run_id = %s "
                "ORDER BY seq",
                (self.corpus_id, agent_run_id),
            ).fetchall()
        return [_to_event(row) for row in rows]

    def _insert_event(self, event: AgentRunEventDraft) -> AgentRunEventRecord:
        """Insert one event, allocating its sequence in the same statement."""
        payload = scrub_mapping(event.payload)
        with self._db.write() as conn:
            row = conn.execute(
                "INSERT INTO agent_run_events (corpus_id, event_id, agent_run_id, pipeline_id, "
                "seq, event_type, role, tool_call_id, tool_name, payload, created_at) "
                "VALUES (%s, %s, %s, %s, "
                "(SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_run_events "
                " WHERE corpus_id = %s AND agent_run_id = %s), "
                "%s, %s, %s, %s, %s, %s) RETURNING seq",
                (
                    self.corpus_id,
                    event.event_id,
                    event.agent_run_id,
                    event.pipeline_id,
                    self.corpus_id,
                    event.agent_run_id,
                    event.event_type,
                    event.role,
                    event.tool_call_id,
                    event.tool_name,
                    Jsonb(payload),
                    _instant(event.created_at),
                ),
            ).fetchone()
        if row is None:
            # RETURNING on an INSERT that did not raise always has a row, so this is a
            # driver that has stopped meaning what it says rather than a case to handle.
            raise RuntimeError(f"no sequence came back for event {event.event_id}")
        return AgentRunEventRecord(
            event_id=event.event_id,
            agent_run_id=event.agent_run_id,
            pipeline_id=event.pipeline_id,
            seq=int(row["seq"]),
            event_type=event.event_type,
            payload=payload,
            created_at=event.created_at,
            role=event.role,
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
        )


class PostgresRecordIndex:
    """:class:`~kira.memory.ports.TextSearch` over two generated tsvectors.

    A candidate generator. Nothing here orders by relevance; see the module docstring.
    """

    def __init__(
        self,
        connection: _Pool | str,
        *,
        schema: str = "kira_memory",
        corpus_id: str = "default",
    ) -> None:
        """Open or adopt a database and create the index if it is not there.

        Args:
            connection: A connection string, or the pool of the :class:`PostgresStore`
                this index belongs to.
            schema: Schema holding the table, when opening a database of its own.
            corpus_id: The campaign this index reads and writes.
        """
        self.corpus_id = corpus_id
        self._db = connection if isinstance(connection, _Pool) else _Pool(connection, schema)
        self._db.script(_RECORD_SCHEMA)

    def close(self) -> None:
        """Close the connections. Worth calling only on a standalone index."""
        self._db.close()

    def upsert_summary(self, memory: SessionMemory) -> None:
        """Write or replace one record.

        Args:
            memory: The distilled record.
        """
        values = [self.corpus_id, *(scrub(getattr(memory, name)) for name in _RECORD_FIELDS)]
        with self._db.write() as conn:
            conn.execute(
                f"INSERT INTO records (corpus_id, {_RECORD_COLUMNS}) "
                f"VALUES (%s, {_RECORD_PLACEHOLDERS}) "
                f"ON CONFLICT (corpus_id, session_id) DO UPDATE "
                f"SET {_RECORD_UPDATES}, written_at = now()",
                values,
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

        Terms are ORed, and every one of them is used: the prototype truncated the query
        to 32 tokens, which silently decided which documents the ranker was allowed to
        see. Each lexeme is quoted so that a token which happens to spell a tsquery
        operator stays a search term.

        Args:
            text: The query, used to select a pool.
            limit: Ceiling on the pool.
            scope: Which projection the query is aimed at.
            exclude: Session ids to withhold.

        Returns:
            Candidate records, in write order, which carries no relevance.
        """
        if not tokenize(text) or limit <= 0:
            return []
        query = _CANDIDATE_QUERY.format(columns=_RECORD_COLUMNS, vector=_vector(scope))
        with self._db.read() as conn:
            rows = conn.execute(query, (text, self.corpus_id, list(exclude), limit)).fetchall()
        return [_to_memory(row) for row in rows]

    def scan_summaries(
        self,
        pattern: str,
        *,
        n_results: int = 5,
        scope: SearchScope = "evidence",
    ) -> list[SessionMemory]:
        """Return only records matching a regular expression.

        Filtered here rather than with ``~*`` so that both backends answer a scan with
        the same regular expression dialect; PostgreSQL's is POSIX and Python's is not,
        and a pattern that means two things depending on where the corpus lives is worse
        than a scan over a few hundred rows. An invalid pattern returns nothing instead
        of raising, which costs a role one turn.

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
                f"SELECT {_RECORD_COLUMNS} FROM records WHERE corpus_id = %s "
                "ORDER BY written_at, session_id",
                (self.corpus_id,),
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
                f"SELECT {_RECORD_COLUMNS} FROM records WHERE corpus_id = %s AND session_id = %s",
                (self.corpus_id, session_id),
            ).fetchone()
        return None if row is None else _to_memory(row)

    def delete_summary(self, session_id: str) -> None:
        """Remove one record.

        Args:
            session_id: Which session.
        """
        with self._db.write() as conn:
            conn.execute(
                "DELETE FROM records WHERE corpus_id = %s AND session_id = %s",
                (self.corpus_id, session_id),
            )

    def count(self) -> int:
        """Report how many records are stored.

        Returns:
            The record count.
        """
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM records WHERE corpus_id = %s", (self.corpus_id,)
            ).fetchone()
        return 0 if row is None else int(row["total"])


def _vector(scope: SearchScope) -> str:
    return "lesson_tsv" if scope == "lesson" else "evidence_tsv"


def _projection(memory: SessionMemory, scope: SearchScope) -> str:
    return memory.lesson_text if scope == "lesson" else memory.evidence_text


def _to_session(row: _Row) -> SessionRecord:
    return SessionRecord(
        session_id=row["session_id"],
        started_at=_iso(row["started_at"]) or "",
        ended_at=_iso(row["ended_at"]),
        state=row["state"],
        run_status=row["run_status"],
        current_stage=row["current_stage"],
        checkpoint=row["checkpoint"],
        awaiting_request_id=row["awaiting_request_id"],
        alert=row["alert"] or {},
        root_cause=row["root_cause"],
        plan=row["plan"],
        execution_result=row["execution_result"],
        verify_result=row["verify_result"],
        distilled=row["distilled"],
        notes=row["notes"],
    )


def _to_approval(row: _Row) -> ApprovalRecord:
    return ApprovalRecord(
        request_id=row["request_id"],
        session_id=row["session_id"],
        kind=row["kind"],
        stage=row["stage"],
        status=row["status"],
        summary=row["summary"],
        payload=row["payload"] or {},
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        prompt=row["prompt"],
        response=row["response"],
        created_at=_iso(row["created_at"]) or "",
        expires_at=_iso(row["expires_at"]),
        decided_at=_iso(row["decided_at"]),
        decided_by=row["decided_by"],
        reason=row["reason"],
    )


def _to_run(row: _Row) -> AgentRunRecord:
    return AgentRunRecord(
        agent_run_id=row["agent_run_id"],
        pipeline_id=row["pipeline_id"],
        agent_name=row["agent_name"],
        attempt=row["attempt"],
        status=row["status"],
        started_at=_iso(row["started_at"]) or "",
        ended_at=_iso(row["ended_at"]),
        output=row["output"],
        trace_summary=row["trace_summary"],
        error=row["error"],
        metadata=row["metadata"] or {},
    )


def _to_event(row: _Row) -> AgentRunEventRecord:
    return AgentRunEventRecord(
        event_id=row["event_id"],
        agent_run_id=row["agent_run_id"],
        pipeline_id=row["pipeline_id"],
        seq=row["seq"],
        event_type=row["event_type"],
        payload=row["payload"] or {},
        created_at=_iso(row["created_at"]) or "",
        role=row["role"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
    )


def _to_memory(row: _Row) -> SessionMemory:
    return SessionMemory(**{name: row[name] for name in _RECORD_FIELDS})


if TYPE_CHECKING:

    def _ports_are_satisfied(
        store: PostgresStore, index: PostgresRecordIndex
    ) -> tuple["SessionStore", "ApprovalStore", "RunStore", "TextSearch"]:
        """Static proof of conformance. Type-checked, never called."""
        return store, store, store, index
