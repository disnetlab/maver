"""The server backend against the shared contract.

The contract is the test budget, so almost nothing here is PostgreSQL-specific. What is
left are the claims the shared suite cannot make: that a NUL is a byte this database
genuinely refuses rather than merely dislikes, that two writers racing for one event
sequence both keep their event, and that a wait on this server is bounded at all.

Needs a live database. Set POSTGRES_TEST_DSN to run it, for instance:

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=maver -e POSTGRES_USER=maver \\
        -e POSTGRES_DB=maver postgres:17-alpine
    POSTGRES_TEST_DSN=postgresql://maver:maver@127.0.0.1:55432/maver uv run pytest \\
        tests/unit/test_postgres_backend.py

Each test gets a schema of its own, dropped afterwards, so the suite can run repeatedly
against one server without inheriting the last run's rows.
"""

import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from typing import Any

import pytest
from memory_contract import (
    NUL,
    MemoryStoreContract,
    OperationalStore,
    a_record,
    a_run,
    an_event,
)

from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.memory.backends.schema import STORE_SCHEMA_VERSION
from maver.memory.backends.sqlite import SqliteStore
from maver.memory.lexemes import lexemes
from maver.memory.ports import TextSearch
from maver.orchestration.outcome_feedback import submit_feedback

DSN = os.environ.get("POSTGRES_TEST_DSN", "")

# Before importing the backend, which needs the driver the extra installs.
psycopg = pytest.importorskip("psycopg", reason="the postgres extra is not installed")

from psycopg import sql  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from maver.memory.backends.postgres import PostgresStore  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_TEST_DSN is unset, so there is no database"),
]


@contextmanager
def _temporary_schema() -> Iterator[str]:
    """A schema of this test's own, dropped whatever happens to it."""
    name = f"maver_test_{uuid.uuid4().hex}"
    try:
        yield name
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


@pytest.fixture
def schema() -> Iterator[str]:
    with _temporary_schema() as name:
        yield name


@pytest.fixture
def postgres_store(schema: str) -> Iterator[PostgresStore]:
    store = PostgresStore(DSN, schema=schema)
    yield store
    store.close()


class TestPostgresBackend(MemoryStoreContract):
    @pytest.fixture
    def store(self, postgres_store: PostgresStore) -> OperationalStore:
        return postgres_store

    @pytest.fixture
    def records(self, postgres_store: PostgresStore) -> TextSearch:
        return postgres_store.records

    @pytest.fixture
    def reopen(self, schema: str) -> Callable[[], PostgresStore]:
        return lambda: PostgresStore(DSN, schema=schema)

    @pytest.fixture
    def stamp_version(self, schema: str) -> Callable[[int], None]:
        def stamp(version: int) -> None:
            with closing(_raw_connection(schema)) as conn:
                conn.execute("UPDATE schema_version SET version = %s", (version,))

        return stamp


def test_version_eight_adds_learning_intents(schema: str) -> None:
    with closing(PostgresStore(DSN, schema=schema)) as store:
        session = store.create_session({"alertname": "HighErrorRate"})
    with _raw_connection(schema) as conn:
        conn.execute("DROP TABLE learning_intents")
        conn.execute("UPDATE schema_version SET version = 8")
    with closing(PostgresStore(DSN, schema=schema)) as reopened:
        assert reopened.schema_version == STORE_SCHEMA_VERSION
        assert reopened.pending_learning_intents() == []
        assert reopened.get_session(session.session_id) is not None


def test_version_nine_adds_explicit_correction_authority(schema: str) -> None:
    with closing(PostgresStore(DSN, schema=schema)) as store:
        store.records.upsert_summary(a_record("old", evidence="pod failure"))
    with _raw_connection(schema) as conn:
        conn.execute("ALTER TABLE records DROP COLUMN authoritative_correction")
        conn.execute("UPDATE schema_version SET version = 9")
    with closing(PostgresStore(DSN, schema=schema)) as reopened:
        assert reopened.schema_version == STORE_SCHEMA_VERSION
        old = reopened.records.open_record("old")
        assert old is not None and old.authoritative_correction is False
        reopened.records.upsert_summary(replace(old, authoritative_correction=True))
    with closing(PostgresStore(DSN, schema=schema)) as reopened:
        assert reopened.records.open_record("old").authoritative_correction is True


def test_old_feedback_table_accepts_new_reviews_after_sequence_migration(schema: str) -> None:
    """ALTER appends the sequence column, so inserts must name their columns."""
    with closing(PostgresStore(DSN, schema=schema)) as store:
        session = store.create_session({})
        store.set_run_status(session.session_id, RunStatus.COMPLETED)
        store.set_state(session.session_id, LifecycleState.INVALIDATED)

    with _raw_connection(schema) as conn:
        conn.execute("DROP TABLE outcome_feedback")
        conn.execute(
            "CREATE TABLE outcome_feedback ("
            "corpus_id TEXT NOT NULL, feedback_id TEXT NOT NULL, session_id TEXT NOT NULL, "
            "verdict TEXT NOT NULL, decided_by TEXT NOT NULL, reason TEXT NOT NULL, "
            "evidence TEXT NOT NULL, corrected_cause TEXT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (corpus_id, feedback_id))"
        )
        for feedback_id, when in (
            ("earlier", "2026-09-22T12:00:00Z"),
            ("later", "2026-09-22T12:00:01Z"),
        ):
            conn.execute(
                "INSERT INTO outcome_feedback VALUES "
                "('default', %s, %s, 'rejected', 'ops', 'reason', 'evidence', '', %s)",
                (feedback_id, session.session_id, when),
            )
        conn.execute("UPDATE schema_version SET version = 7")

    with closing(PostgresStore(DSN, schema=schema)) as reopened:
        history = reopened.feedback_for(session.session_id)
        assert [(item.feedback_id, item.sequence) for item in history] == [
            ("earlier", 1),
            ("later", 2),
        ]
        submitted = submit_feedback(
            reopened,
            session_id=session.session_id,
            verdict="confirmed",
            decided_by="ops",
            reason="reviewed",
            evidence="recovered",
        )
        assert submitted.sequence == 3
        assert reopened.get_session(session.session_id).state == LifecycleState.HUMAN_VERIFIED.value


def test_the_byte_the_store_strips_is_one_this_database_refuses(
    postgres_store: PostgresStore, schema: str
) -> None:
    """Scrubbing NUL is load-bearing here rather than hygiene.

    The contract already pins that the byte comes back stripped. What it cannot say is
    why the write boundary is the place to do it: the same value written unscrubbed is
    rejected outright, and in the prototype that rejection made an episode skip every
    submission and produce a stub result that read as an agent which never answered.
    """
    session = postgres_store.create_session({"detail": f"tool{NUL}output"})

    with _raw_connection(schema) as conn, pytest.raises(psycopg.Error):
        conn.execute(
            "UPDATE sessions SET alert = %s WHERE session_id = %s",
            (Jsonb({"detail": f"tool{NUL}output"}), session.session_id),
        )


def test_the_writer_that_loses_the_sequence_race_is_retried_not_lost(
    postgres_store: PostgresStore, schema: str
) -> None:
    """Under READ COMMITTED two appends to one run can read the same maximum sequence.

    The race is produced rather than waited for. A rival transaction holds an
    uncommitted event at sequence 1, which makes the store's insert wait on the unique
    index; committing the rival turns that wait into the violation the retry exists for,
    and the event has to come back one place later instead of being lost.
    """
    session = postgres_store.create_session({})
    postgres_store.start_run(a_run("run-1", session.session_id))

    with _raw_connection(schema, autocommit=False) as rival:
        rival.execute(
            "INSERT INTO agent_run_events (corpus_id, event_id, agent_run_id, seq, event_type, "
            "payload, created_at) VALUES (%s, 'rival', 'run-1', 1, 'tool_call', %s, now())",
            (postgres_store.corpus_id, Jsonb({})),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                postgres_store.append_event, an_event("event-1", "run-1", session.session_id)
            )
            # Committed unconditionally, even when the wait never appeared, because the
            # append blocks until it is and an assertion first would hang the suite.
            waited = _waited_on_the_index(rival)
            rival.commit()
            appended = pending.result(timeout=30)

    assert waited, "the append should have been blocked by the rival's uncommitted sequence"
    assert appended.seq == 2
    assert [event.seq for event in postgres_store.events_for("run-1")] == [1, 2]


def test_a_call_blocked_on_a_lock_gives_up_rather_than_waiting_forever(schema: str) -> None:
    """Nothing above this can end the wait, so the wait has to end itself.

    These calls are made from the event loop running the incident, and a loop cannot time
    out what it is inside of: with the table locked, a search held the loop and an
    eight-second ``asyncio.wait_for`` around it never fired. Unbounded, a lock nobody
    released stopped the whole run instead of failing one call — which is how a hung cell
    came to be recorded as a model that never answered.
    """
    with closing(PostgresStore(DSN, schema=schema, statement_timeout_ms=250)) as store:
        session = store.create_session({})
        # In a thread, and outside the lock, so that a regression here fails on this
        # clock and releases the lock on the way out. Called inline it would hang, which
        # is the defect and would also be the whole suite.
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            _raw_connection(schema, autocommit=False) as rival,
        ):
            rival.execute("LOCK TABLE sessions IN ACCESS EXCLUSIVE MODE")
            # A transition the graph allows, so that what this waits on is the lock and
            # nothing else.
            blocked = pool.submit(store.set_state, session.session_id, LifecycleState.INVALIDATED)
            with pytest.raises(psycopg.errors.QueryCanceled):
                blocked.result(timeout=10)
            rival.rollback()


def test_the_bounds_the_deployment_set_itself_are_left_alone(schema: str) -> None:
    """An operator who tuned their DSN knows something this backend does not.

    Written in the keyword spelling, which is the one the rest of this file never
    exercises: POSTGRES_TEST_DSN is documented as a URI, and both have to parse.
    """
    tuned = make_conninfo(DSN, connect_timeout=3, options="-c statement_timeout=7s")
    with closing(PostgresStore(tuned, schema=schema)) as store:
        assert _bounds(store) == ("3", "7s")
    with closing(PostgresStore(DSN, schema=schema)) as store:
        assert _bounds(store) == ("10", "30s")


def _bounds(store: PostgresStore) -> tuple[str, str]:
    """The connect and statement bounds in force on a connection the store opened.

    Read off the pool rather than off a connection made here, because the claim is about
    what the store puts on the sessions it uses, not about what the DSN says.
    """
    with store._db.read() as conn:
        timeout = conn.execute("SHOW statement_timeout").fetchone()
        assert timeout is not None
        return conn.info.get_parameters()["connect_timeout"], timeout["statement_timeout"]


def _raw_connection(schema: str, *, autocommit: bool = True) -> Any:
    """A connection outside the store, for writing what the store would not."""
    conn = psycopg.connect(DSN, autocommit=autocommit)
    conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
    return conn


def _waited_on_the_index(conn: Any, timeout_s: float = 15.0) -> bool:
    """Report whether another backend is blocked on a lock, within the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        row = conn.execute(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND state = 'active' AND pid <> pg_backend_pid()"
        ).fetchone()
        if row and row[0]:
            return True
        time.sleep(0.05)
    return False


# The `records` table as it was before the shared normaliser: two vectors built by the
# english configuration straight off the raw text. Written out here rather than imported
# so that changing the current schema cannot quietly change what "the old shape" means.
_OLD_SCHEMA = """
CREATE TABLE records (
    corpus_id      TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    evidence_text  TEXT NOT NULL DEFAULT '',
    lesson_text    TEXT NOT NULL DEFAULT '',
    incident_summary TEXT NOT NULL DEFAULT '',
    lifecycle_state  TEXT NOT NULL DEFAULT 'pending',
    fault_category   TEXT NOT NULL DEFAULT '',
    alert_name       TEXT NOT NULL DEFAULT '',
    namespace        TEXT NOT NULL DEFAULT '',
    affected_resource TEXT NOT NULL DEFAULT '',
    verification_verdict TEXT NOT NULL DEFAULT '',
    evaluation_label     TEXT NOT NULL DEFAULT '',
    evaluation_source    TEXT NOT NULL DEFAULT '',
    evaluation_diagnosis_success  BOOLEAN,
    evaluation_diagnosis_score    DOUBLE PRECISION,
    evaluation_mitigation_success BOOLEAN,
    corrected_cause  TEXT NOT NULL DEFAULT '',
    written_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    evidence_tsv TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(evidence_text, ''))) STORED,
    lesson_tsv   TSVECTOR GENERATED ALWAYS AS
        (to_tsvector('english', coalesce(lesson_text, ''))) STORED,
    PRIMARY KEY (corpus_id, session_id)
);
CREATE INDEX idx_records_evidence_tsv ON records USING GIN (evidence_tsv);
CREATE INDEX idx_records_lesson_tsv ON records USING GIN (lesson_tsv);
"""


def test_a_database_written_before_the_normaliser_is_reindexed_on_open(schema: str) -> None:
    """An existing corpus has to answer the new query, not be silently left behind."""
    with closing(_raw_connection(schema)) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        conn.execute(_OLD_SCHEMA)
        conn.execute(
            "INSERT INTO records (corpus_id, session_id, evidence_text, lesson_text) "
            "VALUES ('default', 'old', %s, %s)",
            (
                "the checkout pods were crashing, RunContainerError/CrashLoopBackOff",
                "raised the memory limit",
            ),
        )

    with closing(PostgresStore(DSN, schema=schema)) as store:
        assert store.schema_version == STORE_SCHEMA_VERSION
        records = store.records
        assert [m.session_id for m in records.candidates("pod crash")] == ["old"]
        assert [m.session_id for m in records.candidates("raising limits", scope="lesson")] == [
            "old"
        ]
        assert records.candidates("the and of") == []
        # The old vector held this as one lexeme, because the parser reads a slash-joined
        # pair as a file name, so the pool it built could only be reached by the pair.
        assert [m.session_id for m in records.candidates("crashloopbackoff")] == ["old"]
        stored = records.open_record("old")
        assert stored is not None
        assert stored.evidence_text.startswith("the checkout"), "raw text is untouched"

    # Idempotent, and writable: a second open finds the new shape and rebuilds nothing.
    with closing(PostgresStore(DSN, schema=schema)) as store:
        store.records.upsert_summary(a_record("new", evidence="coredns resolution failures"))
        assert store.records.count() == 2
        assert [m.session_id for m in store.records.candidates("pod crash")] == ["old"]
        assert [m.session_id for m in store.records.candidates("resolve failure")] == ["new"]

    assert _stamped_versions(schema) == [STORE_SCHEMA_VERSION], "one row, and it is current"
    with _temporary_schema() as fresh, closing(PostgresStore(DSN, schema=fresh)):
        assert _record_layout(schema) == _record_layout(fresh), (
            "a migrated table is the current layout, not a shape of its own"
        )


# The `records` table as it was before the environment's labels: the vectors are over the
# normalised terms already, and there is no labels column. Written out here for the reason
# the shape above is.
_SCHEMA_WITHOUT_LABELS = """
CREATE TABLE records (
    corpus_id      TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    evidence_text  TEXT NOT NULL DEFAULT '',
    lesson_text    TEXT NOT NULL DEFAULT '',
    incident_summary TEXT NOT NULL DEFAULT '',
    lifecycle_state  TEXT NOT NULL DEFAULT 'pending',
    fault_category   TEXT NOT NULL DEFAULT '',
    alert_name       TEXT NOT NULL DEFAULT '',
    namespace        TEXT NOT NULL DEFAULT '',
    affected_resource TEXT NOT NULL DEFAULT '',
    verification_verdict TEXT NOT NULL DEFAULT '',
    evaluation_label     TEXT NOT NULL DEFAULT '',
    evaluation_source    TEXT NOT NULL DEFAULT '',
    evaluation_diagnosis_success  BOOLEAN,
    evaluation_diagnosis_score    DOUBLE PRECISION,
    evaluation_mitigation_success BOOLEAN,
    corrected_cause  TEXT NOT NULL DEFAULT '',
    evidence_terms   TEXT NOT NULL DEFAULT '',
    lesson_terms     TEXT NOT NULL DEFAULT '',
    written_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    evidence_tsv TSVECTOR GENERATED ALWAYS AS
        (CASE WHEN evidence_terms = '' THEN ''::tsvector
              ELSE array_to_tsvector(string_to_array(evidence_terms, ' ')) END) STORED,
    lesson_tsv   TSVECTOR GENERATED ALWAYS AS
        (CASE WHEN lesson_terms = '' THEN ''::tsvector
              ELSE array_to_tsvector(string_to_array(lesson_terms, ' ')) END) STORED,
    PRIMARY KEY (corpus_id, session_id)
);
CREATE INDEX idx_records_evidence_tsv ON records USING GIN (evidence_tsv);
CREATE INDEX idx_records_lesson_tsv ON records USING GIN (lesson_tsv);
CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TIMESTAMPTZ NOT NULL);
INSERT INTO schema_version VALUES (2, now());
"""


def test_a_database_written_before_labels_gains_them_empty_on_open(schema: str) -> None:
    """A corpus from the previous layout keeps its records and answers the same queries.

    The step adds a column and nothing else, so what the records say must be untouched:
    an unlabelled record is one no adapter has claimed, not one that has been rewritten.
    """
    with closing(_raw_connection(schema)) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
        conn.execute(_SCHEMA_WITHOUT_LABELS)
        conn.execute(
            "INSERT INTO records (corpus_id, session_id, evidence_text, evidence_terms) "
            "VALUES ('default', 'old', %s, %s)",
            (
                "the checkout pods were crashing",
                " ".join(lexemes("the checkout pods were crashing")),
            ),
        )

    with closing(PostgresStore(DSN, schema=schema)) as store:
        assert store.schema_version == STORE_SCHEMA_VERSION
        records = store.records
        assert [m.session_id for m in records.candidates("pod crash")] == ["old"]
        stored = records.open_record("old")
        assert stored is not None
        assert stored.labels == {}
        assert records.session_ids_with_label("external_key", "anything") == []

        records.upsert_summary(a_record("new", labels={"external_key": "k1"}))
        assert records.session_ids_with_label("external_key", "k1") == ["new"]

    assert _stamped_versions(schema) == [STORE_SCHEMA_VERSION], "one row, and it is current"
    with _temporary_schema() as fresh, closing(PostgresStore(DSN, schema=fresh)):
        assert _record_layout(schema) == _record_layout(fresh), (
            "a migrated table is the current layout, not a shape of its own"
        )


def test_an_unstamped_database_in_the_current_shape_is_stamped_and_left_alone(
    schema: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Not every database without a version is an old one.

    One written by the build that migrated the vectors but did not yet stamp anything is
    already current, and reading its shape has to say so. Rebuilding it would be correct
    and would still be a table rewrite taken for nothing.
    """
    with closing(PostgresStore(DSN, schema=schema)) as store:
        store.records.upsert_summary(a_record("s1", evidence="the checkout pods were crashing"))
    with closing(_raw_connection(schema)) as conn:
        conn.execute("DROP TABLE schema_version")

    with (
        caplog.at_level(logging.INFO, logger="maver.memory.backends.schema"),
        closing(PostgresStore(DSN, schema=schema)) as reopened,
    ):
        assert reopened.schema_version == STORE_SCHEMA_VERSION
        assert [m.session_id for m in reopened.records.candidates("pod crash")] == ["s1"]
    assert not [record for record in caplog.records if "migrating" in record.getMessage()]
    assert _stamped_versions(schema) == [STORE_SCHEMA_VERSION]


def _stamped_versions(schema: str) -> list[int]:
    with closing(_raw_connection(schema)) as conn:
        return [row[0] for row in conn.execute("SELECT version FROM schema_version").fetchall()]


def _record_layout(schema: str) -> list[tuple[str, str]]:
    """Every live column of the record table, with the expression a generated one uses.

    Read off the catalogue rather than compared as DDL, because a column dropped and
    added back — which is the only way a generated column's expression can be changed —
    leaves a table that is the same shape and a different DDL script.
    """
    with closing(_raw_connection(schema)) as conn:
        rows = conn.execute(
            "SELECT a.attname AS name, coalesce(pg_get_expr(d.adbin, d.adrelid), '') AS expression "
            "FROM pg_attribute a LEFT JOIN pg_attrdef d "
            "  ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
            "WHERE a.attrelid = to_regclass('records') AND a.attnum > 0 AND NOT a.attisdropped "
            "ORDER BY a.attname"
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


# Forty records and fifteen queries in the shapes real incident text takes: hyphenated
# workload names, image tags, addresses, versions, slash-joined error pairs, inflected
# prose, and stopwords. Small enough to read, wide enough that a backend which tokenised
# differently could not pass by luck.
_EVIDENCE = [
    "the checkout pods were crashing repeatedly with OOMKilled",
    "hotel-reservation frontend returns 503 for every request",
    "astronomy-shop ad service exits immediately on start",
    "social-network user-service times out calling mongodb",
    "coredns resolution failures across kube-system",
    "the node became unschedulable under disk pressure",
    "RunContainerError/CrashLoopBackOff on all eight deployments",
    "image yinfangchen/hotelreservation:latest cannot be pulled",
    "nginx:1.21.3 replaced by a tag that does not exist",
    "the readiness probe failed on 10.96.0.1 for ten minutes",
    "cpu throttling raised p99 latency to 250ms",
    "memory limit 512Mi exceeded by the recommendation service",
    "pvc mongodb-data stayed Pending, unbound and unclaimed",
    "the network policy blocked traffic between namespaces",
    "certificate expired and the tls handshake was refused",
    "configmap otel-collector-agent points at a missing endpoint",
    "deployment/checkout was rolled back to the previous revision",
    "replicas scaled to zero by a stale horizontal autoscaler",
    "checkout-7d9f8b6c5-x2klm restarted 14 times in an hour",
    "jaeger traces show 500 responses from the payment path",
    "rate limiting returned 429 for a third of requests",
    "the sidecar container was added and never became ready",
    "internally routed requests error before reaching the service",
    "kubelet evicted the pod for ephemeral storage",
    "the service selector matched no pods at all",
    "an init container hung waiting on a dependency",
    "etcd was slow and the api server flapped",
    "the admission webhook rejected every create",
    "conntrack table exhaustion on the worker node",
    "the pod cidr ran out of addresses",
    "a wrong port made the service unreachable",
    "the load generator saturated the frontend",
    "mongodb connection refused after the restart",
    "the geo service returned empty results",
    "profile pods crashed on a malformed config",
    "search latency doubled after the rollout",
    "reservation writes failed with a duplicate key",
    "the recommendation cache was never warmed",
    "logs show 'executable file not found in $PATH'",
    "no firing alerts remained after the rollout completed",
]
_QUERIES = [
    "pod crash",
    "crashing pods restarting",
    "the and of",
    "",
    "oomkilled memory limit",
    "crashloopbackoff",
    "hotel-reservation frontend unreachable",
    "deployment/checkout rolled back",
    "10.96.0.1 probe failing",
    "nginx:1.21.3 image pull",
    "internally routed errors",
    "added a sidecar",
    "coredns kube-system resolution",
    "checkout-7d9f8b6c5-x2klm restarts",
    "nothingmatchesthisterm",
]


@pytest.mark.parametrize("scope", ["evidence", "lesson"])
def test_the_two_backends_build_the_same_pool_for_the_same_corpus(schema: str, scope: Any) -> None:
    """The claim the shared normaliser exists to make, checked against both databases.

    A contract test cannot make it: it runs each backend in its own process against its
    own fixture, so it can say each is self-consistent and never that the two agree. This
    writes one corpus into both and compares candidate sets query by query.
    """
    corpus = [
        a_record(f"s{index}", evidence=evidence, lesson=f"lesson for {evidence}")
        for index, evidence in enumerate(_EVIDENCE)
    ]
    with (
        closing(SqliteStore(":memory:")) as sqlite,
        closing(PostgresStore(DSN, schema=schema)) as postgres,
    ):
        for record in corpus:
            sqlite.records.upsert_summary(record)
            postgres.records.upsert_summary(record)

        pools = {}
        for query in _QUERIES:
            embedded = {m.session_id for m in sqlite.records.candidates(query, scope=scope)}
            server = {m.session_id for m in postgres.records.candidates(query, scope=scope)}
            assert embedded == server, f"{query!r} on {scope} pools differently"
            pools[query] = embedded

    # Two backends that both returned nothing for everything would agree and mean
    # nothing, so what they agreed on is worth stating.
    assert pools["pod crash"], "an inflected query has to reach the record it describes"
    assert not pools["the and of"], "a stopword is not a search term"
    assert not pools["nothingmatchesthisterm"]
    assert sum(len(pool) for pool in pools.values()) > len(_QUERIES)


def test_durable_intake_receipts_are_idempotent_and_corpus_scoped(schema: str) -> None:
    """One delivery maps to one initialized session within its corpus."""
    with (
        closing(PostgresStore(DSN, schema=schema, corpus_id="one")) as first,
        closing(PostgresStore(DSN, schema=schema, corpus_id="one")) as rival,
        closing(PostgresStore(DSN, schema=schema, corpus_id="two")) as second,
    ):
        created, opened = first.open_idempotent_session(
            "kafka:group:alerts:0",
            "7",
            "hash-a",
            {"alertname": "A"},
            {"selection": {"name": "multi"}},
        )
        assert opened
        assert created.run_status == "running"
        again, opened = first.open_idempotent_session(
            "kafka:group:alerts:0", "7", "hash-a", {"alertname": "A"}, {}
        )
        assert not opened and again.session_id == created.session_id
        assert again.runtime == {"selection": {"name": "multi"}}
        with pytest.raises(ValueError, match="different request"):
            first.open_idempotent_session("kafka:group:alerts:0", "7", "hash-b", {}, {})
        assert (
            first.bind_intake_receipt(
                "kafka:group:alerts:0", "8", "hash-a", created.session_id
            ).session_id
            == created.session_id
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(
                pool.map(
                    lambda store: store.open_idempotent_session(
                        "kafka:group:alerts:0", "9", "hash-c", {"alertname": "C"}, {}
                    ),
                    (first, rival),
                )
            )
        assert sorted(admitted for _, admitted in attempts) == [False, True]
        assert attempts[0][0].session_id == attempts[1][0].session_id
        other, opened = second.open_idempotent_session(
            "kafka:group:alerts:0", "7", "hash-b", {"alertname": "B"}, {}
        )
        assert opened and other.session_id != created.session_id
        assert second.get_intake_receipt("kafka:group:alerts:0", "8", "hash-a") is None
    with closing(PostgresStore(DSN, schema=schema, corpus_id="one")) as reopened:
        restored = reopened.get_intake_receipt("kafka:group:alerts:0", "7", "hash-a")
        assert restored is not None and restored.session_id == created.session_id


def test_quarantine_is_durable_idempotent_and_corpus_scoped(schema: str) -> None:
    """A poison delivery remains inspectable after a store restart."""
    with closing(PostgresStore(DSN, schema=schema, corpus_id="one")) as store:
        saved = store.quarantine("kafka:group:alerts:0", "7", "bm90IEpTT04=", "invalid JSON")
        duplicate = store.quarantine("kafka:group:alerts:0", "7", "changed", "later reason")
        assert duplicate.record_id == saved.record_id
        assert duplicate.raw_base64 == saved.raw_base64
        store.mark_quarantine_retried(saved.record_id, "session-1")
    with (
        closing(PostgresStore(DSN, schema=schema, corpus_id="one")) as reopened,
        closing(PostgresStore(DSN, schema=schema, corpus_id="two")) as other,
    ):
        restored = reopened.get_quarantine(saved.record_id)
        assert restored is not None and restored.retry_session_id == "session-1"
        assert reopened.list_quarantine() == [restored]
        assert other.get_quarantine(saved.record_id) is None
        assert other.list_quarantine() == []


def test_stamped_v4_sessions_migrate_to_all_new_operational_tables(schema: str) -> None:
    """The combined layout preserves an old session while adding three new stores."""
    with closing(PostgresStore(DSN, schema=schema)) as store:
        old = store.create_session({"alertname": "existing"})
        store.update_artefacts(old.session_id, root_cause={"summary": "original"})
    with closing(_raw_connection(schema)) as conn:
        for table in ("intake_receipts", "intake_quarantine", "outcome_feedback"):
            conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
        conn.execute("UPDATE schema_version SET version = 4")
    with closing(PostgresStore(DSN, schema=schema)) as store:
        assert store.schema_version == STORE_SCHEMA_VERSION
        preserved = store.get_session(old.session_id)
        assert preserved is not None and preserved.root_cause == {"summary": "original"}
        receipt, opened = store.open_idempotent_session("http", "retry", "hash", {}, {})
        assert opened and store.get_intake_receipt("http", "retry", "hash") == receipt
        quarantined = store.quarantine("kafka:g:t:0", "7", "", "Kafka tombstone")
        assert store.get_quarantine(quarantined.record_id) == quarantined
        with store._db.read() as conn:
            assert conn.execute("SELECT count(*) AS n FROM outcome_feedback").fetchone()["n"] == 0
