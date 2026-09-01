"""The server backend against the shared contract.

The contract is the test budget, so almost nothing here is PostgreSQL-specific. What is
left are the claims the shared suite cannot make: that a NUL is a byte this database
genuinely refuses rather than merely dislikes, that two writers racing for one event
sequence both keep their event, and that a wait on this server is bounded at all.

Needs a live database. Set POSTGRES_TEST_DSN to run it, for instance:

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=kira -e POSTGRES_USER=kira \\
        -e POSTGRES_DB=kira postgres:17-alpine
    POSTGRES_TEST_DSN=postgresql://kira:kira@127.0.0.1:55432/kira uv run pytest \\
        tests/unit/test_postgres_backend.py

Each test gets a schema of its own, dropped afterwards, so the suite can run repeatedly
against one server without inheriting the last run's rows.
"""

import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

import pytest
from memory_contract import NUL, MemoryStoreContract, OperationalStore, a_run, an_event

from kira.contracts.lifecycle import LifecycleState
from kira.memory.ports import TextSearch

DSN = os.environ.get("POSTGRES_TEST_DSN", "")

# Before importing the backend, which needs the driver the extra installs.
psycopg = pytest.importorskip("psycopg", reason="the postgres extra is not installed")

from psycopg import sql  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402

from kira.memory.backends.postgres import PostgresStore  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="POSTGRES_TEST_DSN is unset, so there is no database"),
]


@pytest.fixture
def schema() -> Iterator[str]:
    name = f"kira_test_{uuid.uuid4().hex}"
    yield name
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


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
            blocked = pool.submit(
                store.set_state, session.session_id, LifecycleState.HUMAN_VERIFIED
            )
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
