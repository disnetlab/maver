"""The contract every memory backend has to satisfy, whatever it stores its data in.

A backend is done when it passes this unmodified. Subclass :class:`MemoryStoreContract`
in a ``test_*.py`` module and supply the two fixtures:

    class TestMyBackend(MemoryStoreContract):
        @pytest.fixture
        def store(self): ...       # SessionStore + ApprovalStore + RunStore
        @pytest.fixture
        def records(self, store): ...   # TextSearch, over the same database

Two objects because the operational ports and the corpus are separate things that a
deployment can have one of without the other. They must share a database: several tests
here read a session's live lifecycle state while holding the record written beside it.

What is deliberately not asserted: the order of a candidate pool. The port says it
carries no meaning, and a test that pinned one would make every backend agree on a
ranking that only the retriever is allowed to have.
"""

from typing import Any, Protocol

import pytest

from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.memory.lifecycle import IllegalTransition
from kira.memory.ports import (
    AgentRunEventDraft,
    AgentRunRecord,
    ApprovalRecord,
    ApprovalStore,
    RunStore,
    SessionMemory,
    SessionStore,
    TextSearch,
)
from kira.memory.ranking import Bm25Ranker
from kira.memory.retrieval import Retriever

__all__ = [
    "MemoryStoreContract",
    "OperationalStore",
    "a_record",
    "a_request",
    "a_run",
    "an_event",
]

NUL = "\x00"


class OperationalStore(SessionStore, ApprovalStore, RunStore, Protocol):
    """The three ports a run touches while it is running."""


def a_record(
    session_id: str, *, evidence: str = "", lesson: str = "", **kwargs: Any
) -> SessionMemory:
    fields: dict[str, Any] = {
        "incident_summary": "the checkout deployment stopped serving",
        "lifecycle_state": LifecycleState.PENDING.value,
        "fault_category": "resource_exhaustion",
        "alert_name": "PodCrashLooping",
        "namespace": "prod",
        "affected_resource": "deployment/checkout",
    }
    fields.update(kwargs)
    return SessionMemory(
        session_id=session_id, evidence_text=evidence, lesson_text=lesson, **fields
    )


def a_run(agent_run_id: str, session_id: str, **kwargs: Any) -> AgentRunRecord:
    fields: dict[str, Any] = {
        "agent_name": "rca",
        "attempt": 1,
        "status": "running",
        "started_at": "2026-01-01T00:00:00+00:00",
        "metadata": {"arm": "ma-r"},
    }
    fields.update(kwargs)
    return AgentRunRecord(agent_run_id=agent_run_id, pipeline_id=session_id, **fields)


def an_event(
    event_id: str, agent_run_id: str, session_id: str, **kwargs: Any
) -> AgentRunEventDraft:
    fields: dict[str, Any] = {
        "event_type": "tool_call",
        "payload": {"tool": "kubectl"},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    fields.update(kwargs)
    return AgentRunEventDraft(
        event_id=event_id, agent_run_id=agent_run_id, pipeline_id=session_id, **fields
    )


def a_request(request_id: str, session_id: str, **kwargs: Any) -> ApprovalRecord:
    fields: dict[str, Any] = {
        "kind": "tool_call",
        "stage": "execution",
        "status": "pending",
        "summary": "delete pod checkout-0",
        "payload": {"tool": "kubectl", "args": ["delete", "pod", "checkout-0"]},
    }
    fields.update(kwargs)
    return ApprovalRecord(request_id=request_id, session_id=session_id, **fields)


class MemoryStoreContract:
    """Port semantics that a second backend could plausibly get wrong."""

    def test_the_backend_satisfies_every_port(
        self, store: OperationalStore, records: TextSearch
    ) -> None:
        assert isinstance(store, SessionStore)
        assert isinstance(store, ApprovalStore)
        assert isinstance(store, RunStore)
        assert isinstance(records, TextSearch)

    # SessionStore ---------------------------------------------------------------

    def test_a_session_round_trips(self, store: OperationalStore) -> None:
        alert = {"name": "PodCrashLooping", "labels": {"ns": "prod", "sévérité": "critique"}}
        created = store.create_session(alert)

        assert created.session_id
        assert created.started_at
        assert created.state == LifecycleState.PENDING.value
        assert created.run_status == RunStatus.PENDING.value
        assert created.distilled is False
        assert created.ended_at is None

        read = store.get_session(created.session_id)
        assert read is not None
        assert read.session_id == created.session_id
        assert read.alert == alert
        assert read.root_cause is None and read.plan is None and read.execution_result is None
        assert store.get_session("no-such-session") is None

        store.mark_distilled(created.session_id)
        distilled = store.get_session(created.session_id)
        assert distilled is not None
        assert distilled.distilled is True

    def test_update_artefacts_is_partial(self, store: OperationalStore) -> None:
        """Planning attaching a plan must not erase the diagnosis it was given."""
        session = store.create_session({})
        store.update_artefacts(session.session_id, root_cause={"cause": "memory limit too low"})
        store.update_artefacts(session.session_id, plan={"steps": ["raise the limit"]})
        store.update_artefacts(session.session_id, verify_result={"verdict": "recovered"})
        store.update_artefacts(session.session_id, notes="operator paged")

        read = store.get_session(session.session_id)
        assert read is not None
        assert read.root_cause == {"cause": "memory limit too low"}
        assert read.plan == {"steps": ["raise the limit"]}
        assert read.verify_result == {"verdict": "recovered"}
        assert read.notes == "operator paged"
        assert read.execution_result is None

    def test_get_states_reads_the_state_now_not_the_one_stored_beside_a_record(
        self, store: OperationalStore, records: TextSearch
    ) -> None:
        """The state copied into a record goes stale the moment the run is re-graded."""
        session = store.create_session({})
        records.upsert_summary(
            a_record(
                session.session_id, evidence="oom", lifecycle_state=LifecycleState.PENDING.value
            )
        )
        store.set_state(session.session_id, LifecycleState.INVALIDATED)

        stored = records.open_record(session.session_id)
        assert stored is not None
        assert stored.lifecycle_state == LifecycleState.PENDING.value
        other = store.create_session({})

        # Read in one call, current state for each, and nothing for an id that is gone.
        assert store.get_states([session.session_id, other.session_id, "gone"]) == {
            session.session_id: LifecycleState.INVALIDATED.value,
            other.session_id: LifecycleState.PENDING.value,
        }
        assert store.get_states([]) == {}

    def test_a_verdict_the_lifecycle_graph_forbids_is_refused_and_not_written(
        self, store: OperationalStore
    ) -> None:
        """A record's standing is a research claim, so no backend may write whatever
        standing a caller passed it."""
        session = store.create_session({})

        # Nothing is distilled from a session still running, so there is nothing yet for
        # an operator to have verified.
        with pytest.raises(IllegalTransition):
            store.set_state(session.session_id, LifecycleState.HUMAN_VERIFIED)
        assert store.get_states([session.session_id]) == {
            session.session_id: LifecycleState.PENDING.value
        }

        store.set_state(session.session_id, LifecycleState.TENTATIVE_SUCCESS)
        # Writing the state a record already holds is a no-op rather than a refusal, so
        # a job retried after a crash is safe to run again.
        store.set_state(session.session_id, LifecycleState.TENTATIVE_SUCCESS)
        with pytest.raises(IllegalTransition):
            store.set_state(session.session_id, LifecycleState.HALTED)
        assert store.get_states([session.session_id]) == {
            session.session_id: LifecycleState.TENTATIVE_SUCCESS.value
        }

        # A session that is not there is not an error, as it is not anywhere else here.
        store.set_state("no-such-session", LifecycleState.INVALIDATED)

    def test_the_run_status_and_its_checkpoint_move_together(self, store: OperationalStore) -> None:
        """No reader may see `awaiting_human` without the checkpoint that goes with it,
        and none may see a resumed run still advertising the checkpoint it consumed."""
        session = store.create_session({})
        store.set_run_status(
            session.session_id,
            RunStatus.AWAITING_HUMAN,
            current_stage="execution",
            checkpoint={"messages": 12},
            awaiting_request_id="req-1",
        )
        parked = store.get_session(session.session_id)
        assert parked is not None
        assert parked.run_status == RunStatus.AWAITING_HUMAN.value
        assert parked.current_stage == "execution"
        assert parked.checkpoint == {"messages": 12}
        assert parked.awaiting_request_id == "req-1"

        store.set_run_status(session.session_id, RunStatus.RUNNING, current_stage="execution")
        resumed = store.get_session(session.session_id)
        assert resumed is not None
        assert resumed.checkpoint is None
        assert resumed.awaiting_request_id is None

    def test_a_session_s_end_is_stamped_by_the_first_terminal_status_and_never_moves(
        self, store: OperationalStore
    ) -> None:
        """``ended_at`` is what a reader takes for when this incident stopped, and the
        incident route reports it. A non-terminal status claiming it would report every
        parked run as over; a later terminal status moving it would date the incident
        from whenever someone last cancelled or re-graded it."""
        session = store.create_session({})

        store.set_run_status(session.session_id, RunStatus.AWAITING_HUMAN, current_stage="rca")
        parked = store.get_session(session.session_id)
        assert parked is not None and parked.ended_at is None

        store.set_run_status(session.session_id, RunStatus.COMPLETED)
        ended = store.get_session(session.session_id)
        assert ended is not None and ended.ended_at

        store.set_run_status(session.session_id, RunStatus.CANCELLED)
        again = store.get_session(session.session_id)
        assert again is not None
        assert again.run_status == RunStatus.CANCELLED.value
        assert again.ended_at == ended.ended_at

    # ApprovalStore --------------------------------------------------------------

    def test_an_approval_round_trips(self, store: OperationalStore) -> None:
        session = store.create_session({})
        created = store.create_request(
            a_request(
                "req-1",
                session.session_id,
                tool_call_id="call-1",
                tool_name="kubectl",
                prompt="Approve?",
                expires_at="2026-01-01T00:00:00+00:00",
            )
        )
        assert created.created_at, "a stored request needs a creation time to be listed by one"

        read = store.get_request("req-1")
        assert read is not None
        assert read.status == "pending"
        assert read.payload == {"tool": "kubectl", "args": ["delete", "pod", "checkout-0"]}
        assert read.tool_call_id == "call-1"
        assert read.tool_name == "kubectl"
        assert read.prompt == "Approve?"
        assert read.expires_at == "2026-01-01T00:00:00+00:00"
        assert read.decided_at is None
        assert store.get_request("no-such-request") is None

    def test_a_second_decision_returns_none_and_changes_nothing(
        self, store: OperationalStore
    ) -> None:
        """This is what makes a late decision and a duplicate one harmless rather than
        a second dispatch of the same cluster write."""
        session = store.create_session({})
        store.create_request(a_request("req-1", session.session_id))

        first = store.decide("req-1", status="approved", decided_by="alice", reason="looks right")
        assert first is not None
        assert first.status == "approved"
        assert first.decided_by == "alice"
        assert first.reason == "looks right"
        assert first.decided_at

        assert store.decide("req-1", status="denied", decided_by="bob") is None
        unchanged = store.get_request("req-1")
        assert unchanged is not None
        assert unchanged.status == "approved"
        assert unchanged.decided_by == "alice"
        assert store.decide("no-such-request", status="approved", decided_by="alice") is None

    def test_expire_due_times_out_what_is_past_and_returns_exactly_that(
        self, store: OperationalStore
    ) -> None:
        session = store.create_session({})
        store.create_request(
            a_request("past", session.session_id, expires_at="2020-01-01T00:00:00+00:00")
        )
        store.create_request(
            a_request("future", session.session_id, expires_at="2099-01-01T00:00:00+00:00")
        )
        store.create_request(a_request("never", session.session_id))
        store.create_request(
            a_request(
                "decided",
                session.session_id,
                status="approved",
                expires_at="2020-01-01T00:00:00+00:00",
            )
        )

        expired = store.expire_due(now="2026-01-01T00:00:00+00:00")
        assert [record.request_id for record in expired] == ["past"]
        assert expired[0].status == "timed_out"
        assert expired[0].decided_at

        assert store.expire_due(now="2026-01-01T00:00:00+00:00") == []
        # A decision arriving after the timeout is as late as a duplicate one.
        assert store.decide("past", status="approved", decided_by="alice") is None
        for request_id, status in (
            ("future", "pending"),
            ("never", "pending"),
            ("decided", "approved"),
        ):
            read = store.get_request(request_id)
            assert read is not None
            assert read.status == status

    def test_requests_list_newest_first_and_filter_by_status(self, store: OperationalStore) -> None:
        session = store.create_session({})
        for index in range(3):
            store.create_request(
                a_request(
                    f"req-{index}",
                    session.session_id,
                    created_at=f"2026-01-0{index + 1}T00:00:00+00:00",
                )
            )
        store.decide("req-1", status="approved", decided_by="alice")

        assert [record.request_id for record in store.list_requests()] == [
            "req-2",
            "req-1",
            "req-0",
        ]
        assert [record.request_id for record in store.list_requests(status="pending")] == [
            "req-2",
            "req-0",
        ]
        assert [record.request_id for record in store.list_requests(status="approved")] == ["req-1"]

    # RunStore -------------------------------------------------------------------

    def test_the_store_allocates_the_event_sequence(self, store: OperationalStore) -> None:
        """The caller cannot choose it, so two writers on one run cannot collide.

        Monotonic from 1, per run, and independent between runs.
        """
        session = store.create_session({})
        for run_id in ("run-1", "run-2"):
            store.start_run(a_run(run_id, session.session_id))

        first = store.append_event(an_event("event-1", "run-1", session.session_id))
        second = store.append_event(an_event("event-2", "run-1", session.session_id))
        other = store.append_event(an_event("event-3", "run-2", session.session_id))

        assert (first.seq, second.seq, other.seq) == (1, 2, 1)
        assert [event.seq for event in store.events_for("run-1")] == [1, 2]
        assert store.events_for("run-1")[0] == first

    def test_events_come_back_ordered_by_seq(self, store: OperationalStore) -> None:
        session = store.create_session({})
        store.start_run(a_run("run-1", session.session_id))
        for index in range(3):
            store.append_event(
                an_event(
                    f"event-{index}",
                    "run-1",
                    session.session_id,
                    payload={"tool": "kubectl", "index": index},
                    role="rca",
                    tool_call_id=f"call-{index}",
                    tool_name="kubectl",
                )
            )

        events = store.events_for("run-1")
        assert [event.seq for event in events] == [1, 2, 3]
        assert [event.event_id for event in events] == ["event-0", "event-1", "event-2"]
        assert events[0].payload == {"tool": "kubectl", "index": 0}
        assert events[0].role == "rca"
        assert events[0].tool_call_id == "call-0"
        assert events[0].tool_name == "kubectl"
        assert events[0].pipeline_id == session.session_id
        assert store.events_for("no-such-run") == []

    def test_runs_are_readable_back_and_ordered_by_start(self, store: OperationalStore) -> None:
        """What a role produced is otherwise write-only."""
        session = store.create_session({})
        store.start_run(
            a_run(
                "run-2",
                session.session_id,
                agent_name="planning",
                started_at="2026-01-01T00:01:00+00:00",
            )
        )
        store.start_run(
            a_run(
                "run-1",
                session.session_id,
                agent_name="rca",
                started_at="2026-01-01T00:00:00+00:00",
            )
        )
        store.start_run(a_run("elsewhere", "another-incident"))
        store.finish_run(
            "run-1",
            status="succeeded",
            output={"cause": "oom"},
            trace_summary={"tool_calls": 4},
        )

        assert [run.agent_run_id for run in store.runs_for(session.session_id)] == [
            "run-1",
            "run-2",
        ]
        assert store.runs_for("no-such-incident") == []

        finished = store.get_run("run-1")
        assert finished is not None
        assert finished.status == "succeeded"
        assert finished.output == {"cause": "oom"}
        assert finished.trace_summary == {"tool_calls": 4}
        assert finished.error is None
        assert finished.ended_at
        assert finished.metadata == {"arm": "ma-r"}
        assert finished.attempt == 1
        assert store.get_run("no-such-run") is None

    def test_finishing_a_run_merges_its_metadata_into_what_starting_it_wrote(
        self, store: OperationalStore
    ) -> None:
        """The edge a run was reached by is written before the role exists to describe
        itself, and has to survive the role doing so."""
        session = store.create_session({})
        store.start_run(a_run("run-1", session.session_id, metadata={"after": "rca"}))

        store.finish_run(
            "run-1",
            status="succeeded",
            metadata={"health_gate": {"ran": True, "fell_open": False}},
        )
        finished = store.get_run("run-1")
        assert finished is not None
        assert finished.metadata == {
            "after": "rca",
            "health_gate": {"ran": True, "fell_open": False},
        }

        # Key by key, never into the objects: a nested value is replaced whole. This is
        # the one point where each backend's natural merge would have said otherwise.
        store.finish_run("run-1", status="succeeded", metadata={"health_gate": {"ran": False}})
        replaced = store.get_run("run-1")
        assert replaced is not None
        assert replaced.metadata == {"after": "rca", "health_gate": {"ran": False}}

        # Alone among these arguments, an omitted metadata leaves what is there.
        store.finish_run("run-1", status="failed", error="boom")
        kept = store.get_run("run-1")
        assert kept is not None
        assert kept.status == "failed", "a run records how it ended, not that it ended"
        assert kept.error == "boom"
        assert kept.metadata == {"after": "rca", "health_gate": {"ran": False}}
        assert kept.output is None

    # TextSearch -----------------------------------------------------------------

    def test_a_record_round_trips_including_what_retrieval_withholds(
        self, records: TextSearch
    ) -> None:
        memory = a_record(
            "s1",
            evidence="checkout pods restarting, OOMKilled in the last hour",
            lesson="raising the memory limit stopped the restarts",
            corrected_cause="the limit was set from a stale load estimate",
            verification_verdict="recovered",
            evaluation_label="correct",
            evaluation_source="oracle",
            evaluation_diagnosis_success=True,
            evaluation_diagnosis_score=0.9,
            evaluation_mitigation_success=False,
        )
        records.upsert_summary(memory)

        assert records.count() == 1
        assert records.open_record("s1") == memory
        assert records.open_record("no-such-record") is None

    def test_upsert_replaces_and_delete_removes(self, records: TextSearch) -> None:
        records.upsert_summary(a_record("s1", evidence="first"))
        records.upsert_summary(a_record("s1", evidence="second"))

        assert records.count() == 1
        stored = records.open_record("s1")
        assert stored is not None
        assert stored.evidence_text == "second"
        assert [m.session_id for m in records.candidates("first")] == []

        records.delete_summary("s1")
        assert records.count() == 0
        assert records.open_record("s1") is None
        records.delete_summary("s1")

    def test_candidates_are_disjunctive(self, records: TextSearch) -> None:
        """Requiring every term empties the pool on most real queries."""
        records.upsert_summary(a_record("s1", evidence="checkout pod oomkilled"))
        records.upsert_summary(a_record("s2", evidence="coredns resolution failure"))

        pool = {m.session_id for m in records.candidates("checkout coredns")}
        assert pool == {"s1", "s2"}
        assert records.candidates("") == []
        assert records.candidates("nothingmatchesthisterm") == []

    def test_candidates_are_the_whole_matching_pool_not_a_ranked_top_k(
        self, records: TextSearch
    ) -> None:
        """A backend that ranked would hand back its best few. The retriever's job is
        to choose, so anything that could plausibly match has to survive to it.

        The limit binds, because that is the only condition under which a backend's own
        ranking can show. The last three records are the ones every lexical ranker
        prefers — short, and the term repeated — so a backend that ordered by relevance
        would return those and an unranked one returns the three written first.
        """
        for index in range(12):
            records.upsert_summary(
                a_record(f"s{index}", evidence=f"checkout replica {index} " + "filler " * 40)
            )
        records.upsert_summary(a_record("unrelated", evidence="disk pressure on a node"))

        pool = {m.session_id for m in records.candidates("checkout")}
        assert pool == {f"s{index}" for index in range(12)}

        for index in range(12, 15):
            records.upsert_summary(a_record(f"s{index}", evidence="checkout checkout checkout"))
        bounded = {m.session_id for m in records.candidates("checkout", limit=3)}
        assert bounded == {"s0", "s1", "s2"}

    def test_candidates_respect_limit_scope_and_exclusion(self, records: TextSearch) -> None:
        for index in range(5):
            records.upsert_summary(
                a_record(f"s{index}", evidence="checkout restarting", lesson="raise the limit")
            )

        assert len(records.candidates("checkout", limit=2)) == 2
        assert records.candidates("checkout", scope="lesson") == []
        assert len(records.candidates("raise", scope="lesson")) == 5
        withheld = records.candidates("checkout", exclude=["s0", "s1"])
        assert {m.session_id for m in withheld} == {"s2", "s3", "s4"}

    def test_the_ranker_over_the_pool_finds_the_right_precedent(self, records: TextSearch) -> None:
        """The pool is only useful if the measured ranker can still pick out of it.

        All three records share a term, so all three reach the ranker and the one the
        query actually describes has to come out on top. The relevance floor is off:
        it is calibrated against a corpus of hundreds, not the three here.
        """
        records.upsert_summary(
            a_record(
                "oom",
                evidence="checkout service deployment crashloopbackoff oomkilled memory limit",
            )
        )
        records.upsert_summary(
            a_record("latency", evidence="payment service p99 latency spike upstream timeout")
        )
        records.upsert_summary(
            a_record("dns", evidence="cart service name resolution failure coredns")
        )

        query = "checkout service crashloopbackoff oomkilled"
        assert len(records.candidates(query)) == 3, "every record shares a term with the query"

        hits = Retriever(text=records, ranker=Bm25Ranker(min_score_per_token=0.0)).search(query)
        assert hits[0].session_id == "oom"
        assert hits[0].score is not None

    # Every write path -----------------------------------------------------------

    def test_a_nul_byte_is_stripped_on_every_write_path(
        self, store: OperationalStore, records: TextSearch
    ) -> None:
        """One byte of kubectl-exec output must not be able to fail a persistence call.

        In the prototype it did, and the episode that hit it skipped every submission
        and produced a result that read as an agent which never answered.
        """
        session = store.create_session({"detail": f"tool{NUL}output"})
        store.update_artefacts(
            session.session_id,
            root_cause={"cause": f"o{NUL}om"},
            execution_result={"stdout": [f"line{NUL}one"]},
            notes=f"note{NUL}here",
        )
        store.set_run_status(
            session.session_id, RunStatus.AWAITING_HUMAN, checkpoint={"buffer": f"a{NUL}b"}
        )
        read = store.get_session(session.session_id)
        assert read is not None
        assert read.alert == {"detail": "tooloutput"}
        assert read.root_cause == {"cause": "oom"}
        assert read.execution_result == {"stdout": ["lineone"]}
        assert read.notes == "notehere"
        assert read.checkpoint == {"buffer": "ab"}

        # The model names the tool and mints the tool-call id, so every column carrying
        # one is as exposed as the payload. A raise here comes back out of the approval
        # gate as an unhandled error rather than as the pause the caller expects.
        store.create_request(
            a_request(
                "req-1",
                session.session_id,
                kind=f"tool{NUL}call",
                stage=f"exec{NUL}ution",
                summary=f"delete{NUL}pod",
                payload={"args": [f"kubectl{NUL}delete"]},
                tool_call_id=f"call{NUL}1",
                tool_name=f"kube{NUL}ctl",
                decided_by=f"op{NUL}erator",
            )
        )
        request = store.get_request("req-1")
        assert request is not None
        assert request.kind == "toolcall"
        assert request.stage == "execution"
        assert request.summary == "deletepod"
        assert request.payload == {"args": ["kubectldelete"]}
        assert request.tool_call_id == "call1"
        assert request.tool_name == "kubectl"
        assert request.decided_by == "operator"

        store.start_run(
            a_run(
                "run-1",
                session.session_id,
                agent_name=f"exec{NUL}ution",
                metadata={"note": f"m{NUL}eta"},
            )
        )
        appended = store.append_event(
            an_event(
                "event-1",
                "run-1",
                session.session_id,
                event_type=f"tool{NUL}result",
                payload={"result": f"binary{NUL}output"},
                role=f"exec{NUL}ution",
                tool_call_id=f"call{NUL}1",
                tool_name=f"kube{NUL}ctl",
            )
        )
        store.finish_run("run-1", status="failed", error=f"boom{NUL}", output={"x": f"y{NUL}z"})
        assert appended.payload == {"result": "binaryoutput"}
        assert appended.event_type == "toolresult"
        assert appended.role == "execution"
        assert appended.tool_call_id == "call1"
        assert appended.tool_name == "kubectl"
        stored_event = store.events_for("run-1")[0]
        assert stored_event.payload == {"result": "binaryoutput"}
        assert stored_event.event_type == "toolresult"
        assert stored_event.role == "execution"
        assert stored_event.tool_call_id == "call1"
        assert stored_event.tool_name == "kubectl"
        run = store.get_run("run-1")
        assert run is not None
        assert run.agent_name == "execution"
        assert run.metadata == {"note": "meta"}
        assert run.error == "boom"
        assert run.output == {"x": "yz"}

        records.upsert_summary(
            a_record("s1", evidence=f"oom{NUL}killed", lesson=f"raise{NUL}the limit")
        )
        stored = records.open_record("s1")
        assert stored is not None
        assert stored.evidence_text == "oomkilled"
        assert stored.lesson_text == "raisethe limit"
        assert [m.session_id for m in records.candidates("oomkilled")] == ["s1"]
