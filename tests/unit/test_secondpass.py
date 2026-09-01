"""The flow, its fail-safes, and what it costs.

The table in `test_the_fail_safe_topology` is the acceptance criterion for the
mechanism: every condition under which a second pass declines or a component fails
resolves to the first opinion, and there is no path on which a broken gate, nominator,
selector or store can promote the second run. It is driven end to end through the real
strategies against a fake diagnosis runner, a fake model and a fake store, so what it
pins is the composition rather than a mock of it.
"""

import asyncio
import json
import pathlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import RootCause
from kira.contracts.state import IncidentState
from kira.llm.types import ModelRequest, ModelResponse, Usage
from kira.memory.ports import SearchHit, SearchScope
from kira.secondpass import (
    MAR,
    MAS,
    RESAMPLE,
    BlindSelector,
    DiagnosisRun,
    JournalEmitter,
    QualityAuditGate,
    SecondPass,
    SecondPassContext,
    SecondPassPolicy,
    blinded_labels,
    build_evidence_bundle,
)
from kira.settings.profiles import load_profile

SAMPLING = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models")).sampling

ALERT = {
    "alertname": "HighErrorRate",
    "namespace": "shop",
    "service": "checkout",
    "_problem_id": "target_port_misconfig",
}


def diagnosis(
    summary: str = "the probe points at the wrong port", *, halt: bool = False
) -> RootCause:
    return RootCause(
        fault_type="misconfiguration",
        affected_resource="shop/Service/checkout",
        confidence=0.7,
        evidence=["readiness probe on :9090 while the container listens on :8080"],
        summary=summary,
        halt=halt,
        halt_reason="no evidence" if halt else None,
    )


def transcript(tool: str, body: str, call_id: str = "c1") -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


class FakeModel:
    """Answers by caller, and records the exact multiset of calls made."""

    def __init__(self, replies: Mapping[str, object]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, ModelRequest]] = []

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        self.calls.append((caller, request))
        reply = self._replies.get(caller, "{}")
        if isinstance(reply, Exception):
            raise reply
        if reply == "__slow__":
            await asyncio.sleep(5)
        return ModelResponse(
            text=str(reply),
            tool_calls=(),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )

    def callers(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for caller, _ in self.calls:
            counts[caller] = counts.get(caller, 0) + 1
        return counts


class FakeStore:
    """Returns scripted hits, and records the query and the exclusion it was given."""

    def __init__(self, hits: Sequence[SearchHit] | Exception = ()) -> None:
        self._hits = hits
        self.queries: list[tuple[str, tuple[str, ...]]] = []

    def search(
        self, query: str, *, scope: SearchScope = "evidence", exclude: Sequence[str] = ()
    ) -> list[SearchHit]:
        del scope
        self.queries.append((query, tuple(exclude)))
        if isinstance(self._hits, Exception):
            raise self._hits
        return list(self._hits)


@dataclass
class FakeRca:
    """Two scripted diagnosis runs, and a record of how each was entered."""

    artifacts: tuple[RootCause | None, RootCause | None] = (None, None)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(
        self,
        state: IncidentState,
        *,
        brief: str | None,
        retrieval: bool,
        run_label: str,
    ) -> DiagnosisRun:
        index = len(self.calls)
        self.calls.append(
            {"state": dict(state), "brief": brief, "retrieval": retrieval, "label": run_label}
        )
        return DiagnosisRun(
            artifact=self.artifacts[index],
            run_id=f"run-{index + 1}",
            messages=tuple(transcript("get_logs", f"log body {index}")),
        )


def hit(record_id: str, category: str = "probe_misconfiguration") -> SearchHit:
    return SearchHit(
        session_id=record_id,
        summary=f"{record_id} readiness probe pointed at a closed port",
        metadata={"fault_category": category},
        score=0.9,
    )


def context(
    model: FakeModel,
    *,
    store: FakeStore | None = None,
    policy_name: str = "ma-s",
    withheld: frozenset[str] = frozenset(),
    snapshot: str = "pod checkout-0 is not ready: readiness probe failed",
) -> tuple[SecondPassContext, ListSink]:
    sink = ListSink()

    async def snapshot_provider() -> str:
        return snapshot

    ctx = SecondPassContext(
        session_id="incident-0001",
        alert={k: v for k, v in ALERT.items() if not k.startswith("_")},
        state=cast(IncidentState, {"session_id": "incident-0001", "alert": dict(ALERT)}),
        completion=model,
        sampling=SAMPLING,
        memory=store,
        snapshot=snapshot_provider,
        withheld_record_ids=withheld,
        emit=JournalEmitter(
            Journal(sink, session_id="incident-0001", stage="secondpass"), policy=policy_name
        ),
    )
    return ctx, sink


FIRE = json.dumps({"needs_second_opinion": True, "weakness": "cause cannot reach the symptom"})
PASS = json.dumps({"needs_second_opinion": False, "weakness": "none"})
PICK_A = json.dumps(
    {"winner": "A", "confidence": 0.8, "reason": "better localized", "causal_check": "traced"}
)
DIFFERS = json.dumps(
    {"differs": True, "family": "message_queue", "directive": "check the head of the queue"}
)
SAME = json.dumps({"differs": False, "family": "probe_misconfiguration", "directive": ""})


# Row numbers are the fail-safe topology in the specification. Each row states the
# condition, and every one of them must leave the first opinion standing.
FAIL_SAFE_ROWS = [
    ("1 first run produced nothing", MAS, (None, None), {}, None, "first_run_failed", "none"),
    (
        "2 first run halted",
        MAS,
        (diagnosis(halt=True), None),
        {},
        None,
        "first_run_halted",
        "first",
    ),
    (
        "5 gate declines",
        MAS,
        (diagnosis(), None),
        {"secondpass.gate": PASS},
        FakeStore([hit("m1")]),
        "quality_gate_pass",
        "first",
    ),
    (
        "6 retrieval raised",
        MAS,
        (diagnosis(), None),
        {"secondpass.gate": FIRE},
        FakeStore(RuntimeError("store unreachable")),
        "search_failed",
        "first",
    ),
    (
        "9 nominator unparseable",
        MAR,
        (diagnosis(), None),
        {"secondpass.nominator": "no json here"},
        FakeStore([hit("m1")]),
        "nominator_unparseable",
        "first",
    ),
    (
        "10 nomination rejected by the vocabulary",
        MAR,
        (diagnosis(), None),
        {
            "secondpass.nominator": json.dumps(
                {"differs": True, "family": "fault_injection", "directive": "look again"}
            )
        },
        FakeStore([hit("m1")]),
        "nominator_rejected:family_not_in_vocabulary:fault_injection",
        "first",
    ),
    (
        "11 nomination agrees with the first opinion",
        MAR,
        (diagnosis(), None),
        {"secondpass.nominator": SAME},
        FakeStore([hit("m1")]),
        "same_family",
        "first",
    ),
]


@pytest.mark.parametrize(
    ("label", "policy", "artifacts", "replies", "store", "skipped", "source"),
    FAIL_SAFE_ROWS,
    ids=[row[0] for row in FAIL_SAFE_ROWS],
)
async def test_the_fail_safe_topology_returns_the_first_opinion(
    label: str,
    policy: SecondPassPolicy,
    artifacts: tuple[RootCause | None, RootCause | None],
    replies: dict[str, object],
    store: FakeStore | None,
    skipped: str,
    source: str,
) -> None:
    del label
    model = FakeModel(replies)
    ctx, sink = context(model, store=store, policy_name=policy.name)
    rca = FakeRca(artifacts)

    result = await SecondPass(policy, rca=rca).run(ctx)

    assert result.skipped == skipped
    assert result.source == source
    assert result.chosen is artifacts[0]
    assert result.second is None
    assert len(rca.calls) == 1
    # A skip that leaves no trace makes the mechanism's dose unmeasurable.
    assert any(event.payload.step == "skip" for event in sink.events)


@pytest.mark.parametrize(
    ("reply", "fallback", "winner"),
    [
        (
            json.dumps({"winner": "tie", "confidence": 0.2, "reason": "cannot tell"}),
            "selector_tie",
            "tie",
        ),
        ("no json at all", "selector_parse_failure", "fallback"),
        (RuntimeError("endpoint refused"), "selector_exception", "fallback"),
    ],
)
async def test_a_selector_that_cannot_decide_keeps_the_first_opinion(
    reply: object, fallback: str, winner: str
) -> None:
    first, second = diagnosis(), diagnosis("the queue head is poisoned")
    model = FakeModel({"secondpass.gate": FIRE, "secondpass.selector": reply})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))

    result = await SecondPass(MAS, rca=FakeRca((first, second))).run(ctx)

    assert result.chosen is first
    assert result.source == "first"
    assert result.selection is not None
    assert result.selection.fallback_reason == fallback
    assert result.selection.public_winner == winner


async def test_a_selector_that_exceeds_its_deadline_keeps_the_first_opinion() -> None:
    first, second = diagnosis(), diagnosis("the queue head is poisoned")
    model = FakeModel({"secondpass.gate": FIRE, "secondpass.selector": "__slow__"})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))
    policy = replace(MAS, selector=BlindSelector(max_tokens=1600, timeout_s=0))

    result = await SecondPass(policy, rca=FakeRca((first, second))).run(ctx)

    assert result.chosen is first
    assert result.selection is not None
    assert result.selection.fallback_reason == "selector_timeout"


async def test_a_second_run_that_fails_leaves_the_first_opinion_unopposed() -> None:
    first = diagnosis()
    model = FakeModel({"secondpass.gate": FIRE})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))

    result = await SecondPass(MAS, rca=FakeRca((first, None))).run(ctx)

    assert result.chosen is first
    assert result.source == "first"
    assert result.selection is not None
    assert result.selection.fallback_reason == "second_branch_failed"
    # Row 17 of the topology, the unopposed promotion of a second opinion, is
    # unreachable from here: rows 1 and 2 return before any second run exists.
    assert "secondpass.selector" not in model.callers()


@pytest.mark.parametrize("policy", [MAS, MAR, RESAMPLE], ids=lambda p: p.name)
async def test_both_diagnosis_runs_are_retrieval_free_and_the_second_is_blind(
    policy: SecondPassPolicy,
) -> None:
    first, second = diagnosis(), diagnosis("the queue head is poisoned")
    model = FakeModel(
        {"secondpass.gate": FIRE, "secondpass.nominator": DIFFERS, "secondpass.selector": PICK_A}
    )
    ctx, _ = context(model, store=FakeStore([hit("m1")]), policy_name=policy.name)
    rca = FakeRca((first, second))

    await SecondPass(policy, rca=rca).run(ctx)

    assert [call["retrieval"] for call in rca.calls] == [False, False]
    assert rca.calls[0]["brief"] is None
    # The second run is entered with the original incoming state, so the first run's
    # diagnosis, transcript and trace summaries are simply not in it.
    assert rca.calls[1]["state"] == dict(ctx.state)
    assert "root_cause" not in rca.calls[1]["state"]
    assert "messages" not in rca.calls[1]["state"]


async def test_a_gate_decline_costs_one_call_and_never_touches_the_store() -> None:
    store = FakeStore([hit("m1")])
    model = FakeModel({"secondpass.gate": PASS})
    ctx, _ = context(model, store=store)

    result = await SecondPass(MAS, rca=FakeRca((diagnosis(), None))).run(ctx)

    assert model.callers() == {"secondpass.gate": 1}
    assert store.queries == []
    assert result.verdict is not None and result.verdict.reason == "quality_gate_pass"


@pytest.mark.parametrize(
    ("policy", "replies", "expected"),
    [
        (MAS, {"secondpass.gate": FIRE}, {"secondpass.gate": 1, "secondpass.selector": 1}),
        (
            MAR,
            {"secondpass.nominator": DIFFERS},
            {"secondpass.nominator": 1, "secondpass.selector": 1},
        ),
        (RESAMPLE, {}, {"secondpass.selector": 1}),
    ],
    ids=["ma-s", "ma-r", "resample"],
)
async def test_every_helper_call_names_a_caller(
    policy: SecondPassPolicy, replies: dict[str, object], expected: dict[str, int]
) -> None:
    model = FakeModel({**replies, "secondpass.selector": PICK_A})
    ctx, _ = context(model, store=FakeStore([hit("m1")]), policy_name=policy.name)

    await SecondPass(policy, rca=FakeRca((diagnosis(), diagnosis("a different cause")))).run(ctx)

    assert model.callers() == expected


async def test_the_quality_audit_sees_only_the_candidate() -> None:
    model = FakeModel({"secondpass.gate": PASS})
    ctx, _ = context(model, store=FakeStore([hit("m1", "message_queue")]))

    await SecondPass(MAS, rca=FakeRca((diagnosis(), None))).run(ctx)

    prompt = str(model.calls[0][1].messages[0]["content"])
    assert "precedent" not in prompt
    assert "log body" not in prompt
    assert "HighErrorRate" not in prompt


@pytest.mark.parametrize(
    ("reply", "parsed", "fires"),
    [
        (json.dumps({"needs_second_opinion": "yes"}), False, True),
        (json.dumps({"needs_second_opinion": 1}), False, True),
        ("the diagnosis looks fine to me", False, True),
        (RuntimeError("endpoint refused"), False, True),
        (PASS, True, False),
        (FIRE, True, True),
    ],
)
async def test_an_unreadable_audit_never_suppresses_a_second_opinion(
    reply: object, parsed: bool, fires: bool
) -> None:
    model = FakeModel({"secondpass.gate": reply, "secondpass.selector": PICK_A})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))
    rca = FakeRca((diagnosis(), diagnosis("a different cause")))

    result = await SecondPass(MAS, rca=rca).run(ctx)

    assert result.verdict is not None
    assert result.verdict.parsed is parsed
    assert result.verdict.fires is fires
    assert len(rca.calls) == (2 if fires else 1)


async def test_a_quality_fire_lifts_the_nominators_veto() -> None:
    """The two gates are ORed, never ANDed: ANDing scored worse than either alone."""
    policy = replace(MAR, gate=QualityAuditGate(max_tokens=600, on_unreadable="defer"))
    model = FakeModel(
        {"secondpass.gate": FIRE, "secondpass.nominator": SAME, "secondpass.selector": PICK_A}
    )
    ctx, _ = context(model, store=FakeStore([hit("m1")]), policy_name=policy.name)
    rca = FakeRca((diagnosis(), diagnosis("a verified attribution")))

    result = await SecondPass(policy, rca=rca).run(ctx)

    assert result.skipped is None
    assert result.brief is not None
    assert result.brief.text is not None
    assert result.brief.text.startswith("## Second-opinion assignment\n")
    assert "Independently verify or falsify" in result.brief.text
    assert result.brief.text.endswith("\n") and not result.brief.text.endswith("\n\n")
    assert len(rca.calls) == 2


async def test_an_empty_result_set_under_a_firing_gate_runs_blind() -> None:
    model = FakeModel({"secondpass.gate": FIRE, "secondpass.selector": PICK_A})
    ctx, sink = context(model, store=FakeStore([]))
    rca = FakeRca((diagnosis(), diagnosis("a different cause")))

    result = await SecondPass(MAS, rca=rca).run(ctx)

    assert result.brief is not None and result.brief.text is None
    assert rca.calls[1]["brief"] is None
    assert any(
        event.payload.step == "brief" and event.payload.reason == "no_precedents_blind_run"
        for event in sink.events
    )


async def test_the_resampling_control_never_reads_the_store() -> None:
    store = FakeStore([hit("m1")])
    model = FakeModel({"secondpass.selector": PICK_A})
    ctx, _ = context(model, store=store, policy_name="resample")
    rca = FakeRca((diagnosis(), diagnosis("a different cause")))

    result = await SecondPass(RESAMPLE, rca=rca).run(ctx)

    assert store.queries == []
    assert result.precedents is None
    assert rca.calls[1]["brief"] is None
    assert model.callers() == {"secondpass.selector": 1}


async def test_a_skip_still_carries_the_verdict_and_the_precedents() -> None:
    model = FakeModel({"secondpass.nominator": SAME})
    ctx, _ = context(
        model, store=FakeStore([hit("m1")]), policy_name="ma-r", withheld=frozenset({"x"})
    )

    result = await SecondPass(MAR, rca=FakeRca((diagnosis(), None))).run(ctx)

    assert result.skipped == "same_family"
    assert result.verdict is not None and result.verdict.reason == "nominator"
    assert result.precedents is not None
    assert [p.record_id for p in result.precedents.precedents] == ["m1"]
    assert result.precedents.withheld_count == 1


async def test_both_run_ids_survive_the_selection() -> None:
    model = FakeModel({"secondpass.gate": FIRE, "secondpass.selector": PICK_A})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))

    result = await SecondPass(MAS, rca=FakeRca((diagnosis(), diagnosis("a rival")))).run(ctx)

    assert result.first.run_id == "run-1"
    assert result.second is not None and result.second.run_id == "run-2"


def test_the_label_shuffle_is_keyed_on_the_session_id() -> None:
    # sha256("incident-0001").digest()[0] == 190, even: the second opinion takes label A.
    assert blinded_labels("incident-0001") == {"A": "second", "B": "first"}
    # sha256("test-session").digest()[0] == 73, odd: the first opinion takes label A.
    assert blinded_labels("test-session") == {"A": "first", "B": "second"}


async def test_the_selector_prompt_never_names_a_source() -> None:
    model = FakeModel({"secondpass.gate": FIRE, "secondpass.selector": PICK_A})
    ctx, _ = context(model, store=FakeStore([hit("m1")]))

    result = await SecondPass(MAS, rca=FakeRca((diagnosis(), diagnosis("a rival")))).run(ctx)

    assert result.selection is not None
    prompt = result.selection.prompt.lower()
    # The template's own header mentions memory, to tell the selector that memory
    # content is excluded. What must never appear is which candidate came from where,
    # or any retrieved text.
    for marker in ("first opinion", "second opinion", "first_opinion", "second_opinion"):
        assert marker not in prompt
    assert "--- precedent" not in prompt
    assert "readiness probe pointed at a closed port" not in prompt


def test_the_evidence_bundle_excludes_retrieval_and_the_submission() -> None:
    bundle = build_evidence_bundle(
        ALERT,
        [
            transcript("memory_search", "a past incident about a poisoned queue"),
            transcript("get_logs", "connection refused on :8080")
            + transcript("submit_root_cause", "accepted", call_id="c2"),
        ],
    )

    assert "poisoned queue" not in bundle
    assert "accepted" not in bundle
    assert "[get_logs] connection refused on :8080" in bundle
    # The benchmark's own fault label never reaches a prompt.
    assert "target_port_misconfig" not in bundle


def test_the_evidence_bundle_keeps_the_last_twelve_bounded_results() -> None:
    long_body = "x" * 4000
    messages: list[dict[str, Any]] = []
    for index in range(20):
        messages += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"c{index}", "type": "function", "function": {"name": "get_logs"}}
                ],
            },
            {"role": "tool", "tool_call_id": f"c{index}", "content": f"{index} {long_body}"},
        ]

    bundle = build_evidence_bundle({}, [messages])

    kept = [line for line in bundle.splitlines() if line.startswith("[get_logs]")]
    assert len(kept) == 12
    assert kept[0].startswith("[get_logs] 8 ")
    assert all(len(line) <= len("[get_logs] ") + 1800 for line in kept)
