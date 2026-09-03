"""The pieces a second pass is composed from, and the bytes they put in a prompt.

The prompt digests are the load-bearing check here. Every published number was produced
against this exact wording, so a reformatting is a behaviour change; the digests catch
it for a reader who does not have the prototype checked out beside this repository,
which `test_prompt_assets.py` needs and this does not.
"""

import hashlib
import pathlib
from collections.abc import Sequence
from typing import Any, cast

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import RootCause
from kira.contracts.state import IncidentState
from kira.memory.ports import SearchHit, SearchScope
from kira.secondpass import (
    ALLOWED_FAMILIES,
    NOMINATED_ASSIGNMENT,
    QUALITY_GATED_PRECEDENT,
    UNCONDITIONAL_RESAMPLE,
    BlindSelector,
    FamilyAssignmentBrief,
    GateVerdict,
    JournalEmitter,
    MergedQuery,
    Nomination,
    ObservationQuery,
    PreAnchorQuery,
    Precedent,
    PrecedentSet,
    PrecedentTextBrief,
    QualityAuditGate,
    SecondPassConfigError,
    SecondPassContext,
    StorePrecedents,
    policy_for,
    render_precedents,
    validate_nomination,
)
from kira.secondpass.strategies import _COMPOSITIONS
from kira.settings.groups import MemorySettings, SecondPassSettings
from kira.settings.profiles import load_profile

PROMPTS = pathlib.Path(__file__).resolve().parents[2] / "src" / "kira" / "prompts" / "secondpass"
SAMPLING = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models")).sampling

# SHA-256 of each asset as it was carried over from the prototype on branch
# feat/simplify at 30070ee. These are measured artifacts, not editable text.
PROMPT_DIGESTS = {
    "assignment.txt": "87a3f79ea9796ba75924b8643e03eb00fcbcecf261c726f28215b12c6b422601",
    "context.txt": "5cf08568a06ef3625ae8ccd6d0d7eb5ba7443d9960e33586e1d340727fd62609",
    "incident_context.txt": "9a88bd1c94554b1636453a7c4632f1ddedbe00b15ee93671ec4e4663407e8843",
    "nominator_system.txt": "24e26cf36fc7cc92bca9342503793a14969ef67ac93e480426a7e90a7415e477",
    "nominator_user.txt": "c67ce02522852da89aa1260965d32378f8e2acbe61a47d6ecaae30657a862daa",
    "quality_gate_system.txt": "1699d7583ac7413d9f994ed477df69eac7c2792ccf811ca7f270c7a2e000a364",
    "quality_gate_user.txt": "41e5b9004d2b357870ece467f52713d815a7afa1719c56fca168794f72855b12",
    "selector_system.txt": "5bacdbed71868d7b6fe331064c933a613a4dd18ea875d52bca28cb3272ab5d28",
    "selector_user.txt": "7dd89124d81e5cebf9aa447aa1b2c33122a3adacb38481232fbc0178d52ce677",
}


@pytest.mark.parametrize(("name", "digest"), sorted(PROMPT_DIGESTS.items()))
def test_prompt_digests(name: str, digest: str) -> None:
    assert hashlib.sha256((PROMPTS / name).read_bytes()).hexdigest() == digest


def test_every_shipped_secondpass_prompt_is_pinned() -> None:
    assert {p.name for p in PROMPTS.glob("*.txt")} == set(PROMPT_DIGESTS)


def diagnosis() -> RootCause:
    return RootCause(
        fault_type="misconfiguration",
        affected_resource="shop/Service/checkout",
        confidence=0.7,
        evidence=["readiness probe on :9090 while the container listens on :8080", "x" * 400],
        summary="the probe points at the wrong port",
    )


class FakeStore:
    def __init__(self, per_call: Sequence[Sequence[SearchHit]]) -> None:
        self._per_call = list(per_call)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def search(
        self, query: str, *, scope: SearchScope = "evidence", exclude: Sequence[str] = ()
    ) -> list[SearchHit]:
        del scope
        self.calls.append((query, tuple(exclude)))
        return list(self._per_call[len(self.calls) - 1])


def hit(record_id: str) -> SearchHit:
    return SearchHit(
        session_id=record_id,
        summary=f"{record_id} summary",
        metadata={"fault_category": "message_queue"},
        score=0.5,
    )


def context(
    *, store: Any = None, snapshot: str | Exception = "", withheld: frozenset[str] = frozenset()
) -> SecondPassContext:
    async def snapshot_provider() -> str:
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot

    return SecondPassContext(
        session_id="incident-0001",
        alert={"alertname": "HighErrorRate", "namespace": "shop", "service": "checkout"},
        state=cast(IncidentState, {"session_id": "incident-0001"}),
        completion=cast(Any, None),
        sampling=SAMPLING,
        memory=store,
        snapshot=snapshot_provider,
        withheld_record_ids=withheld,
        emit=JournalEmitter(
            Journal(ListSink(), session_id="incident-0001", stage="secondpass"), policy="test"
        ),
    )


async def test_the_observation_query_excludes_the_conclusion_and_is_bounded() -> None:
    candidate = diagnosis()
    (query,) = await ObservationQuery().build(context(), candidate)

    for conclusion in (candidate.fault_type, candidate.affected_resource, candidate.summary):
        assert conclusion not in query.text
    assert "alertname=HighErrorRate" in query.text
    assert len(query.text) <= 1200
    assert all(len(line) <= 160 for line in query.text.splitlines())


async def test_the_pre_anchor_query_survives_a_snapshot_failure() -> None:
    ctx = context(snapshot=RuntimeError("the cluster is unreachable"))
    (query,) = await PreAnchorQuery().build(ctx, diagnosis())

    assert query.text.splitlines() == [
        "Current incident observations:",
        "Identifiers: alertname=HighErrorRate; namespace=shop; service=checkout",
    ]


async def test_the_pre_anchor_query_keeps_the_campaign_budget() -> None:
    """Every published cell was retrieved unfiltered, under a 3,600-character cap.

    Both screened mitigations shipped off in the campaign, and either one changes what
    reaches the ranker for every arm that carries a number.
    """
    lines = ["RECENT CHANGES (Warning events, last 30m)"] + [
        f"pod checkout-{i} restarting: CrashLoopBackOff " + "x" * 320 for i in range(8)
    ]
    (query,) = await PreAnchorQuery().build(context(snapshot="\n".join(lines)), diagnosis())

    assert "RECENT CHANGES (Warning events" in query.text
    assert 2400 < len(query.text) <= 3600


async def test_merged_queries_are_deduplicated_evidence_first() -> None:
    ctx = context(
        store=FakeStore([[hit("a"), hit("b")], [hit("b"), hit("c")]]),
        snapshot="pod checkout-0 restarting: CrashLoopBackOff",
    )
    queries = await MergedQuery(ObservationQuery(), PreAnchorQuery()).build(ctx, diagnosis())
    result = await StorePrecedents(k=2, cap=6).fetch(ctx, queries)

    assert [q.name for q in queries] == ["evidence", "pre_anchor"]
    assert [p.record_id for p in result.precedents] == ["a", "b", "c"]
    assert result.per_query_ids == {"evidence": ("a", "b"), "pre_anchor": ("b", "c")}


async def test_the_withheld_set_reaches_the_store_and_is_counted() -> None:
    store = FakeStore([[hit("a")]])
    ctx = context(store=store, withheld=frozenset({"own-record"}))

    result = await StorePrecedents(k=3, cap=6).fetch(
        ctx, await ObservationQuery().build(ctx, diagnosis())
    )

    assert store.calls[0][1] == ("own-record",)
    assert result.withheld_count == 1


async def test_the_union_is_capped() -> None:
    store = FakeStore([[hit(str(i)) for i in range(5)]])
    ctx = context(store=store)

    result = await StorePrecedents(k=5, cap=2).fetch(
        ctx, await ObservationQuery().build(ctx, diagnosis())
    )

    assert [p.record_id for p in result.precedents] == ["0", "1"]


@pytest.mark.parametrize(
    ("score", "rendered"), [(0.5, "score=0.500"), (None, "score=n/a")], ids=["scored", "null"]
)
def test_the_precedent_card_format(score: float | None, rendered: str) -> None:
    card = render_precedents(
        [Precedent(record_id="m1", fault_category="message_queue", summary="a  b\nc", score=score)]
    )

    assert card == f"--- precedent (fault_category=message_queue, {rendered}) ---\na b c"


async def test_the_briefs_render_the_measured_blocks() -> None:
    precedents = PrecedentSet(
        precedents=(
            Precedent(
                record_id="m1", fault_category="message_queue", summary="a poisoned head", score=0.5
            ),
        ),
        queries=(),
        per_query_ids={},
        withheld_count=0,
    )
    ctx = context()
    text_brief = await PrecedentTextBrief().brief(
        ctx, diagnosis(), precedents, GateVerdict(True, "x")
    )

    assert text_brief.text is not None
    assert text_brief.text.startswith("## Prior-incident review assignment\n")
    assert "--- precedent (fault_category=message_queue, score=0.500) ---" in text_brief.text
    assert "Use these records as hypotheses only" in text_brief.text
    # Exactly one newline: the diagnosing role's assignment slot adds the second.
    assert text_brief.text.endswith("\n") and not text_brief.text.endswith("\n\n")


@pytest.mark.parametrize(
    ("nomination", "uses_family", "rejection"),
    [
        (Nomination(True, "message_queue", "check the head", 0.5, ""), False, None),
        (Nomination(True, "Message-Queue", "check the head", 0.5, ""), False, None),
        (
            Nomination(True, "queue_gremlins", "look", 0.5, ""),
            False,
            "family_not_in_vocabulary:queue_gremlins",
        ),
        (Nomination(False, "", "", 0.0, ""), False, None),
        (Nomination(False, "", "", 0.0, ""), True, "family_missing"),
        (
            Nomination(True, "message_queue", "this is a fault injection", 0.5, ""),
            False,
            "meta_language_in_directive",
        ),
        # The asymmetry is deliberate: the vocabulary check reads the normalised family
        # and the meta check reads the raw one, so a family that normalises into the
        # vocabulary is still rejected for how it was written.
        (
            Nomination(True, "message queue", "planted by the harness", 0.5, ""),
            False,
            "meta_language_in_directive",
        ),
    ],
)
def test_nomination_validation(
    nomination: Nomination, uses_family: bool, rejection: str | None
) -> None:
    assert validate_nomination(nomination, uses_family=uses_family) == rejection


def test_the_family_vocabulary_is_the_eleven_the_prompt_lists() -> None:
    listed = (PROMPTS / "nominator_system.txt").read_text(encoding="utf-8")
    assert len(ALLOWED_FAMILIES) == 11
    assert all(family in listed for family in ALLOWED_FAMILIES)


def test_a_composition_is_named_for_the_settings_that_select_it() -> None:
    """The name is what the event log records, and what an arm gap is divided by.

    Two compositions were built by varying the gate of a third and kept its name, so a
    quality-gated resampling control recorded itself as the unconditional one. Deriving
    the name from the key is the only arrangement in which they cannot disagree.
    """
    for (gate, delivery), composition in _COMPOSITIONS.items():
        assert composition.name == f"{gate}+{delivery}"


@pytest.mark.parametrize(
    ("gate", "delivery", "name", "gate_type"),
    [
        ("quality", "context", "quality+context", QualityAuditGate),
        ("always", "none", "always+none", type(UNCONDITIONAL_RESAMPLE.gate)),
        ("nominator", "assignment", "nominator+assignment", type(NOMINATED_ASSIGNMENT.gate)),
        ("quality", "none", "quality+none", QualityAuditGate),
        ("quality", "assignment", "quality+assignment", QualityAuditGate),
    ],
)
def test_the_policy_resolution_table(gate: str, delivery: str, name: str, gate_type: type) -> None:
    policy = policy_for(
        SecondPassSettings(enabled=True, gate=cast(Any, gate), delivery=cast(Any, delivery)),
        MemorySettings(),
    )

    assert policy is not None
    assert policy.name == name
    assert isinstance(policy.gate, gate_type)


def test_a_disabled_second_pass_resolves_to_nothing() -> None:
    assert policy_for(SecondPassSettings(enabled=False), MemorySettings()) is None


def test_an_unmapped_combination_fails_rather_than_silently_choosing_an_arm() -> None:
    with pytest.raises(SecondPassConfigError):
        policy_for(
            SecondPassSettings(enabled=True, gate="nominator", delivery="context"),
            MemorySettings(),
        )


def test_delivery_critique_is_rejected() -> None:
    with pytest.raises(ValueError, match="critique"):
        SecondPassSettings(delivery=cast(Any, "critique"))


def test_an_arm_may_vary_the_five_numbers_and_nothing_else() -> None:
    """The selector's two are here because they were the settings nothing applied.

    A run record dumps the whole settings tree, so a bound computed once for sending
    and again for recording is a record that says 400 about a call made at 1600.
    """
    policy = policy_for(
        SecondPassSettings(
            enabled=True,
            gate="quality",
            delivery="context",
            gate_max_tokens=400,
            max_precedents=2,
            selector_max_tokens=400,
            selector_timeout_s=20,
        ),
        MemorySettings(top_k=3),
    )

    assert policy is not None
    assert isinstance(policy.gate, QualityAuditGate)
    assert policy.gate.max_tokens == 400
    assert isinstance(policy.retriever, StorePrecedents)
    assert (policy.retriever.k, policy.retriever.cap) == (3, 2)
    assert isinstance(policy.selector, BlindSelector)
    assert (policy.selector.max_tokens, policy.selector.timeout_s) == (400, 20)
    # The shape is what gives the composition its name and never moves with an arm.
    assert type(policy.briefer) is type(QUALITY_GATED_PRECEDENT.briefer)
    assert type(policy.selector) is type(QUALITY_GATED_PRECEDENT.selector)


def test_the_composition_owns_the_per_leg_depth() -> None:
    """Three hits per leg, with 0.51 mean overlap, is the dose the published cells
    were served. Resolving the depth from memory.top_k instead handed anyone running the
    second pass without an arm file five, which is a different composition."""
    policy = policy_for(SecondPassSettings(enabled=True), MemorySettings())

    assert policy is not None
    assert isinstance(policy.retriever, StorePrecedents)
    assert isinstance(QUALITY_GATED_PRECEDENT.retriever, StorePrecedents)
    assert policy.retriever.k == QUALITY_GATED_PRECEDENT.retriever.k == 3


def test_force_lifts_the_nominators_veto_and_nothing_else() -> None:
    policy = policy_for(
        SecondPassSettings(enabled=True, gate="nominator", delivery="assignment", force=True),
        MemorySettings(),
    )

    assert policy is not None
    assert isinstance(policy.briefer, FamilyAssignmentBrief)
    assert policy.briefer.on_same_family == "verify"
