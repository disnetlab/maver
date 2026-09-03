"""The whole configuration surface of the mechanism, on one screen.

Five compositions and the resolver that maps settings onto them. An unmapped
combination raises rather than falling back: a silent fallback to a different
composition is how an experiment records the wrong mechanism.

A composition is named for the two settings that select it, ``gate+delivery``, and for
nothing else. Naming them after the arms they were measured under put an experiment's
vocabulary in the code that has to keep working when that experiment is over — and it
drifted immediately, because two compositions built by varying a third derived their
name from the one they were varied from and recorded a mechanism they were not. The
published label an arm carries lives in that arm's file, beside the campaign it names.

| Gate | Query and retrieve | Brief to the second run | Selector |
|---|---|---|---|
| quality | two observation-keyed queries, BM25 | precedent text | blind, label-shuffled |
| nominator | the same | a fault-family assignment, no text | the same |
| always | none | nothing | the same |
"""

from dataclasses import replace

from kira.secondpass.brief import FamilyAssignmentBrief, NoBrief, PrecedentTextBrief
from kira.secondpass.gate import AlwaysGate, NominatorGate, QualityAuditGate
from kira.secondpass.nominator import FamilyNominator
from kira.secondpass.query import MergedQuery, ObservationQuery, PreAnchorQuery
from kira.secondpass.retrieve import StorePrecedents
from kira.secondpass.select import BlindSelector
from kira.secondpass.types import SecondPassConfigError, SecondPassPolicy
from kira.settings.groups import MemorySettings, SecondPassSettings

__all__ = [
    "NOMINATED_ASSIGNMENT",
    "QUALITY_GATED_PRECEDENT",
    "UNCONDITIONAL_RESAMPLE",
    "policy_for",
]

_NOMINATOR = FamilyNominator(max_tokens=900)
"""One nominator object, held by both the gate and the briefer of the composition that
uses it, so the cap and the vocabulary guard are configured once and it issues one
call."""

# Twelve pre-anchor observation lines rather than the eight-line offline champion. The
# measured campaign ran twelve with the tail trim off, and this is the composition those
# cells are reproduced from; the A/B that would settle which value is better was
# specified and never executed, and the effect reverses by corpus.
_MEASURED_QUERY = MergedQuery(ObservationQuery(), PreAnchorQuery(observation_lines=12))

QUALITY_GATED_PRECEDENT = SecondPassPolicy(
    name="quality+context",
    gate=QualityAuditGate(max_tokens=600, on_unreadable="fire"),
    query=_MEASURED_QUERY,
    retriever=StorePrecedents(k=3, cap=6),
    briefer=PrecedentTextBrief(on_empty="blind_pass"),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
    on_retrieval_error="skip",
)
"""A retrieval-free audit of the first diagnosis decides, and precedent text is what the
second run receives. 48.3% of problems diagnosed against 36.8% for the baseline, 19 wins
to 6 losses, p = 0.015."""

UNCONDITIONAL_RESAMPLE = SecondPassPolicy(
    name="always+none",
    gate=AlwaysGate(),
    query=None,
    retriever=None,
    briefer=NoBrief(),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
)
"""The attribution control. A memory-mode gain must beat plain diagnose-twice or the
memory claim is not supportable, which makes shipping this configuration part of making
the headline claim readable rather than an optional extra."""

NOMINATED_ASSIGNMENT = SecondPassPolicy(
    name="nominator+assignment",
    gate=NominatorGate(),
    query=_MEASURED_QUERY,
    retriever=StorePrecedents(k=5, cap=6),
    briefer=FamilyAssignmentBrief(nominator=_NOMINATOR, on_same_family="cancel"),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
    on_retrieval_error="skip",
)
"""The nominator both decides and supplies the content: a verdict that the retrieved
family differs fires the second run and is all it is told. A reproduced negative result
— guided minus unconditional resampling is -1.7 composite points, CI [-11.1, +7.3] —
and it varies three axes at once against the quality-gated precedent composition."""

_COMPOSITIONS: dict[tuple[str, str], SecondPassPolicy] = {
    ("quality", "context"): QUALITY_GATED_PRECEDENT,
    ("always", "none"): UNCONDITIONAL_RESAMPLE,
    ("nominator", "assignment"): NOMINATED_ASSIGNMENT,
    # Two further combinations the shipped settings can express and that a reader will
    # reach for: a quality-gated resampling control, and a quality-gated nomination in
    # which the audit keeps the nominator's voice without its veto.
    ("quality", "none"): replace(
        UNCONDITIONAL_RESAMPLE,
        name="quality+none",
        gate=QualityAuditGate(max_tokens=600, on_unreadable="fire"),
    ),
    ("quality", "assignment"): replace(
        NOMINATED_ASSIGNMENT,
        name="quality+assignment",
        gate=QualityAuditGate(max_tokens=600, on_unreadable="defer"),
    ),
}
"""Every composition, keyed by the settings that select it.

The name is the key, and a test holds it there. Both derived compositions above kept
the name of the one they were built from, so a quality-gated resampling control
recorded itself in the event log as the unconditional one — the mechanism a reader
would divide an arm gap by."""


def policy_for(settings: SecondPassSettings, memory: MemorySettings) -> SecondPassPolicy | None:
    """Resolve the configured composition, or None when no second pass runs.

    Args:
        settings: The second-pass group.
        memory: The memory group, whose top_k is the ceiling the store can return
            and therefore the ceiling on a query leg.

    Returns:
        The composition to run, or None when the mechanism is disabled.

    Raises:
        SecondPassConfigError: When the gate and delivery combination names no
            composition. Failing here is deliberate: a silent fallback to a different
            composition is how an experiment records the wrong label.
    """
    if not settings.enabled:
        return None
    base = _COMPOSITIONS.get((settings.gate, settings.delivery))
    if base is None:
        raise SecondPassConfigError(
            f"no second-pass composition for gate={settings.gate!r} delivery={settings.delivery!r}"
        )
    return _apply(base, settings, memory)


def _apply(
    base: SecondPassPolicy, settings: SecondPassSettings, memory: MemorySettings
) -> SecondPassPolicy:
    """Overlay the five numbers an arm may legitimately vary.

    Only five, because everything else about a composition is what gives it its name.
    An arm that wants a different gate, briefer or selector is a different arm and says
    so in its own file.
    """
    gate = base.gate
    if isinstance(gate, QualityAuditGate):
        gate = QualityAuditGate(
            max_tokens=settings.gate_max_tokens, on_unreadable=gate.on_unreadable
        )
    briefer = base.briefer
    if isinstance(briefer, FamilyAssignmentBrief):
        briefer = FamilyAssignmentBrief(
            nominator=FamilyNominator(max_tokens=settings.nominator_max_tokens),
            # The force flag lifts the nominator's veto and nothing else. It never
            # overrode a quality-gate pass in the prototype, which returned before the
            # flag was read.
            on_same_family="verify" if settings.force else briefer.on_same_family,
        )
    retriever = base.retriever
    if isinstance(retriever, StorePrecedents):
        # The composition owns the per-leg depth. Three hits per leg against the
        # nominated composition's five is part of what separates the two, so reading it
        # off memory.top_k left one of them unable to run the value its own definition
        # above publishes. What memory.top_k
        # still does here is impose a ceiling rather than assert a depth: the store's
        # retriever is built at that k, so a shallower one cannot hand a leg more than
        # it returns and a policy claiming otherwise would misreport the dose.
        retriever = replace(
            retriever, k=min(retriever.k, memory.top_k), cap=settings.max_precedents
        )
    selector = base.selector
    if isinstance(selector, BlindSelector):
        selector = BlindSelector(
            max_tokens=settings.selector_max_tokens, timeout_s=settings.selector_timeout_s
        )
    return replace(base, gate=gate, briefer=briefer, retriever=retriever, selector=selector)
