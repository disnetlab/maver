"""The whole configuration surface of the mechanism, on one screen.

Three published compositions and the resolver that maps settings onto them. An
unmapped combination raises rather than falling back: a silent fallback to a different
composition is how an experiment records the wrong label.

| Paper name | Gate | Query and retrieve | Brief to the second run | Selector |
|---|---|---|---|---|
| MA+S | retrieval-free quality audit | two observation-keyed queries, BM25 | precedent text | blind, label-shuffled |
| MA+R | the nominator's differing family | the same | a fault-family assignment, no text | the same |
| Resample | always | none | nothing | the same |
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

__all__ = ["MAR", "MAS", "RESAMPLE", "policy_for"]

_MAR_NOMINATOR = FamilyNominator(max_tokens=900)
"""One nominator object for MA+R, held by both its gate and its briefer, so the cap and
the vocabulary guard are configured once and the composition issues one call."""

# Twelve pre-anchor observation lines rather than the eight-line offline champion. The
# measured campaign ran twelve with the tail trim off, and this is the composition those
# cells are reproduced from; the A/B that would settle which value is better was
# specified and never executed, and the effect reverses by corpus.
_MEASURED_QUERY = MergedQuery(ObservationQuery(), PreAnchorQuery(observation_lines=12))

MAS = SecondPassPolicy(
    name="ma-s",
    gate=QualityAuditGate(max_tokens=600, on_unreadable="fire"),
    query=_MEASURED_QUERY,
    retriever=StorePrecedents(k=3, cap=6),
    briefer=PrecedentTextBrief(on_empty="blind_pass"),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
    on_retrieval_error="skip",
)
"""The published configuration: 48.3% of problems diagnosed against 36.8% for the
baseline, 19 wins to 6 losses, p = 0.015."""

RESAMPLE = SecondPassPolicy(
    name="resample",
    gate=AlwaysGate(),
    query=None,
    retriever=None,
    briefer=NoBrief(),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
)
"""The attribution control. A memory-mode gain must beat plain diagnose-twice or the
memory claim is not supportable, which makes shipping this configuration part of making
the headline claim readable rather than an optional extra."""

MAR = SecondPassPolicy(
    name="ma-r",
    gate=NominatorGate(),
    query=_MEASURED_QUERY,
    retriever=StorePrecedents(k=5, cap=6),
    briefer=FamilyAssignmentBrief(nominator=_MAR_NOMINATOR, on_same_family="cancel"),
    selector=BlindSelector(max_tokens=1600, timeout_s=180),
    on_retrieval_error="skip",
)
"""The earlier variant, a reproduced negative result: guided minus resample is -1.7
composite points, CI [-11.1, +7.3]. It varies three axes at once against MA+S."""

_PUBLISHED: dict[tuple[str, str], SecondPassPolicy] = {
    ("quality", "context"): MAS,
    ("always", "none"): RESAMPLE,
    ("nominator", "assignment"): MAR,
    # Two unpublished combinations that the shipped settings can express and that a
    # reader will reach for: a quality-gated resampling control, and a quality-gated
    # MA+R in which the audit keeps the nominator's voice without its veto.
    ("quality", "none"): replace(
        RESAMPLE, gate=QualityAuditGate(max_tokens=600, on_unreadable="fire")
    ),
    ("quality", "assignment"): replace(
        MAR, gate=QualityAuditGate(max_tokens=600, on_unreadable="defer")
    ),
}


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
    base = _PUBLISHED.get((settings.gate, settings.delivery))
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
        # The composition owns the per-leg depth. Three hits per leg against MA+R's
        # five is part of what separates the two, so reading it off memory.top_k left
        # MA+S unable to run the value its own module publishes. What memory.top_k
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
