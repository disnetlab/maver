"""The gated second diagnostic opinion, expressed as composable strategies.

The mechanism is the system's headline optional feature and the only one whose effect
survives its confidence interval: 48.3% of problems diagnosed against 36.8% for the
one-pass baseline, 19 wins to 6 losses, p = 0.015. It is therefore the part of this
artifact a reader is most likely to want to re-run, which is why it is five small
strategies over one flow rather than the prototype's single 777-line function with five
decision points and four mode branches interleaved.

Five stages, one per line of the manuscript's Algorithm 1: gate, query, retrieve, brief,
select. Three published compositions of them, in :mod:`kira.secondpass.strategies`.

Invariants the mechanism rests on, each preserved deliberately: the first diagnosis is
retrieval-free; the retrieval query carries observations and never the conclusion; the
second diagnosis is blind to the first; the selector sees shuffled labels; an empty
result set degrades to a blind pass rather than to no pass; and every failure resolves
to the first answer.
"""

from kira.secondpass.audit import JournalEmitter
from kira.secondpass.brief import FamilyAssignmentBrief, NoBrief, PrecedentTextBrief
from kira.secondpass.gate import AlwaysGate, NominatorGate, QualityAuditGate
from kira.secondpass.nominator import (
    ALLOWED_FAMILIES,
    FamilyNominator,
    Nomination,
    parse_nomination,
    validate_nomination,
)
from kira.secondpass.query import MergedQuery, ObservationQuery, PreAnchorQuery
from kira.secondpass.retrieve import StorePrecedents, render_precedents
from kira.secondpass.runner import SecondPass
from kira.secondpass.select import BlindSelector, blinded_labels, build_evidence_bundle
from kira.secondpass.strategies import MAR, MAS, RESAMPLE, policy_for
from kira.secondpass.types import (
    Brief,
    DiagnosisRun,
    GateVerdict,
    NamedQuery,
    Precedent,
    PrecedentSet,
    PrecedentSourceError,
    PrecedentStore,
    RcaRunner,
    SecondPassConfigError,
    SecondPassContext,
    SecondPassEmitter,
    SecondPassPolicy,
    SecondPassResult,
    Selection,
    SnapshotProvider,
)

__all__ = [
    "ALLOWED_FAMILIES",
    "MAR",
    "MAS",
    "RESAMPLE",
    "AlwaysGate",
    "BlindSelector",
    "Brief",
    "DiagnosisRun",
    "FamilyAssignmentBrief",
    "FamilyNominator",
    "GateVerdict",
    "JournalEmitter",
    "MergedQuery",
    "NamedQuery",
    "NoBrief",
    "Nomination",
    "NominatorGate",
    "ObservationQuery",
    "PreAnchorQuery",
    "Precedent",
    "PrecedentSet",
    "PrecedentSourceError",
    "PrecedentStore",
    "PrecedentTextBrief",
    "QualityAuditGate",
    "RcaRunner",
    "SecondPass",
    "SecondPassConfigError",
    "SecondPassContext",
    "SecondPassEmitter",
    "SecondPassPolicy",
    "SecondPassResult",
    "Selection",
    "SnapshotProvider",
    "StorePrecedents",
    "blinded_labels",
    "build_evidence_bundle",
    "parse_nomination",
    "policy_for",
    "render_precedents",
    "validate_nomination",
]
