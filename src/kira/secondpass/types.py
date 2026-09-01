"""Value types and strategy protocols for the second pass.

No module here performs I/O, reads settings, or calls a model. The strategies do both;
their inputs and outputs are these frozen records, which is what makes a policy testable
without an endpoint or a store.

The prototype passed two of these values as undeclared state keys injected into the
diagnosing role's state and defensively stripped on the way out, because a leaked flag
would silently disable second opinions on a later re-diagnosis. Here they are the
``brief`` and ``retrieval`` parameters of :class:`RcaRunner`, so there is nothing to
strip and nothing to leak.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from kira.contracts.artifacts import RootCause
from kira.contracts.state import IncidentState
from kira.llm.types import Completion
from kira.memory.ports import SearchHit, SearchScope
from kira.settings.profiles import SamplingSet

__all__ = [
    "Brief",
    "Briefer",
    "DiagnosisRun",
    "Gate",
    "GateVerdict",
    "NamedQuery",
    "Precedent",
    "PrecedentSet",
    "PrecedentSource",
    "PrecedentSourceError",
    "PrecedentStore",
    "QueryBuilder",
    "RcaRunner",
    "SecondPassConfigError",
    "SecondPassContext",
    "SecondPassEmitter",
    "SecondPassPolicy",
    "SecondPassResult",
    "Selection",
    "Selector",
    "SnapshotProvider",
    "clamped_confidence",
    "first_json_object",
]

_JSON_SPAN = re.compile(r"\{.*\}", re.S)


def first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first greedy ``{...}`` span of a bounded reply and decode it.

    Shared by the gate, the nominator and the selector because all three ask for one
    JSON object and all three have to survive a model that wraps it in prose. The
    greedy span and the strict decode are carried over unchanged: the repairing decoder
    the submission path uses would accept replies these three rejected, which moves the
    fire, nomination and selection rates the published numbers were measured at.

    Args:
        text: The model's reply.

    Returns:
        The decoded object, or None when there is none or it is not an object.
    """
    match = _JSON_SPAN.search(text or "")
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def clamped_confidence(value: object) -> float:
    """Read a model-supplied confidence, defaulting to zero on anything unusable.

    Args:
        value: Whatever the reply carried.

    Returns:
        The value clamped to [0, 1], or 0.0.
    """
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


class SecondPassConfigError(RuntimeError):
    """A composition names no published configuration, or contradicts its own prompt."""


class PrecedentSourceError(RuntimeError):
    """Retrieval could not be performed. Resolves to the first opinion under every policy."""


@dataclass(frozen=True, slots=True)
class NamedQuery:
    """One retrieval query, and the name its results are audited under."""

    name: str
    text: str


@dataclass(frozen=True, slots=True)
class Precedent:
    """One retrieved record, reduced to what a brief may show."""

    record_id: str
    fault_category: str
    summary: str
    score: float | None
    """None is legal. Only ranked retrieval assigns a score; a pattern scan does not."""
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrecedentSet:
    """Retrieval output, with enough provenance to replay it offline."""

    precedents: tuple[Precedent, ...]
    queries: tuple[NamedQuery, ...]
    per_query_ids: Mapping[str, tuple[str, ...]]
    withheld_count: int
    """Size of the leave-one-out list actually applied. Zero on a generalisation arm
    means the cell was served its own precedent and its result is void, which is
    checkable after the fact only because the count is recorded."""


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Whether a second look is warranted, and why."""

    fires: bool
    reason: str
    """Audited verbatim: quality_gate_pass, quality_gate_fire, always, nominator."""
    weakness: str = ""
    confidence: float | None = None
    """Prompted for and never read by the mechanism. Recorded because a calibration
    study needs it and discarding it costs the same as keeping it."""
    parsed: bool = True
    """False when the call or its parse failed."""
    overrules_brief: bool = False
    """The quality audit lifting the nominator's veto.

    The two gates are ORed, never ANDed. Over the 116-cell campaign the nominator gate
    caught 35 of 69 wrong first opinions and the quality gate 61 of 69 at the same
    precision; requiring both scored 34 of 69, worse than the nominator alone, because
    40 audit-flagged cells carry a differs=false nomination and 27 of those first
    opinions are wrong.
    """


@dataclass(frozen=True, slots=True)
class Brief:
    """What the second run receives beyond the alert.

    ``text is None`` is a blind pass: a second diagnosis with no additional input. That
    is Resample's normal state and MA+S's empty-retrieval degradation, and the two must
    be indistinguishable to the second run for the paired contrast to mean anything.
    """

    text: str | None
    kind: Literal["none", "precedent_text", "family_assignment"]
    cancels_run: bool = False
    reason: str = ""
    family: str | None = None
    directive: str | None = None


@dataclass(frozen=True, slots=True)
class Selection:
    """The blinded selector's decision, with everything needed to audit it."""

    chosen: RootCause | None
    source: Literal["first", "second", "none"]
    public_winner: str
    """"A", "B", "tie", "unopposed" or "fallback"."""
    confidence: float
    reason: str
    causal_check: str
    label_to_source: Mapping[str, str]
    fallback_reason: str | None
    prompt: str
    """The exact blinded prompt, so a selection can be re-judged offline."""
    raw_response: str


@dataclass(frozen=True, slots=True)
class DiagnosisRun:
    """One completed diagnosis, as the second pass needs to see it."""

    artifact: RootCause | None
    run_id: str | None
    messages: tuple[Mapping[str, object], ...]
    trace_summaries: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class SecondPassResult:
    """The outcome, whether or not a second run happened.

    Both run ids are reachable through ``first`` and ``second``. The prototype returned
    one of the two states, so the loser's run id never entered the ordered path: both
    rows existed in the database but the chain was lossy.
    """

    chosen: RootCause | None
    source: Literal["first", "second", "none"]
    skipped: str | None
    """None when both runs happened and were selected over."""
    first: DiagnosisRun
    second: DiagnosisRun | None = None
    selection: Selection | None = None
    verdict: GateVerdict | None = None
    brief: Brief | None = None
    precedents: PrecedentSet | None = None


class RcaRunner(Protocol):
    """A diagnosis run, injected by orchestration so this package imports no role."""

    async def __call__(
        self,
        state: IncidentState,
        *,
        brief: str | None,
        retrieval: bool,
        run_label: str,
    ) -> DiagnosisRun:
        """Run one diagnosis to a submission.

        Args:
            state: The incident state the run is entered with.
            brief: Text rendered into the role's assignment slot, or None for a run
                that receives nothing beyond the incident.
            retrieval: Whether the run holds the retrieval tools and the memory prompt
                block. Both diagnosis runs of a second pass are retrieval-free.
            run_label: Which of the two runs this is, for the audit record.

        Returns:
            What the run produced.
        """
        ...


class PrecedentStore(Protocol):
    """The narrow read port the second pass needs from the experience store.

    :class:`kira.memory.retrieval.Retriever` satisfies it. Narrow on purpose: this
    package reads precedents and never writes, opens or re-ranks a record.
    """

    def search(
        self,
        query: str,
        *,
        scope: SearchScope = "evidence",
        exclude: Sequence[str] = (),
    ) -> list[SearchHit]:
        """Return the records worth showing, best first, possibly none.

        Args:
            query: The query text.
            scope: Match against the evidence projection or the lesson.
            exclude: Record ids to withhold.

        Returns:
            Hits carrying their scores.
        """
        ...


class SnapshotProvider(Protocol):
    """Collects a namespace overview for the hypothesis-free retrieval query."""

    async def __call__(self) -> str:
        """Return the snapshot text.

        Returns:
            The rendered snapshot, or an empty string when none could be collected.
        """
        ...


class SecondPassEmitter(Protocol):
    """Where the mechanism's decisions are recorded.

    One surface, so that emitting from inside a branch and forgetting the other branch
    is structurally impossible. The prototype had that habit and the events a cell left
    behind depended on which mode branch it had taken.

    The named composition is bound when the emitter is built rather than passed at each
    call, so a strategy can record a decision without being handed the name of the
    policy it happens to be composed into.
    """

    def gate(self, verdict: GateVerdict) -> None:
        """Record the gate's decision.

        Args:
            verdict: What the gate decided.
        """
        ...

    def search(self, precedents: PrecedentSet) -> None:
        """Record a retrieval, its queries and what they returned.

        Args:
            precedents: The merged result.
        """
        ...

    def search_failed(self, error: Exception) -> None:
        """Record a retrieval that could not be performed.

        Args:
            error: Why it failed.
        """
        ...

    def nomination(self, *, family: str | None, confidence: float | None, reason: str) -> None:
        """Record the nominator's verdict.

        Args:
            family: The nominated fault family, when one survived validation.
            confidence: The nominator's own confidence.
            reason: Why the nomination was accepted or rejected.
        """
        ...

    def brief(self, brief: "Brief") -> None:
        """Record what the second run was given.

        Args:
            brief: The assembled brief.
        """
        ...

    def selection(
        self,
        selection: "Selection",
        *,
        brief: "Brief | None",
        precedents: PrecedentSet | None,
    ) -> None:
        """Record the blinded selection.

        Args:
            selection: What was chosen and why.
            brief: What the second run received.
            precedents: What retrieval returned, when it ran.
        """
        ...

    def skipped(
        self,
        reason: str,
        *,
        verdict: GateVerdict | None,
        precedents: PrecedentSet | None,
    ) -> None:
        """Record a terminal path on which no second run happened.

        Recorded even when no model call was made. A skip that leaves no trace makes
        the mechanism's dose unmeasurable, and the dose is what an arm gap has to be
        divided by.

        Args:
            reason: Which fail-safe fired.
            verdict: The gate's decision, when one was taken.
            precedents: What retrieval returned before the skip, when it ran.
        """
        ...


@dataclass(frozen=True, slots=True)
class SecondPassContext:
    """Everything a strategy may reach. Nothing is read from the environment."""

    session_id: str
    alert: Mapping[str, object]
    state: IncidentState
    completion: Completion
    sampling: SamplingSet
    """The auxiliary set: short structured classifications, decoded without the
    diversity knobs the agent profile is tuned for."""
    memory: PrecedentStore | None
    """None for a policy that reads no store, which is Resample's normal state."""
    snapshot: SnapshotProvider
    withheld_record_ids: frozenset[str]
    emit: SecondPassEmitter


class Gate(Protocol):
    """Decides whether a second look is warranted."""

    name: str

    async def decide(self, ctx: SecondPassContext, first: RootCause) -> GateVerdict:
        """Judge the first opinion.

        Args:
            ctx: The pass's context.
            first: The diagnosis to second-guess.

        Returns:
            Whether to look again, and why.
        """
        ...


class QueryBuilder(Protocol):
    """Composes the retrieval queries, from observations and never from a conclusion."""

    async def build(self, ctx: SecondPassContext, first: RootCause) -> list[NamedQuery]:
        """Build this leg's queries.

        Args:
            ctx: The pass's context.
            first: The diagnosis whose observations may be used.

        Returns:
            The queries, in the order they should be searched.
        """
        ...


class PrecedentSource(Protocol):
    """Turns queries into a bounded, deduplicated set of precedent cards."""

    async def fetch(self, ctx: SecondPassContext, queries: Sequence[NamedQuery]) -> PrecedentSet:
        """Search with every query and merge the results.

        Args:
            ctx: The pass's context.
            queries: What to search with.

        Returns:
            The merged set, with per-query provenance.

        Raises:
            PrecedentSourceError: When the store could not be read.
        """
        ...


class Briefer(Protocol):
    """Decides what, if anything, the second run is told."""

    async def brief(
        self,
        ctx: SecondPassContext,
        first: RootCause,
        precedents: PrecedentSet | None,
        verdict: GateVerdict,
    ) -> Brief:
        """Assemble the second run's brief.

        Args:
            ctx: The pass's context.
            first: The first opinion.
            precedents: What retrieval returned, or None when none ran.
            verdict: The gate's decision, whose overrule may lift a veto.

        Returns:
            The brief, which may cancel the second run.
        """
        ...


class Selector(Protocol):
    """Chooses between the two diagnoses under shuffled labels."""

    async def select(
        self,
        ctx: SecondPassContext,
        first: RootCause,
        second: RootCause | None,
        evidence: str,
    ) -> Selection:
        """Arbitrate.

        Args:
            ctx: The pass's context.
            first: The first opinion.
            second: The second opinion, or None when its run produced nothing.
            evidence: The bundled live tool results both runs gathered.

        Returns:
            The decision, which resolves to the first opinion on any failure.
        """
        ...


@dataclass(frozen=True, slots=True)
class SecondPassPolicy:
    """A named composition. ``strategies.py`` holds all three published ones."""

    name: str
    gate: Gate
    query: QueryBuilder | None
    """None means no retrieval at all."""
    retriever: PrecedentSource | None
    briefer: Briefer
    selector: Selector
    on_retrieval_error: Literal["skip", "continue"] = "skip"
