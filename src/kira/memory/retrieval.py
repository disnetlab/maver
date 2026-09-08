"""Finding the past incidents worth showing a present one.

Three decisions here differ from what the prototype deployed, and all three come from
the same offline campaign over this corpus.

**Lexical ranking is the ceiling.** Pure BM25 beat six dense embedders, two
cross-encoders and a model reranker, so it is the default and a vector lane is optional.
That is not a claim about retrieval in general; it is a claim about short records dense
with service names, error strings and resource identifiers — which is why scoring sits
behind a seam. See :mod:`kira.memory.ranking`.

**A query is short.** The deployed query reached roughly 206 terms by appending
namespace-snapshot lines and tool output, and 84.6% of its body was boilerplate carrying
no incident-specific token. Boilerplate cannot discriminate between records but does
dominate a bag-of-words query, so the long query retrieved roughly six times fewer true
precedents than a short one. Nothing here builds a query: a role writes its own, and the
second pass builds two in :mod:`kira.secondpass.query`, which is where that finding is
acted on and where the filter it suggests is a knob rather than a default.

**Retrieval may answer "nothing".** Ranked retrieval returns its top-k whether or not
anything is relevant, and on this corpus roughly 88% of what a role read at k=5 was
irrelevant. A floor on the normalised score lets the retriever return nothing at all,
which is a real answer and a more useful one than five unrelated incidents. The floor
ships off, because no measured run set one; it is a knob the offline screen argues for.

**A record is demoted for its lifecycle state, never suppressed.** Excluding invalidated
records was measured and lost ground: coverage fell 88% -> 72% and 90% -> 68%, and 5.7pp
and 9.1pp fewer queries were served a true precedent first, because invalidation records
what the prior session's remediation did rather than whether what it observed was true.
So an invalidated record ranks at 0.3 of its lexical score and still comes back; knowing
what was already tried and how it failed is exactly what a later incident needs. State is
resolved as a candidate is ranked rather than read from the index, so an operator's
verdict changes the next retrieval and nothing has to be reindexed. See
:mod:`kira.memory.lifecycle`.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from kira.contracts.lifecycle import LifecycleState
from kira.memory.lifecycle import LIFECYCLE_WEIGHTS, weight_for
from kira.memory.ports import SearchHit, SearchScope, SessionMemory, TextSearch
from kira.memory.ranking import Bm25Ranker, Ranker

__all__ = ["Retriever"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Retriever:
    """Ranks a candidate pool and decides whether any of it is worth returning."""

    text: TextSearch
    k: int = 5
    pool: int = 500
    """How many candidates to rank, from ``memory.candidate_pool``.

    Corpora here are hundreds of records, so this is normally the whole temporally valid
    corpus. A backend fills it in write order rather than by rank — ranking is the
    retriever's job, not the database's — so a corpus larger than this loses its newest
    records from every retrieval, which is why an outgrown pool says so below."""
    ranker: Ranker = field(default_factory=Bm25Ranker)
    """Scores the pool and decides how much of it is worth returning.

    Relevance belongs to the ranker because only what produced a score knows its scale.
    Everything this class does around it — the lifecycle weight, the tie-break, the cut
    to k — is the same whatever scored the pool."""
    states: Callable[[Sequence[str]], Mapping[str, str]] | None = None
    """Reads the current lifecycle state of many candidates at once.

    :meth:`kira.memory.ports.SessionStore.get_states` is the implementation. It is here
    rather than read off each record because the state stored beside a record is a
    write-time snapshot: an operator's verdict, or a re-grade, changes the session row
    and leaves that snapshot behind. Supplying nothing falls back to the snapshot, which
    is what a deployment retrieving from a corpus whose sessions it does not hold has.
    """
    weights: Mapping[LifecycleState, float] = LIFECYCLE_WEIGHTS
    """What a record in each state is worth, from :func:`kira.memory.lifecycle.resolve_weights`."""

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
            exclude: Session ids to withhold. An evaluation harness supplies these so
                a run cannot retrieve the record its own earlier attempt wrote.

        Returns:
            Hits carrying their scores, at most ``k`` of them.
        """
        if not query.strip():
            # Before the backend, not after. A blank query is not a search, and asking
            # a database to narrow a corpus by nothing costs a round trip to be told so.
            return []

        candidates = self.text.candidates(query, limit=self.pool, scope=scope, exclude=exclude)
        if not candidates:
            return []
        if len(candidates) >= self.pool:
            # The pool is filled oldest first, so what a full one drops is the recent
            # experience this deployment most wants back. Silence here is how a store
            # that has outgrown its pool looks exactly like one that is working.
            _log.warning(
                "the candidate pool filled at %d records, so the newest were not scored; "
                "raise memory.candidate_pool above the corpus size",
                self.pool,
            )

        documents = [_document(memory, scope) for memory in candidates]
        states = self._current_states(candidates)

        hits: list[SearchHit] = []
        # The ranker has already dropped what it does not consider relevant, and it
        # judged that on the raw score, before any lifecycle weight. That order matters:
        # applying a relevance test after the demotion would let the demotion drop a
        # record from the results, which is the exclusion the campaign measured and
        # rejected.
        for index, score in self.ranker.rank(query, documents):
            memory = candidates[index]
            state = states.get(memory.session_id, memory.lifecycle_state)
            hits.append(_to_hit(memory, scope, score * weight_for(state, self.weights), state))

        # Weighting reorders, so the ranker's order no longer holds. The session id
        # breaks a tie, because a tie broken by pool order is a tie broken by whichever
        # database returned the candidates.
        hits.sort(key=lambda hit: (-(hit.score or 0.0), hit.session_id))
        return hits[: self.k]

    def open(self, session_id: str) -> SessionMemory | None:
        """Read one stored record whole, including what retrieval withholds.

        Args:
            session_id: Which session.

        Returns:
            The record, or None.
        """
        return self.text.open_record(session_id)

    def _current_states(self, candidates: Sequence[SessionMemory]) -> Mapping[str, str]:
        """Resolve every candidate's state now, falling back to its stored snapshot."""
        if self.states is None:
            return {}
        try:
            return self.states([memory.session_id for memory in candidates])
        except Exception:
            # A retrieval must not fail because the session store hiccupped. Each
            # record's own snapshot is stale but serviceable, and returning nothing at
            # all would be a worse answer than returning a slightly mis-ranked one.
            _log.warning("could not read lifecycle states; ranking on the stored ones")
            return {}


def _document(memory: SessionMemory, scope: SearchScope) -> str:
    """Select the text a record is indexed and matched on.

    The incident summary is deliberately not included. It is prose containing the prior
    conclusion, and the projections keep causal text out of what retrieval returns so a
    later incident cannot read an answer it has not reached.
    """
    return memory.lesson_text if scope == "lesson" else memory.evidence_text


def _to_hit(
    memory: SessionMemory,
    scope: SearchScope,
    score: float | None = None,
    state: str | None = None,
) -> SearchHit:
    # The resolved state, not the snapshot, so that whatever reads a hit downstream —
    # prompt formatting, a split by standing — sees what retrieval itself ranked on.
    return SearchHit(
        session_id=memory.session_id,
        summary=_document(memory, scope),
        metadata={
            "state": state or memory.lifecycle_state,
            "fault_category": memory.fault_category,
            "alert_name": memory.alert_name,
            "namespace": memory.namespace,
            "affected_resource": memory.affected_resource,
        },
        incident_summary=memory.incident_summary or None,
        scope=scope,
        evidence_text=memory.evidence_text,
        lesson_text=memory.lesson_text,
        score=score,
    )
