"""Finding the past incidents worth showing a present one.

Three decisions here differ from what the prototype deployed, and all three come from
the same offline campaign over this corpus.

**Lexical ranking is the ceiling.** Pure BM25 beat six dense embedders, two
cross-encoders and a model reranker, so it is the default and a vector lane is optional.
That is not a claim about retrieval in general; it is a claim about short records dense
with service names, error strings and resource identifiers.

**A query is short.** The deployed query reached roughly 206 terms by appending
namespace-snapshot lines and tool output, and 84.6% of its body was boilerplate carrying
no incident-specific token. Boilerplate cannot discriminate between records but does
dominate a bag-of-words query, so the long query retrieved roughly six times fewer true
precedents than a short one. The query here is identifiers plus a handful of salient
observations, one per line.

**Retrieval may answer "nothing".** Ranked retrieval returns its top-k whether or not
anything is relevant, and on this corpus roughly 88% of what a role read at k=5 was
irrelevant. A floor on the normalised score lets the retriever return nothing at all,
which is a real answer and a more useful one than five unrelated incidents.

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
from dataclasses import dataclass

from kira.contracts.lifecycle import LifecycleState
from kira.memory import bm25
from kira.memory.lifecycle import LIFECYCLE_WEIGHTS, weight_for
from kira.memory.ports import SearchHit, SearchScope, SessionMemory, TextSearch

__all__ = ["Retriever", "build_query"]

_log = logging.getLogger(__name__)

_BOILERPLATE = (
    "current incident observations",
    "no data",
    "not available",
    "unavailable",
    "none found",
    "no results",
)


def build_query(
    identifiers: Sequence[str],
    observations: Sequence[str],
    *,
    limit: int = 8,
) -> str:
    """Compose the query text for an incident.

    Identifiers first, then the most salient observations, one per line. Lines that
    say nothing incident-specific are dropped: they cannot separate one record from
    another, and in a bag-of-words query they crowd out the terms that can.

    Args:
        identifiers: Alert name, namespace, affected resource, and the like.
        observations: Observation lines, most salient first.
        limit: How many observation lines to keep. Eight was measured; more stops
            helping and considerably more actively hurts.

    Returns:
        The query text, or an empty string when nothing usable was supplied.
    """
    lines = [line.strip() for line in identifiers if line and line.strip()]
    kept = 0
    for line in observations:
        if kept >= limit:
            break
        text = line.strip()
        if not text or any(marker in text.lower() for marker in _BOILERPLATE):
            continue
        lines.append(text)
        kept += 1
    if not lines:
        return ""
    return "Current incident observations:\n" + "\n".join(f"- {line}" for line in lines)


@dataclass(frozen=True)
class Retriever:
    """Ranks a candidate pool and decides whether any of it is worth returning."""

    text: TextSearch
    k: int = 5
    pool: int = 500
    """How many candidates to rank. Corpora here are hundreds of records, so this is
    normally the whole temporally valid corpus."""
    min_score_per_token: float = 0.5
    """Floor on a hit's score divided by the query's token count.

    Divided by token count so the threshold does not move with query length. Zero
    disables the floor, which restores the behaviour of always returning k results —
    what the prototype shipped, and what an arm reproducing one of its campaigns pins
    through ``memory.min_score_per_token`` rather than inheriting the default here.
    """
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
            query: The query text, from :func:`build_query`.
            scope: Match against the evidence projection or the lesson.
            exclude: Session ids to withhold. An evaluation harness supplies these so
                a run cannot retrieve the record its own earlier attempt wrote.

        Returns:
            Hits carrying their scores, at most ``k`` of them.
        """
        tokens = bm25.tokenize(query)
        if not tokens:
            return []

        candidates = self.text.candidates(query, limit=self.pool, scope=scope, exclude=exclude)
        if not candidates:
            return []

        documents = [_document(memory, scope) for memory in candidates]
        floor = self.min_score_per_token * len(tokens)
        states = self._current_states(candidates)

        hits: list[SearchHit] = []
        for index, score in bm25.rank(documents, query):
            if score < floor:
                # Ranked best first, so the first hit under the floor ends the list.
                # The floor reads the lexical score alone, before the lifecycle weight:
                # it is a relevance test calibrated on that score, and applying it after
                # the weight would let the demotion drop a record from the results,
                # which is the exclusion the campaign measured and rejected.
                break
            memory = candidates[index]
            state = states.get(memory.session_id, memory.lifecycle_state)
            hits.append(_to_hit(memory, scope, score * weight_for(state, self.weights), state))

        # Weighting reorders, so the ranker's order no longer holds. The session id
        # breaks a tie, because a tie broken by pool order is a tie broken by whichever
        # database returned the candidates.
        hits.sort(key=lambda hit: (-(hit.score or 0.0), hit.session_id))
        return hits[: self.k]

    def scan(
        self,
        pattern: str,
        *,
        scope: SearchScope = "evidence",
    ) -> list[SearchHit]:
        """Return only records matching a pattern.

        Distinct from :meth:`search` in what an empty result means. Ranked retrieval
        returning nothing means nothing scored above the floor; a scan returning nothing
        means the corpus does not mention this, which is a fact a role can act on.

        Args:
            pattern: A case-insensitive regular expression.
            scope: Which projection to match.

        Returns:
            Matching hits, unscored because nothing ranked them.
        """
        found = self.text.scan_summaries(pattern, n_results=self.k, scope=scope)
        return [_to_hit(memory, scope) for memory in found]

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
