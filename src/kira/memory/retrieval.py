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

One thing deliberately *not* done: records are not weighted by lifecycle state. A
precedent that turned out wrong carries the same weight as one that turned out right,
because knowing what was already tried and how it failed is exactly what a later
incident needs. See :mod:`kira.contracts.lifecycle`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from kira.memory import bm25
from kira.memory.ports import SearchHit, SearchScope, SessionMemory, TextSearch

__all__ = ["Retriever", "build_query"]

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
    disables the floor, which restores the behaviour of always returning k results.
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

        hits: list[SearchHit] = []
        for index, score in bm25.rank(documents, query):
            if len(hits) >= self.k:
                break
            if score < floor:
                # Ranked best first, so the first hit under the floor ends the list.
                break
            hits.append(_to_hit(candidates[index], scope, score))
        return hits

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


def _document(memory: SessionMemory, scope: SearchScope) -> str:
    """Select the text a record is indexed and matched on.

    The incident summary is deliberately not included. It is prose containing the prior
    conclusion, and the projections keep causal text out of what retrieval returns so a
    later incident cannot read an answer it has not reached.
    """
    return memory.lesson_text if scope == "lesson" else memory.evidence_text


def _to_hit(memory: SessionMemory, scope: SearchScope, score: float | None = None) -> SearchHit:
    return SearchHit(
        session_id=memory.session_id,
        summary=_document(memory, scope),
        metadata={
            "state": memory.lifecycle_state,
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
