"""Reading the experience store directly, while the agent-facing tools stay disabled.

Leave-one-out exclusion is applied here as well as inside the retrieval tool. This path
reads the store directly, so an arm that supplied a withheld set and relied on the
tool-layer filter would serve the evaluated problem its own precedent and fail open:
silently, and in the direction that flatters the result.

The card a brief shows is a bounded summary, not the record. The manuscript's "receives
the precedent records verbatim" overstates it; "bounded summary cards of the retrieved
records" is what happens. No grade or outcome appears on a card, which is why the
corpus had to be filtered to externally-graded-correct rows before the arm could run: an
quality-gated composition's incorrect card is indistinguishable transplant poison,
and it has no nominator left to
filter it.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from maver.memory.ports import SearchHit, SearchScope
from maver.secondpass.types import (
    NamedQuery,
    Precedent,
    PrecedentSet,
    PrecedentSourceError,
    SecondPassContext,
)

__all__ = ["StorePrecedents", "render_precedents"]

_SNIPPET_CHARS = 500


def render_precedents(precedents: Sequence[Precedent]) -> str:
    """Render precedent cards for a prompt.

    The format is inside a measured prompt and is carried byte-for-byte, with one
    correction: the prototype formatted the score with ``:.3f`` against a value that its
    own pattern-scan retrieval path returns as ``None``, which raised. A null score now
    renders as ``n/a`` rather than as a fabricated ``0.000``.

    Args:
        precedents: The retrieved records, in merged order.

    Returns:
        The cards, joined by newlines. Empty when nothing was retrieved.
    """
    cards = []
    for precedent in precedents:
        score = "n/a" if precedent.score is None else f"{precedent.score:.3f}"
        body = " ".join(precedent.summary.split())[:_SNIPPET_CHARS]
        cards.append(
            f"--- precedent (fault_category={precedent.fault_category}, score={score}) ---\n{body}"
        )
    return "\n".join(cards)


def _to_precedent(hit: SearchHit) -> Precedent:
    metadata: Mapping[str, str] = {k: str(v) for k, v in hit.metadata.items()}
    return Precedent(
        record_id=hit.session_id,
        fault_category=metadata.get("fault_category") or "?",
        summary=hit.summary,
        score=hit.score,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class StorePrecedents:
    """Searches the store once per query and merges the hits, evidence-first.

    ``k`` bounds each query and ``cap`` bounds the union. Both matter: the campaign ran
    three hits per leg with 0.51 mean overlap for a merged mean of 5.49, so the widely
    quoted "up to three precedents" understates the delivered dose by about two-fold.
    """

    k: int = 3
    cap: int = 6
    scope: SearchScope = "evidence"

    async def fetch(self, ctx: SecondPassContext, queries: Sequence[NamedQuery]) -> PrecedentSet:
        """Search with every query and merge the results.

        Args:
            ctx: The pass's context, carrying the store and the withheld ids.
            queries: What to search with, in merge order.

        Returns:
            The merged set, capped, with per-query provenance.

        Raises:
            PrecedentSourceError: When no store is configured, or reading it failed.
        """
        if ctx.memory is None:
            raise PrecedentSourceError("no experience store is configured for this run")

        # Exclusion travels with the query rather than being read from the environment
        # at call time. The prototype over-fetched by the exclusion count and filtered
        # afterwards, which is a heuristic and not a guarantee: more excluded rows in
        # the over-fetched head than the count and the final list comes back short.
        exclude = sorted(ctx.withheld_record_ids)

        seen: set[str] = set()
        merged: list[Precedent] = []
        per_query: dict[str, tuple[str, ...]] = {}
        for query in queries:
            if not query.text.strip():
                per_query[query.name] = ()
                continue
            try:
                # Off the loop. The store is a database on another host and the port is
                # synchronous, so called inline this holds the incident for as long as
                # the database takes and nothing above can cut it short: a timeout is a
                # coroutine, and a coroutine cannot run while the loop is inside a
                # synchronous call. Same call, same reason, as the retrieval tool.
                hits = await asyncio.to_thread(
                    ctx.memory.search, query.text, scope=self.scope, exclude=exclude
                )
            except Exception as exc:
                raise PrecedentSourceError(f"searching for {query.name!r} failed: {exc}") from exc
            hits = hits[: self.k]
            per_query[query.name] = tuple(hit.session_id for hit in hits)
            for hit in hits:
                if hit.session_id in seen:
                    continue
                seen.add(hit.session_id)
                merged.append(_to_precedent(hit))

        return PrecedentSet(
            precedents=tuple(merged[: self.cap]),
            queries=tuple(queries),
            per_query_ids=per_query,
            withheld_count=len(ctx.withheld_record_ids),
        )
