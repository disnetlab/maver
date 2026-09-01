"""The one place a second-pass decision is written down.

Every terminal path emits, including the roughly one incident in three where the gate
declines and no model call is made. The prototype's equivalent emitted from inside each
mode branch, so which events a cell left behind depended on which branch it had taken,
and the skips — the majority of cells — left the least.

The selector's exact prompt and raw reply are recorded on the selection event alone.
They are large, and they are what makes a selection re-judgeable offline months later
against a different judge.
"""

import json

from kira.audit.journal import Journal
from kira.contracts.events import SecondPassEvent
from kira.secondpass.types import Brief, GateVerdict, PrecedentSet, Selection

__all__ = ["JournalEmitter"]


class JournalEmitter:
    """Writes the mechanism's decisions to an incident's ordered event log."""

    def __init__(self, journal: Journal, *, policy: str) -> None:
        """Bind to one incident's journal and the composition being run.

        Args:
            journal: Where the events are stamped and written.
            policy: The named composition every event is attributed to.
        """
        self._journal = journal
        self._policy = policy

    def gate(self, verdict: GateVerdict) -> None:
        """Record the gate's decision.

        Args:
            verdict: What the gate decided.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="gate",
                fired=verdict.fires,
                # `gate_call_failed` and `gate_unparseable` are reasons in their own
                # right, so a fire on a broken audit stays distinguishable from a fire
                # on merit without a second field for it.
                reason=verdict.reason,
                confidence=verdict.confidence,
                weakness=verdict.weakness,
            )
        )

    def search(self, precedents: PrecedentSet) -> None:
        """Record a retrieval, its queries and what they returned.

        Args:
            precedents: The merged result.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="search",
                queries=[
                    {
                        "name": query.name,
                        "text": query.text,
                        "record_ids": list(precedents.per_query_ids.get(query.name, ())),
                    }
                    for query in precedents.queries
                ],
                precedent_ids=[p.record_id for p in precedents.precedents],
                withheld_count=precedents.withheld_count,
            )
        )

    def search_failed(self, error: Exception) -> None:
        """Record a retrieval that could not be performed.

        Args:
            error: Why it failed.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="search",
                fired=False,
                reason=f"search_failed:{str(error)[:300]}",
            )
        )

    def nomination(self, *, family: str | None, confidence: float | None, reason: str) -> None:
        """Record the nominator's verdict.

        Args:
            family: The nominated fault family, when one survived validation.
            confidence: The nominator's own confidence.
            reason: Why the nomination was accepted or rejected.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="nomination",
                family=family,
                confidence=confidence,
                reason=reason,
            )
        )

    def brief(self, brief: Brief) -> None:
        """Record what the second run was given.

        Args:
            brief: The assembled brief.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="brief",
                fired=not brief.cancels_run,
                reason=brief.reason or brief.kind,
                family=brief.family,
            )
        )

    def selection(
        self,
        selection: Selection,
        *,
        brief: Brief | None,
        precedents: PrecedentSet | None,
    ) -> None:
        """Record the blinded selection.

        Args:
            selection: What was chosen and why.
            brief: What the second run received.
            precedents: What retrieval returned, when it ran.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="selection",
                fired=True,
                reason=selection.fallback_reason or selection.reason,
                winner=selection.public_winner,
                confidence=selection.confidence,
                label_to_source=dict(selection.label_to_source),
                family=brief.family if brief is not None else None,
                precedent_ids=([p.record_id for p in precedents.precedents] if precedents else []),
                withheld_count=precedents.withheld_count if precedents else 0,
                selector_prompt=selection.prompt or None,
                selector_response=json.dumps(
                    {"raw": selection.raw_response, "causal_check": selection.causal_check},
                    sort_keys=True,
                ),
            )
        )

    def skipped(
        self,
        reason: str,
        *,
        verdict: GateVerdict | None,
        precedents: PrecedentSet | None,
    ) -> None:
        """Record a terminal path on which no second run happened.

        Args:
            reason: Which fail-safe fired.
            verdict: The gate's decision, when one was taken.
            precedents: What retrieval returned before the skip, when it ran.
        """
        self._journal.record(
            SecondPassEvent(
                policy=self._policy,
                step="skip",
                fired=False,
                reason=reason,
                confidence=verdict.confidence if verdict is not None else None,
                precedent_ids=([p.record_id for p in precedents.precedents] if precedents else []),
                withheld_count=precedents.withheld_count if precedents else 0,
            )
        )
