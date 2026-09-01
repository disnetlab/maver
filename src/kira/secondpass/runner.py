"""One flow, five stages, no mode branch.

The word ``mode`` does not appear here. What separates MA+S from MA+R from Resample is
which objects are in the policy, and the flow is the manuscript's Algorithm 1 read one
line at a time: diagnose, gate, query, retrieve, brief, diagnose again, select.

Four properties of this shape are the point of the rewrite. There is one skip helper, so
the fail-safe direction cannot be got wrong at one of eleven return sites. There is one
emit surface, so an event cannot be written on one branch and forgotten on the other.
The result is complete on every path, so the roughly one incident in three that declines
is as analysable as the ones that fire. And every failure resolves to the first opinion:
no broken gate, nominator, selector or store can promote the second run.
"""

import logging

from kira.secondpass.select import build_evidence_bundle
from kira.secondpass.types import (
    Brief,
    DiagnosisRun,
    GateVerdict,
    PrecedentSet,
    PrecedentSourceError,
    RcaRunner,
    SecondPassContext,
    SecondPassPolicy,
    SecondPassResult,
)

__all__ = ["SecondPass"]

_log = logging.getLogger(__name__)


class SecondPass:
    """Runs one policy over one incident. The only place the five stages meet."""

    def __init__(self, policy: SecondPassPolicy, *, rca: RcaRunner) -> None:
        """Compose the mechanism.

        Args:
            policy: The named composition to run.
            rca: The diagnosis runner, injected by orchestration so this package
                imports no role. The prototype reached the role through a circular
                import in both directions; here a fake runner is all a test needs.
        """
        self._policy = policy
        self._rca = rca

    async def run(self, ctx: SecondPassContext) -> SecondPassResult:
        """Diagnose, decide whether to look again, and arbitrate if it did.

        Args:
            ctx: Everything the strategies may reach.

        Returns:
            The outcome, carrying both diagnosis runs when both happened and the gate's
            verdict and any precedents even when the pass declined.
        """
        policy = self._policy
        first = await self._rca(ctx.state, brief=None, retrieval=False, run_label="rca.first")
        if first.artifact is None or first.artifact.halt:
            # Nothing credible to second-guess. A failed or halted first opinion is
            # already the pipeline's failure mode, and arbitrating against it would only
            # launder a weak candidate through an unopposed selection.
            reason = "first_run_failed" if first.artifact is None else "first_run_halted"
            return self._skipped(ctx, first, reason)

        candidate = first.artifact
        verdict = await policy.gate.decide(ctx, candidate)
        ctx.emit.gate(verdict)
        if not verdict.fires:
            return self._skipped(ctx, first, verdict.reason, verdict=verdict)

        precedents: PrecedentSet | None = None
        if policy.query is not None and policy.retriever is not None:
            try:
                queries = await policy.query.build(ctx, candidate)
                precedents = await policy.retriever.fetch(ctx, queries)
            except PrecedentSourceError as exc:
                _log.warning("second-pass retrieval failed: %s", exc)
                ctx.emit.search_failed(exc)
                if policy.on_retrieval_error == "skip":
                    return self._skipped(ctx, first, "search_failed", verdict=verdict)
                precedents = PrecedentSet((), (), {}, 0)
            else:
                ctx.emit.search(precedents)

        brief = await policy.briefer.brief(ctx, candidate, precedents, verdict)
        ctx.emit.brief(brief)
        if brief.cancels_run:
            return self._skipped(
                ctx, first, brief.reason, verdict=verdict, precedents=precedents, brief=brief
            )

        # The original incoming state, not the first run's. That is the blindness
        # mechanism: the first run's messages, its diagnosis and its trace summaries are
        # simply not in it, so the second run cannot read them.
        second = await self._rca(
            ctx.state, brief=brief.text, retrieval=False, run_label="rca.second"
        )
        rival = second.artifact
        if rival is not None and rival.halt:
            rival = None

        evidence = build_evidence_bundle(ctx.alert, (first.messages, second.messages))
        selection = await policy.selector.select(ctx, candidate, rival, evidence)
        ctx.emit.selection(selection, brief=brief, precedents=precedents)

        chosen = selection.chosen if selection.chosen is not None else candidate
        return SecondPassResult(
            chosen=chosen,
            source=selection.source if selection.chosen is not None else "first",
            skipped=None,
            first=first,
            second=second,
            selection=selection,
            verdict=verdict,
            brief=brief,
            precedents=precedents,
        )

    def _skipped(
        self,
        ctx: SecondPassContext,
        first: DiagnosisRun,
        reason: str,
        *,
        verdict: GateVerdict | None = None,
        precedents: PrecedentSet | None = None,
        brief: Brief | None = None,
    ) -> SecondPassResult:
        """Return the first opinion unopposed, and say why.

        The only place a pass ends without a second run, so the fail-safe direction is
        a property of the shape rather than of eleven separate return statements.
        """
        ctx.emit.skipped(reason, verdict=verdict, precedents=precedents)
        return SecondPassResult(
            chosen=first.artifact,
            source="first" if first.artifact is not None else "none",
            skipped=reason,
            first=first,
            verdict=verdict,
            brief=brief,
            precedents=precedents,
        )
