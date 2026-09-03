"""What the second run is told, which is the only axis the three configurations vary.

The resampling control tells it nothing, the quality-gated composition gives it bounded
precedent cards, the nominated one gives it a fault
family and a directive with no precedent text at all. Everything else about the two
diagnosis runs is held identical, which is what makes the paired contrasts readable.

A brief's text ends with exactly one newline, matching what the prototype assembled and
what the diagnosing role's assignment slot then wraps in a second one.
"""

from kira.contracts.artifacts import RootCause
from kira.prompts.loader import read_asset, render
from kira.secondpass.nominator import (
    VERIFICATION_DIRECTIVE,
    FamilyNominator,
    validate_nomination,
)
from kira.secondpass.retrieve import render_precedents
from kira.secondpass.types import (
    Brief,
    GateVerdict,
    PrecedentSet,
    SecondPassContext,
)

__all__ = ["FamilyAssignmentBrief", "NoBrief", "PrecedentTextBrief"]

_BLIND = Brief(text=None, kind="none")


class NoBrief:
    """Gives the second run nothing. The resampling control's briefer.

    The second run's opening message is then byte-identical to the first's apart from
    its own freshly collected snapshot, which is the whole point of the control: the
    contrast against a precedent-carrying brief measures the retrieved content and
    nothing else.
    """

    async def brief(
        self,
        ctx: SecondPassContext,
        first: RootCause,
        precedents: PrecedentSet | None,
        verdict: GateVerdict,
    ) -> Brief:
        """Return the blind brief.

        Args:
            ctx: Unused.
            first: Unused.
            precedents: Unused. No retrieval runs under this policy.
            verdict: Unused.

        Returns:
            A brief carrying no text.
        """
        del ctx, first, precedents, verdict
        return _BLIND


class PrecedentTextBrief:
    """Gives the second run bounded summary cards of the retrieved records.

    An empty result set does not necessarily cancel the run. Under the quality gate the
    audit — not retrieval — decided that a second look is warranted, so an empty set
    degrades the second run to a blind pass instead of cancelling it. That keeps this
    composition's dose identical to the resampling control's, so a paired
    context-versus-resample contrast
    measures the precedent text and nothing else, and the audited reason makes the
    degraded cells countable.
    """

    def __init__(self, *, on_empty: str = "blind_pass") -> None:
        """Configure the empty-set behaviour.

        Args:
            on_empty: ``blind_pass`` to degrade to a blind second run when the gate
                overruled, or ``cancel`` to always decline on an empty set.
        """
        self.on_empty = on_empty

    async def brief(
        self,
        ctx: SecondPassContext,
        first: RootCause,
        precedents: PrecedentSet | None,
        verdict: GateVerdict,
    ) -> Brief:
        """Render the precedent cards, or degrade.

        Args:
            ctx: Unused.
            first: Unused. The second run stays blind to it.
            precedents: What retrieval returned.
            verdict: The gate's decision, whose overrule permits a blind pass.

        Returns:
            The brief, which may be blind or may cancel the second run.
        """
        del ctx, first
        cards = precedents.precedents if precedents is not None else ()
        if not cards:
            if self.on_empty == "blind_pass" and verdict.overrules_brief:
                return Brief(text=None, kind="none", reason="no_precedents_blind_run")
            return Brief(text=None, kind="precedent_text", cancels_run=True, reason="no_precedents")
        text = render(
            read_asset("secondpass/context.txt"), {"PRECEDENTS": render_precedents(cards)}
        )
        return Brief(text=f"{text}\n", kind="precedent_text")


class FamilyAssignmentBrief:
    """Gives the second run a fault family and a verify-or-falsify directive.

    The nomination happens here rather than in :class:`~kira.secondpass.gate.NominatorGate`
    because nominating requires precedents and the gate runs before retrieval. One call
    per fired cell, therefore, and the gate that shares this nominator makes none.
    """

    def __init__(self, *, nominator: FamilyNominator, on_same_family: str = "cancel") -> None:
        """Configure the assignment.

        Args:
            nominator: The nominator to consult, shared with the policy's gate so the
                cap and the vocabulary guard are configured once.
            on_same_family: ``cancel`` to honour the nominator's veto, or ``verify`` to
                lift it and turn the second run into a blind verification pass of the
                shared attribution. The quality gate's overrule selects ``verify`` for
                its own cells, which is how the audit keeps the nominator's voice
                without its veto.
        """
        self.nominator = nominator
        self.on_same_family = on_same_family

    async def brief(
        self,
        ctx: SecondPassContext,
        first: RootCause,
        precedents: PrecedentSet | None,
        verdict: GateVerdict,
    ) -> Brief:
        """Nominate a family and render the assignment.

        Args:
            ctx: The pass's context.
            first: The candidate, which the nominator sees in full.
            precedents: What retrieval returned.
            verdict: The gate's decision, whose overrule lifts the veto.

        Returns:
            The brief, which cancels the second run on any nominator failure.
        """
        if precedents is None or not precedents.precedents:
            return Brief(
                text=None,
                kind="family_assignment",
                cancels_run=True,
                reason="no_precedents",
            )

        nomination = await self.nominator.nominate(ctx, first, precedents)
        veto_lifted = self.on_same_family == "verify" or verdict.overrules_brief
        if nomination is None:
            ctx.emit.nomination(family=None, confidence=None, reason="nominator_unparseable")
            return Brief(
                text=None,
                kind="family_assignment",
                cancels_run=True,
                reason="nominator_unparseable",
            )

        rejection = validate_nomination(nomination, uses_family=veto_lifted)
        ctx.emit.nomination(
            family=nomination.family or None,
            confidence=nomination.confidence,
            reason=rejection or ("differs" if nomination.differs else "same_family"),
        )
        if rejection:
            return Brief(
                text=None,
                kind="family_assignment",
                cancels_run=True,
                reason=f"nominator_rejected:{rejection}",
            )
        if not nomination.differs and not veto_lifted:
            return Brief(
                text=None,
                kind="family_assignment",
                cancels_run=True,
                reason="same_family",
            )

        directive = (
            nomination.directive
            if nomination.differs
            else (nomination.directive or VERIFICATION_DIRECTIVE)
        )
        text = render(
            read_asset("secondpass/assignment.txt"),
            {"FAMILY": nomination.family, "DIRECTIVE": directive},
        )
        return Brief(
            text=f"{text}\n",
            kind="family_assignment",
            family=nomination.family,
            directive=directive,
            reason="differs" if nomination.differs else "verify_shared_attribution",
        )
