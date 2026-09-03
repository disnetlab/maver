"""Deciding whether a first opinion is worth second-guessing.

Three implementations behind one protocol, which is what replaces the prototype's five
interleaved decision points. Two of them make no model call at all.

The quality audit runs *before* retrieval and sees only the candidate's own
``RootCause`` — no precedents, no cluster, no transcript. That is deliberate: the
fire/skip signal is a property of the candidate, so paying for retrieval before deciding
is wasted on the cells that skip, and the gate declines on roughly one incident in
three. It is why the mechanism costs about 2% more model calls and 14% more wall clock
per incident rather than doubling both.
"""

import json
import logging
from typing import Literal

from kira.contracts.artifacts import RootCause
from kira.llm.types import ModelRequest
from kira.prompts.loader import read_asset, render
from kira.secondpass.types import (
    GateVerdict,
    SecondPassContext,
    clamped_confidence,
    first_json_object,
)

__all__ = ["AlwaysGate", "NominatorGate", "QualityAuditGate", "candidate_json"]

_log = logging.getLogger(__name__)

_WEAKNESS_CHARS = 300


def candidate_json(candidate: RootCause) -> str:
    """Render a diagnosis as the JSON payload a helper prompt receives.

    Sorted and indented, so the same diagnosis produces the same prompt bytes on a
    replay months later.

    Args:
        candidate: The diagnosis.

    Returns:
        The rendered JSON.
    """
    return json.dumps(candidate.model_dump(mode="json"), sort_keys=True, indent=2)


class AlwaysGate:
    """Fires unconditionally and makes no model call. The resampling control's gate.

    Also the honest name for what the prototype did when the resampling mode was
    combined with the nominator gate: the quality-gate branch required the quality
    setting and no nominator branch existed for resampling, so control fell straight
    through to the second run.
    """

    name = "always"

    async def decide(self, ctx: SecondPassContext, first: RootCause) -> GateVerdict:
        """Fire.

        Args:
            ctx: Unused.
            first: Unused.

        Returns:
            A firing verdict.
        """
        del ctx, first
        return GateVerdict(fires=True, reason="always")


class NominatorGate:
    """Defers to the nomination, which cannot run until retrieval has.

    You must retrieve to nominate, and the flow retrieves after the gate, so this gate
    fires to let retrieval happen and the nominator's verdict lands one stage later as
    the assignment brief's veto. That keeps the single flow honest: the same
    ``differs=false`` that skipped the pass in the prototype cancels the brief here,
    with the same audited reason.
    """

    name = "nominator"

    async def decide(self, ctx: SecondPassContext, first: RootCause) -> GateVerdict:
        """Fire, so that retrieval and the nomination can run.

        Args:
            ctx: Unused.
            first: Unused.

        Returns:
            A firing verdict.
        """
        del ctx, first
        return GateVerdict(fires=True, reason="nominator")


class QualityAuditGate:
    """One bounded audit of the candidate against its own cited evidence.

    The audit reads four things and nothing else: whether the stated cause could
    mechanically produce the listed evidence, whether each evidence line is an
    observation rather than an inference, whether the target is the failing resource
    rather than a symptom, and whether the summary hedges between explanations.
    """

    name = "quality"

    def __init__(self, *, max_tokens: int, on_unreadable: Literal["fire", "defer"]) -> None:
        """Configure the audit.

        Args:
            max_tokens: Generation cap. Six hundred is what every published cell ran.
            on_unreadable: What an unreadable audit means. ``fire`` is for a policy with
                no nominator behind it, where nothing else can stop the pass; ``defer``
                leaves the decision to the nominator. Under both the pass proceeds,
                because an unreadable audit must never suppress a second opinion.
        """
        self.max_tokens = max_tokens
        self.on_unreadable = on_unreadable

    async def decide(self, ctx: SecondPassContext, first: RootCause) -> GateVerdict:
        """Audit the candidate.

        Args:
            ctx: The pass's context.
            first: The diagnosis to audit.

        Returns:
            Whether a second look is warranted, and the weakness the audit named.
        """
        prompt = render(
            read_asset("secondpass/quality_gate_user.txt"),
            {"CANDIDATE_DIAGNOSIS": candidate_json(first)},
        )
        try:
            response = await ctx.completion.complete(
                ModelRequest(
                    system=read_asset("secondpass/quality_gate_system.txt"),
                    messages=({"role": "user", "content": prompt},),
                    sampling=ctx.sampling.model_copy(update={"max_tokens": self.max_tokens}),
                ),
                caller="secondpass.gate",
            )
        except Exception as exc:
            _log.warning("the quality audit call failed: %s", exc, exc_info=exc)
            return self._unreadable("gate_call_failed")

        payload = first_json_object(response.text or "")
        if payload is None or not isinstance(payload.get("needs_second_opinion"), bool):
            # A bool or nothing. A truthy string does not become a verdict: coercing
            # "yes" would let a model that ignored the schema decide the arm's dose.
            return self._unreadable("gate_unparseable")

        weakness = str(payload.get("weakness") or "")[:_WEAKNESS_CHARS]
        confidence = clamped_confidence(payload.get("confidence", 0.0))
        if not payload["needs_second_opinion"]:
            return GateVerdict(False, "quality_gate_pass", weakness, confidence)
        return GateVerdict(True, "quality_gate_fire", weakness, confidence, overrules_brief=True)

    def _unreadable(self, reason: str) -> GateVerdict:
        """Resolve an audit that could not be read, without suppressing the pass.

        The overrule travels with ``fire`` because a policy with no nominator behind it
        has nothing else standing between an unreadable audit and a cancelled pass: on
        an empty result set it would otherwise decline, which is the suppression this
        path exists to prevent. It costs no parity, because the measured campaign had
        no cell with an empty result set.
        """
        return GateVerdict(
            fires=True,
            reason=reason,
            parsed=False,
            overrules_brief=self.on_unreadable == "fire",
        )
