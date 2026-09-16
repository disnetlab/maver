"""Mapping retrieved precedents onto a closed vocabulary of fault families.

This is the nominated composition's mechanism, and the arm it was measured under is a
reproduced negative result
rather than a recommendation: guided minus resample is -1.7 composite points with a
confidence interval spanning [-11.1, +7.3]. It ships so the published number can be
reproduced, and the arm file records that it varies three axes at once.

Memory enters here as a candidate nominator and never as context: no precedent text
reaches the second run under this configuration, only a family label and a directive.
Note the asymmetry and read it precisely — only *retrieval* is blinded to the first
run's conclusion. The nominator prompt and the quality-gate prompt are both handed the
whole ``RootCause``.
"""

import logging
import re
from dataclasses import dataclass

from maver.contracts.artifacts import RootCause
from maver.llm.types import ModelRequest
from maver.prompts.loader import read_asset, render
from maver.secondpass.gate import candidate_json
from maver.secondpass.retrieve import render_precedents
from maver.secondpass.types import (
    PrecedentSet,
    SecondPassConfigError,
    SecondPassContext,
    clamped_confidence,
    first_json_object,
)

__all__ = [
    "ALLOWED_FAMILIES",
    "VERIFICATION_DIRECTIVE",
    "FamilyNominator",
    "Nomination",
    "parse_nomination",
    "validate_nomination",
]

_log = logging.getLogger(__name__)

# Closed family vocabulary: technical causal mechanisms an operator could find in
# production. A free-text family let the nominator absorb benchmark-meta framing from
# corpus text ("fault_injection") and pass it to run 2 — a hint that is degenerately
# true in the benchmark and would unfairly advantage the memory arm.
ALLOWED_FAMILIES = frozenset(
    {
        "probe_misconfiguration",
        "config_change",
        "network_connectivity",
        "dns_resolution",
        "resource_exhaustion",
        "application_bug",
        "message_queue",
        "storage_volume",
        "auth_permission",
        "scheduling_placement",
        "certificate_tls",
    }
)
_META = re.compile(
    r"(?i)(fault[\s_-]?inject|intentional|planted|benchmark|chaos[\s_-]?mesh|simulat)"
)

_FAMILY_CHARS = 80
_DIRECTIVE_CHARS = 500

VERIFICATION_DIRECTIVE = "Independently verify or falsify this attribution against live evidence."
"""The directive a same-family nomination falls back to when the veto is lifted."""


@dataclass(frozen=True, slots=True)
class Nomination:
    """One nominator verdict, parsed and bounded.

    ``differs`` is a boolean and stays one. It conflates three different situations —
    every precedent agrees, none is applicable, and the model was unconvinced — and 65
    of 70 skips in the campaign had two to five distinct families on the table even
    though the prompt rule is ANY rather than ALL. A three-valued verdict is the right
    shape and is also a different prompt, therefore a different arm, so it belongs to a
    successor rather than here.
    """

    differs: bool
    family: str
    directive: str
    confidence: float
    raw_response: str


def parse_nomination(raw: str) -> Nomination | None:
    """Read a nominator reply.

    ``differs`` must be a real boolean or the whole nomination is discarded, and a
    ``differs=true`` verdict with no family or no directive is discarded too: it names
    a redirect the second run cannot act on.

    Args:
        raw: The model's reply.

    Returns:
        The nomination, or None when the reply cannot be trusted.
    """
    payload = first_json_object(raw)
    if payload is None or not isinstance(payload.get("differs"), bool):
        return None
    family = " ".join(str(payload.get("family") or "").split())[:_FAMILY_CHARS]
    directive = " ".join(str(payload.get("directive") or "").split())[:_DIRECTIVE_CHARS]
    if payload["differs"] and not (family and directive):
        return None
    return Nomination(
        differs=payload["differs"],
        family=family,
        directive=directive,
        confidence=clamped_confidence(payload.get("confidence", 0.0)),
        raw_response=raw,
    )


def validate_nomination(nomination: Nomination, *, uses_family: bool = False) -> str | None:
    """Reject a nomination that escaped the vocabulary or carries benchmark framing.

    Historically these checks were skipped on ``differs=false``, because that verdict
    always meant a skip and the family was therefore never used. Once the forced
    delivery path and the quality gate began consuming the family on a false verdict,
    an empty or benchmark-meta family ("planted", "fault_injection") could reach the
    second run's assignment — exactly the leak the closed vocabulary exists to prevent,
    and one that leaks in the direction that flatters the memory arm.

    Args:
        nomination: What the nominator returned.
        uses_family: Whether the caller will build an assignment from a ``differs=false``
            nomination, which is what makes its family worth checking.

    Returns:
        A rejection reason, or None when the nomination is clean.
    """
    if not (nomination.differs or uses_family):
        return None
    family = nomination.family.strip().lower().replace(" ", "_").replace("-", "_")
    if not family:
        return "family_missing"
    if family not in ALLOWED_FAMILIES:
        return f"family_not_in_vocabulary:{family[:40]}"
    # Deliberately asymmetric: the vocabulary check reads the normalised family and the
    # meta check reads the raw one. Aligning them would change which nominations are
    # rejected, and the published cells were measured against this.
    if _META.search(nomination.family) or _META.search(nomination.directive):
        return "meta_language_in_directive"
    return None


class FamilyNominator:
    """Issues the single nomination call for a run."""

    def __init__(self, *, max_tokens: int = 900) -> None:
        """Configure the nominator.

        Args:
            max_tokens: Generation cap. Nine hundred is what the published cells ran.
        """
        self.max_tokens = max_tokens

    async def nominate(
        self, ctx: SecondPassContext, first: RootCause, precedents: PrecedentSet
    ) -> Nomination | None:
        """Ask whether any precedent points at a materially different fault family.

        Args:
            ctx: The pass's context.
            first: The candidate diagnosis, conclusion included.
            precedents: The retrieved records, rendered as bounded cards.

        Returns:
            The parsed nomination, or None when the call or its parse failed.
        """
        prompt = render(
            read_asset("secondpass/nominator_user.txt"),
            {
                "CANDIDATE_DIAGNOSIS": candidate_json(first),
                "RETRIEVED_PRECEDENTS": render_precedents(precedents.precedents),
            },
        )
        try:
            response = await ctx.completion.complete(
                ModelRequest(
                    system=read_asset("secondpass/nominator_system.txt"),
                    messages=({"role": "user", "content": prompt},),
                    sampling=ctx.sampling.model_copy(update={"max_tokens": self.max_tokens}),
                    # Every measured run made this call with the reasoning channel off.
                    suppress_reasoning=True,
                ),
                caller="secondpass.nominator",
            )
        except Exception as exc:
            _log.warning("the nominator call failed: %s", exc, exc_info=exc)
            return None
        return parse_nomination(str(response.text or ""))


def _assert_vocabulary_matches_prompt() -> None:
    """The prompt lists the vocabulary; the validator enforces it. Drift is silent.

    Raises:
        SecondPassConfigError: When the prompt omits a family the validator accepts.
    """
    text = read_asset("secondpass/nominator_system.txt")
    missing = sorted(family for family in ALLOWED_FAMILIES if family not in text)
    if missing:
        raise SecondPassConfigError(
            f"the nominator prompt does not list: {missing}. The prompt and "
            "ALLOWED_FAMILIES must agree, or the nominator is asked for a value the "
            "validator will reject."
        )


_assert_vocabulary_matches_prompt()
