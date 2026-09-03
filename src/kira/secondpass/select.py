"""Choosing between two diagnoses under shuffled labels.

One bounded model call over two typed candidates and a bundle of live tool results.
Where precedent text is delivered it is the only safety mechanism standing between a
wrong precedent and the
pipeline's answer, so every failure mode here resolves to the first opinion.

Thinking must be off for this call, and the reason is a measured failure rather than a
preference. The selector has no tools, so tracing a mechanism from cause to symptom has
to happen in context — but it must happen in ``content``, not in the reasoning channel.
With thinking on, the model spent the budget reasoning and returned empty content: every
selector call in one rerun parsed to nothing and fell back to the first answer. In Kira
that is a property of the auxiliary sampling set the context carries rather than a
per-call flag.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from kira.contracts.artifacts import RootCause
from kira.llm.types import ModelRequest
from kira.prompts.loader import read_asset, render
from kira.secondpass.gate import candidate_json
from kira.secondpass.types import (
    SecondPassContext,
    Selection,
    clamped_confidence,
    first_json_object,
)
from kira.tools.allowlists import RETRIEVAL_TOOLS

__all__ = ["BlindSelector", "blinded_labels", "build_evidence_bundle"]

_log = logging.getLogger(__name__)

_NON_EVIDENCE_TOOLS = RETRIEVAL_TOOLS | {"submit_root_cause"}
"""What never reaches the selector: precedent text, and the submission itself.

The selector judges live evidence. A memory result in the bundle would let a candidate
be supported by the very precedent that produced it.
"""

_RESULT_CHARS = 1800
_MAX_RESULTS = 12
_REASON_CHARS = 300
_CAUSAL_CHECK_CHARS = 600


def blinded_labels(session_id: str) -> dict[str, str]:
    """Assign the two candidates to A and B deterministically in the session id.

    Stable across retries and across Python processes, so a rerun of the same cell
    produces the same assignment and the map is reproducible from the session id alone
    months later. ``hashlib`` rather than ``random``: there is no seed to set and no
    global state to pollute. The source-to-label map is absent from the prompt and is
    persisted only after the decision.

    Args:
        session_id: The incident.

    Returns:
        Label to source, one of ``first`` or ``second`` per label.
    """
    second_first = hashlib.sha256(session_id.encode("utf-8")).digest()[0] % 2 == 0
    return {"A": "second", "B": "first"} if second_first else {"A": "first", "B": "second"}


def _tool_names(transcript: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in transcript:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            names[str(call.get("id") or "")] = str(function.get("name") or "")
    return names


def build_evidence_bundle(
    alert: Mapping[str, Any],
    transcripts: Sequence[Sequence[Mapping[str, Any]]],
) -> str:
    """Bundle the live tool results both diagnosis runs gathered.

    Args:
        alert: The incident's alert payload.
        transcripts: The two runs' message lists, in run order.

    Returns:
        The incident header, and the deduplicated tail of live results beneath it.
    """
    # Underscore-prefixed keys carry the benchmark's own fault label. Stripping them is
    # the problem-id blinding and is not optional; the context that reaches here is
    # already stripped, and this is the one place the alert becomes a prompt.
    visible = {k: v for k, v in alert.items() if not k.startswith("_")}
    header = render(
        read_asset("secondpass/incident_context.txt"),
        {"ALERT_JSON": json.dumps(visible, indent=2, sort_keys=True)},
    )

    kept: list[str] = []
    seen: set[str] = set()
    for transcript in transcripts:
        names = _tool_names(transcript)
        for message in transcript:
            if message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or names.get(call_id) or "")
            if not name or name in _NON_EVIDENCE_TOOLS:
                continue
            body = " ".join(str(message.get("content") or "").split())
            if not body:
                continue
            rendered = f"[{name}] {body[:_RESULT_CHARS]}"
            if rendered in seen:
                continue
            seen.add(rendered)
            kept.append(rendered)
    if not kept:
        return header
    # Prefer the most recent observations. Diagnosis typically starts broad and
    # localizes later, so the tail is more likely to contain the decisive
    # resource-level fact.
    kept = kept[-_MAX_RESULTS:]
    return header + "\n\nLive evidence gathered before and after the fork:\n" + "\n".join(kept)


def _fallback(
    first: RootCause | None,
    *,
    reason: str,
    public_winner: str,
    fallback_reason: str,
    prompt: str = "",
    raw_response: str = "",
    label_to_source: Mapping[str, str] | None = None,
    confidence: float = 0.0,
) -> Selection:
    """Resolve to the first opinion. Every failure on this path lands here."""
    return Selection(
        chosen=first,
        source="first" if first is not None else "none",
        public_winner=public_winner,
        confidence=confidence,
        reason=reason,
        causal_check="",
        label_to_source=dict(label_to_source or {}),
        fallback_reason=fallback_reason,
        prompt=prompt,
        raw_response=raw_response,
    )


class BlindSelector:
    """Arbitrates between two candidates without being told where either came from."""

    def __init__(self, *, max_tokens: int = 1600, timeout_s: int = 180) -> None:
        """Configure the selector.

        Args:
            max_tokens: Generation cap. Deliberately well under the run-wide cap: an
                unbounded reasoning budget under a frequency penalty degenerates rather
                than deciding better.
            timeout_s: Wall-clock bound. A timeout falls back to the first answer.
        """
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    async def select(
        self,
        ctx: SecondPassContext,
        first: RootCause | None,
        second: RootCause | None,
        evidence: str,
    ) -> Selection:
        """Choose the candidate better supported by the live evidence.

        Args:
            ctx: The pass's context.
            first: The first opinion.
            second: The second opinion, or None when its run produced nothing.
            evidence: The bundled live tool results.

        Returns:
            The decision. Any ambiguity or failure resolves to the first opinion.
        """
        if first is None and second is None:
            return _fallback(
                None,
                reason="both branches failed",
                public_winner="none",
                fallback_reason="both_branches_failed",
            )
        if second is None:
            return _fallback(
                first,
                reason="second branch failed",
                public_winner="unopposed",
                fallback_reason="second_branch_failed",
            )
        if first is None:
            # Unreachable from the second pass, which returns before a second run exists
            # when the first produced nothing. Kept because this is a general
            # two-candidate arbiter and a caller without that guarantee needs it.
            return Selection(
                chosen=second,
                source="second",
                public_winner="unopposed",
                confidence=0.0,
                reason="first branch failed",
                causal_check="",
                label_to_source={},
                fallback_reason="first_branch_failed",
                prompt="",
                raw_response="",
            )

        labels = blinded_labels(ctx.session_id)
        candidates = {
            label: (first if source == "first" else second) for label, source in labels.items()
        }
        prompt = render(
            read_asset("secondpass/selector_user.txt"),
            {
                "LIVE_EVIDENCE": evidence,
                "CANDIDATE_A": candidate_json(candidates["A"]),
                "CANDIDATE_B": candidate_json(candidates["B"]),
            },
        )
        try:
            response = await asyncio.wait_for(
                ctx.completion.complete(
                    ModelRequest(
                        system=read_asset("secondpass/selector_system.txt"),
                        messages=({"role": "user", "content": prompt},),
                        sampling=ctx.sampling.model_copy(update={"max_tokens": self.max_tokens}),
                    ),
                    caller="secondpass.selector",
                ),
                timeout=self.timeout_s,
            )
        except TimeoutError:
            return _fallback(
                first,
                reason="selector timed out",
                public_winner="fallback",
                fallback_reason="selector_timeout",
                prompt=prompt,
                label_to_source=labels,
            )
        except Exception as exc:
            _log.warning("the selector call failed: %s", exc, exc_info=exc)
            return _fallback(
                first,
                reason=f"selector failed: {type(exc).__name__}",
                public_winner="fallback",
                fallback_reason="selector_exception",
                prompt=prompt,
                label_to_source=labels,
            )

        raw = str(response.text or "")
        parsed = _parse(raw)
        if parsed is None:
            return _fallback(
                first,
                reason="invalid selector output",
                public_winner="fallback",
                fallback_reason="selector_parse_failure",
                prompt=prompt,
                raw_response=raw,
                label_to_source=labels,
            )
        winner, confidence, reason, causal_check = parsed
        if winner == "tie":
            return Selection(
                chosen=first,
                source="first",
                public_winner="tie",
                confidence=confidence,
                reason=reason,
                causal_check=causal_check,
                label_to_source=labels,
                fallback_reason="selector_tie",
                prompt=prompt,
                raw_response=raw,
            )
        source = labels[winner]
        return Selection(
            chosen=candidates[winner],
            source="first" if source == "first" else "second",
            public_winner=winner,
            confidence=confidence,
            reason=reason,
            causal_check=causal_check,
            label_to_source=labels,
            fallback_reason=None,
            prompt=prompt,
            raw_response=raw,
        )


def _parse(raw: str) -> tuple[str, float, str, str] | None:
    """Read a selector reply, or reject it whole."""
    payload = first_json_object(raw)
    if payload is None:
        return None
    winner = str(payload.get("winner") or "").strip().upper()
    if winner not in {"A", "B", "TIE"}:
        return None
    reason = " ".join(str(payload.get("reason") or "").split())[:_REASON_CHARS]
    # Persisted deliberately. The model keeps most of its reasoning in a channel that is
    # not stored, so an unwritten causal check is an unreviewable one.
    causal_check = " ".join(str(payload.get("causal_check") or "").split())[:_CAUSAL_CHECK_CHARS]
    return (
        winner.lower() if winner == "TIE" else winner,
        clamped_confidence(payload.get("confidence", 0.0)),
        reason,
        causal_check,
    )
