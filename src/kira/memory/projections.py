"""Deterministic rendering of a distilled session into its two retrievable texts.

A completed incident reaches the corpus as exactly two documents. ``evidence_text``
carries the symptoms a later incident can match on; ``lesson_text`` carries what was
attempted, what was observed, and the test that separates this incident's cause from the
plausible wrong one. Raw transcripts never enter the corpus, and neither does the
incident summary.

Both templates are carried from the research prototype character for character. They are
at once what gets embedded and indexed and what a later incident's model reads, so a
reworded label is a different corpus, and every published retrieval number was measured
on these exact strings. The comments below record the incidents that shaped the rules;
they are evidence, not commentary. Do not reword either template without a replacement
measurement.

The module is pure: records in, strings out. It reads no settings, opens no connection,
and takes every choice as a parameter.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from kira.contracts import LearningResult

__all__ = ["FeedbackDepth", "MemoryProjections", "build_projections"]

FeedbackDepth = Literal["none", "scores", "details"]
"""How much external evaluation the session's feedback carried."""


@dataclass(frozen=True, slots=True)
class MemoryProjections:
    """The two indexable texts of one distilled session, plus what must stay out of them.

    Attributes:
        evidence_text: Identifiers and observations. The evidence-scope search document.
        lesson_text: What was attempted, what happened, and what discriminates the cause.
            The lesson-scope search document.
        corrected_cause: The authoritative cause, held out of both texts on purpose and
            carried here so a caller can store it as metadata.
        feedback_depth: Whether external feedback carried a correction (``details``),
            verdicts alone (``scores``), or was absent (``none``).
        authoritative_correction: Whether ``corrected_cause`` came from an external
            source rather than from the model's own account of the session.

    The last two are the working values behind the two above them and are returned
    rather than stored, because the stored record already answers the question they
    exist for: a correction is authoritative exactly when ``corrected_cause`` is set and
    ``evaluation_source`` is not empty. A grader that sends a reference cause for a cell
    it graded *correct* leaves no trace of having done so, which is the one distinction
    the record cannot recover and the one nothing reads: that correction is discarded
    here, so it never reaches the corpus either way.
    """

    evidence_text: str
    lesson_text: str
    corrected_cause: str
    feedback_depth: FeedbackDepth
    authoritative_correction: bool


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _feedback_components(feedback: Mapping[str, Any] | None) -> tuple[bool | None, bool | None]:
    if feedback is None:
        return None, None
    diagnosis = feedback.get("diagnosis")
    mitigation = feedback.get("mitigation")
    diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
    mitigation = mitigation if isinstance(mitigation, Mapping) else {}
    diag = diagnosis.get("success")
    mitig = mitigation.get("success")
    return (
        diag if isinstance(diag, bool) else None,
        mitig if isinstance(mitig, bool) else None,
    )


def _mask_identifiers(
    text: str, alert: Mapping[str, Any], resource: str, also: Sequence[str] = ()
) -> str:
    """Replace this incident's own identifiers with type placeholders.

    A lesson phrased around `user-service` in `social-network` shares almost no tokens
    with the same fault on `cartservice` in `astronomy-shop`, so lexical retrieval cannot
    transfer it — yet the mechanism is identical. Substituting placeholders keeps the
    sentence shape and leaves the mechanism vocabulary intact next to a constant slot.

    Screened over three snapshots against cross-application precedents (the only kind a
    fault-family train/test split leaves available), recall@5:

        vm4_strat      0.455 -> 0.475
        upperbound_n3  0.433 -> 0.448
        vm3_memv3      0.417 -> 0.500

    Same-family recall pays 0.005-0.027 for it, because that metric is dominated by
    same-application (often same-problem) matches which a split removes anyway.

    Deliberately uses only the row's own fields — no corpus-wide statistics — so it runs
    at write time. ``also`` is the exception, and it is supplied rather than known:
    the prototype carried the three benchmark namespaces as a constant here, which is a
    benchmark's vocabulary sitting in the layer that must not know one.
    """
    if not text:
        return text
    out = text
    res = " ".join(str(resource or "").split())
    if "/" in res:
        kind, _, name = res.partition("/")
        if len(name) > 2:
            out = re.sub(rf"\b{re.escape(name)}\b", "<resource>", out)
        out = re.sub(rf"\b{re.escape(res)}\b", f"{kind}/<resource>", out)
    elif len(res) > 2:
        out = re.sub(rf"\b{re.escape(res)}\b", "<resource>", out)
    namespace = " ".join(str(alert.get("namespace") or "").split())
    if len(namespace) > 2:
        out = re.sub(rf"\b{re.escape(namespace)}\b", "<namespace>", out)
    for other in also:
        cleaned = " ".join(str(other).split())
        if len(cleaned) > 2:
            out = re.sub(rf"\b{re.escape(cleaned)}\b", "<namespace>", out)
    return out


def build_projections(
    result: LearningResult,
    *,
    alert: Mapping[str, Any] | None,
    external_feedback: Mapping[str, Any] | None,
    evidence_observations: Sequence[str] | None = None,
    mask_identifiers: bool = False,
    mask_namespaces: Sequence[str] = (),
    affected_resource: str = "",
) -> MemoryProjections:
    """Render the two searchable documents without generative role ambiguity.

    External feedback controls causal authority. In particular, scores-only feedback can
    reject a hypothesis but can never manufacture its correction.

    The corrected cause is returned as its own field and never reaches ``lesson_text``.
    It is the most copyable sentence a search result can carry and it is authoritative
    about a *different* incident: a memory branch once submitted one verbatim as its own
    diagnosis for an unrelated fault. Removing the affordance rather than warning against
    it also keeps the correction out of the lesson embedding. The long-form record of
    that decision is at the point of exclusion below.

    Args:
        result: The distillation produced for one completed session.
        alert: The normalised alert, used for the identifier line and for masking.
        external_feedback: Verdicts from an external evaluator, when one exists. A
            ``ground_truth_root_cause`` in it makes the feedback authoritative.
        evidence_observations: Observations captured *before* feedback was visible. The
            caller supplies these; see the isolation comment below for why passing
            ``None`` is a standalone-rendering convenience rather than the write path.
        mask_identifiers: Whether to replace this incident's own resource and namespace
            names with type placeholders. Off in every published measurement; it changes
            every stored record, so it is a deliberate arm rather than a default.
        mask_namespaces: Other namespaces to replace as well, for a deployment running
            more than one application. Supplied rather than known, because which
            applications exist is not something the memory layer can be told at import.
        affected_resource: The resource this incident was localised to, used only when
            masking. Empty falls back to the alert's service.

    Returns:
        The rendered projections and the correction held out of them.
    """
    alert = alert or {}
    diagnosis_success, mitigation_success = _feedback_components(external_feedback)
    ground_truth = _clean((external_feedback or {}).get("ground_truth_root_cause"))
    feedback_depth: FeedbackDepth = (
        "details" if ground_truth else ("scores" if external_feedback else "none")
    )

    authoritative_correction = bool(ground_truth and diagnosis_success is False)
    if authoritative_correction:
        corrected_cause = ground_truth
    elif diagnosis_success is True:
        # A confirmed diagnosis has nothing to correct. Learning writes one anyway: across
        # VM3 and VM4, 18 rows on externally-confirmed sessions carried a `corrected_cause`,
        # most of it the reference text copied verbatim, and every one shipped under the
        # "(unverified)" label because authority is keyed on rejection. That put a true
        # statement behind a weak label, duplicated the confirmed-hypothesis line, and spent
        # retrieval volume the tail-trim arms showed is better left unspent. Only reachable
        # when an external verdict says the diagnosis was right, so the deployable
        # no-feedback path is unaffected.
        corrected_cause = ""
    elif feedback_depth == "scores":
        corrected_cause = ""  # scores contain no correction to learn
    else:
        corrected_cause = _clean(result.corrected_cause)

    identifiers = []
    for label, keys in (
        ("alert", ("alertname", "alert_name")),
        ("namespace", ("namespace",)),
        ("service", ("service", "app_name")),
    ):
        value = next((_clean(alert.get(k)) for k in keys if _clean(alert.get(k))), "")
        if value:
            identifiers.append(f"{label}={value}")
    # Post-feedback Learning sees the oracle correction, so its generated
    # `observations` cannot be trusted as an isolation boundary: a local model
    # may copy the correction into that field despite the schema/prompt.  The
    # caller therefore supplies observations captured before feedback (the RCA
    # trace evidence).  `None` preserves the pure renderer's standalone/test
    # behaviour; an explicitly empty list intentionally renders identifiers
    # only rather than falling back to feedback-contaminated model text.
    source_observations = (
        result.observations if evidence_observations is None else evidence_observations
    )
    observations = [_clean(item) for item in source_observations if _clean(item)]
    evidence_lines = [
        "Incident observations:",
        *(["Identifiers: " + "; ".join(identifiers)] if identifiers else []),
        *[f"- {item}" for item in observations],
    ]

    if diagnosis_success is True:
        diagnosis_label = "Confirmed diagnosis (external evaluation)"
    elif diagnosis_success is False:
        diagnosis_label = "Rejected hypothesis (external evaluation)"
    else:
        diagnosis_label = "Unverified hypothesis"

    if mitigation_success is True and diagnosis_success is False:
        action_label = (
            "Observed mitigation success, but not a causally validated fix because "
            "the diagnosis was rejected"
        )
    elif mitigation_success is True:
        action_label = "Successful remediation (external evaluation)"
    elif mitigation_success is False:
        action_label = "Failed remediation (external evaluation)"
    else:
        action_label = "Unverified remediation outcome"

    # Discriminator first, conclusions after and explicitly incident-bound. A reader
    # skimming for the most detailed, most copyable sentence used to find a conclusion
    # about a different incident and transplant it; leading with the test that
    # separates the two causes is the only part that legitimately transfers.
    discriminator = _clean(result.discriminator)
    lesson_lines = []
    if discriminator:
        lesson_lines.append(
            "DISCRIMINATOR — which observation separated the correct cause from the "
            f"plausible wrong one, and how to tell them apart here: {discriminator}"
        )
    # Emitted unconditionally, unlike the prototype, which emitted it only alongside a
    # discriminator. `discriminator` defaults to empty and the schema does not require
    # it, so a lesson written without one carried neither marker and read as a bare
    # conclusion about incidents in general — which is the exact failure the marker
    # exists to prevent, occurring in precisely the records most likely to mislead.
    lesson_lines.append("INCIDENT-BOUND (this incident only — not an answer for yours):")
    lesson_lines = [line for line in lesson_lines if line]
    lesson_lines += [
        f"{diagnosis_label}: {_clean(result.attempted_hypothesis)}",
        f"Attempted action: {_clean(result.attempted_action)}",
        f"{action_label}: {_clean(result.observed_outcome)}",
    ]
    # The corrected cause is deliberately NOT in the lesson projection. Labelled
    # "authoritative external detail" and carrying a machine-readable
    # `[fault_spec] component=…` header, it was the most copyable sentence in a
    # search result — and it is authoritative about a *different* incident. A memory
    # branch once submitted one verbatim as its own diagnosis for an unrelated fault.
    # Warnings did not stop that (the tool description already says precedent is not
    # ground truth), so the affordance is removed instead: the correction persists in
    # metadata and surfaces only through `open_memory`, which the agent must call
    # deliberately on one memory it has already judged relevant. Keeping it out of
    # lesson_text also keeps it out of the lesson embedding.
    constraints = [_clean(item) for item in result.case_negative_constraints if _clean(item)]
    if constraints:
        lesson_lines.append("Case-specific negative constraints:")
        lesson_lines.extend(f"- {item}" for item in constraints)

    evidence_text = "\n".join(evidence_lines).strip()
    lesson_text = "\n".join(lesson_lines).strip()
    if mask_identifiers:
        resource = _clean(affected_resource) or _clean(alert.get("service"))
        evidence_text = _mask_identifiers(evidence_text, alert, resource, mask_namespaces)
        lesson_text = _mask_identifiers(lesson_text, alert, resource, mask_namespaces)

    return MemoryProjections(
        evidence_text=evidence_text,
        lesson_text=lesson_text,
        corrected_cause=corrected_cause,
        feedback_depth=feedback_depth,
        authoritative_correction=authoritative_correction,
    )
