"""Deterministic evidence/lesson projection for episodic memory."""

import re
from dataclasses import dataclass
from typing import Any

from agent.config import memory_mask_identifiers
from agent.state import LearningResult


@dataclass(frozen=True)
class MemoryProjections:
    evidence_text: str
    lesson_text: str
    corrected_cause: str
    feedback_depth: str
    authoritative_correction: bool


# Namespaces of the benchmark applications. A memory written about one app is only
# reusable on another if the app's own names do not dominate its tokens.
_APP_NAMESPACES = (
    "astronomy-shop", "hotel-reservation", "social-network",
    "astronomy_shop", "hotel_reservation", "social_network",
)


def _mask_identifiers(text: str, alert: dict[str, Any], resource: str) -> str:
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
    at write time.
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
    for ns in _APP_NAMESPACES:
        out = re.sub(rf"\b{re.escape(ns)}\b", "<namespace>", out)
    return out


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _feedback_components(feedback: dict | None) -> tuple[bool | None, bool | None]:
    if not isinstance(feedback, dict):
        return None, None
    diagnosis = feedback.get("diagnosis")
    mitigation = feedback.get("mitigation")
    diagnosis = diagnosis if isinstance(diagnosis, dict) else {}
    mitigation = mitigation if isinstance(mitigation, dict) else {}
    diag = diagnosis.get("success")
    mitig = mitigation.get("success")
    return (
        diag if isinstance(diag, bool) else None,
        mitig if isinstance(mitig, bool) else None,
    )


def build_memory_projections(
    result: LearningResult,
    *,
    alert: dict[str, Any] | None,
    external_feedback: dict | None,
    evidence_observations: list[str] | None = None,
) -> MemoryProjections:
    """Render the two searchable documents without generative role ambiguity.

    External feedback controls causal authority. In particular, scores-only
    feedback can reject a hypothesis but can never manufacture its correction.
    """
    alert = alert or {}
    feedback = external_feedback if isinstance(external_feedback, dict) else None
    diagnosis_success, mitigation_success = _feedback_components(feedback)
    ground_truth = _clean((feedback or {}).get("ground_truth_root_cause"))
    feedback_depth = "details" if ground_truth else ("scores" if feedback else "none")

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
        result.observations
        if evidence_observations is None
        else evidence_observations
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
    discriminator = _clean(getattr(result, "discriminator", ""))
    lesson_lines = []
    if discriminator:
        lesson_lines.append(
            "DISCRIMINATOR — which observation separated the correct cause from the "
            f"plausible wrong one, and how to tell them apart here: {discriminator}"
        )
    lesson_lines.append(
        "INCIDENT-BOUND (this incident only — not an answer for yours):"
        if discriminator
        else ""
    )
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
    constraints = [
        _clean(item) for item in result.case_negative_constraints if _clean(item)
    ]
    if constraints:
        lesson_lines.append("Case-specific negative constraints:")
        lesson_lines.extend(f"- {item}" for item in constraints)

    evidence_text = "\n".join(evidence_lines).strip()
    lesson_text = "\n".join(lesson_lines).strip()
    if memory_mask_identifiers():
        resource = _clean(getattr(result, "affected_resource", "")) or _clean(
            alert.get("service")
        )
        evidence_text = _mask_identifiers(evidence_text, alert, resource)
        lesson_text = _mask_identifiers(lesson_text, alert, resource)

    return MemoryProjections(
        evidence_text=evidence_text,
        lesson_text=lesson_text,
        corrected_cause=corrected_cause,
        feedback_depth=feedback_depth,
        authoritative_correction=authoritative_correction,
    )
