"""The incident state carried through the graph.

This is the only mutable object shared between roles. It holds the typed artifacts
each role produced, the retry budgets, and the per-role conversation histories kept
for audit. Roles read the artifacts of their predecessors, never their transcripts.

Message histories and trace summaries are keyed by role name rather than held in
fixed fields, so registering a new role does not require changing this schema.
"""

from typing import Any, Literal, TypedDict

from kira.contracts.artifacts import (
    ExecutionResult,
    Plan,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    VerifyResult,
)

__all__ = ["IncidentState", "TargetStage"]

TargetStage = Literal["full", "diagnose"]
"""How far the pipeline should run.

``full`` runs diagnosis through remediation. ``diagnose`` stops after the diagnosis,
for environments that grade detection or localisation as a separate task.
"""


class IncidentState(TypedDict, total=False):
    """State threaded through the incident graph.

    Every key is optional because the graph populates them progressively and each
    node returns the merged state.
    """

    # Inputs.
    session_id: str
    alert: dict[str, Any]
    target_stage: TargetStage

    # Filesystem paths injected by the runner. The incident directory is writable for
    # this run only; shared memory is reachable through the memory tools and never by
    # direct file access.
    incident_dir: str
    shared_dir: str

    # Per-role conversation histories, retained for audit and never passed between
    # roles. Keyed by role name.
    messages: dict[str, list[dict[str, Any]]]

    # Bounded per-role process summaries, consumed by the offline learning pass.
    trace_summaries: list[dict[str, Any]]

    # Ordered identifiers of the persisted role runs along this pipeline path.
    run_ids: list[str]

    # The typed artifacts. The single-agent configuration reuses the diagnosis and
    # result slots with its own unstructured models.
    root_cause: RootCause | SingleDiagnosis | None
    plan: Plan | None
    execution_result: ExecutionResult | SingleResult | None
    verify_result: VerifyResult | None

    # Remediation retry accounting. Three back-edges draw on one budget so that
    # replan and re-diagnose cycles cannot compound into independent budgets:
    #
    #   Execution -> Planning     the diagnosis holds but the plan could not be
    #                             executed; prior_plans accumulates the failed plan
    #                             and its execution note so the revision differs.
    #   Verify    -> Planning     the plan applied but was insufficient for the same
    #                             cause.
    #   Verify    -> RCA          the change applied but the cause was wrong;
    #                             prior_attempts accumulates the earlier diagnosis
    #                             and remediation so the re-diagnosis digs deeper.
    #
    # remediation_retries counts every back-edge taken and is the budget gate;
    # reassessment_count tracks the re-diagnosis subset, for telemetry only.
    remediation_retries: int
    reassessment_count: int

    # Re-checks after an inconclusive verification verdict. Held separately from the
    # remediation budget on purpose: verification is read-only, so a second look
    # costs a few telemetry reads and mutates nothing, whereas closing on "could not
    # tell" ends the incident with recovery unconfirmed.
    verify_retries: int

    prior_attempts: list[dict[str, Any]]
    prior_plans: list[dict[str, Any]]

    # Terminal outputs surfaced to the caller.
    final_answer: str | None
    error: str | None
