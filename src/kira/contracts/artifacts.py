"""The typed artifacts roles exchange, and the submission schemas they validate.

These models are the frozen application binary interface of the system. Roles hand
each other these records and never transcripts, so a prompt can change while the
handoff stays fixed.

The field descriptions are not documentation. They are shipped to the model as the
JSON Schema of its submission tool, so wording changes are behaviour changes. They
are carried over verbatim from the research prototype, together with the validators,
because each validator encodes a failure that was measured to destroy graded work.

The same is true of a class docstring, which Pydantic turns into the schema's
top-level description. That is why these models derive from ``SubmissionModel``, which
strips it: documentation and instruction are separate channels, and an edit to one
must not silently rewrite the other. Model-facing text at the class level is declared
explicitly in ``schema_description``. Both are pinned by
``tests/unit/test_submission_schema_fidelity.py`` against what the prototype shipped.

Changing a field here invalidates published comparisons between experiment
configurations, which are only single-variable while the submission schema is held
constant. See STYLE_GUIDE.md, "Typed artifacts are frozen".
"""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, GetJsonSchemaHandler, field_validator, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from kira.contracts.parsing import (
    cites_request_path,
    decode_container,
    is_blank_reason,
    normalize_nullish,
)

__all__ = [
    "ExecutionResult",
    "LearningPacket",
    "LearningResult",
    "Plan",
    "PlanStep",
    "RoleTraceSummary",
    "RootCause",
    "SingleDiagnosis",
    "SingleResult",
    "StepResult",
    "SubmissionModel",
    "VerifyResult",
]


class SubmissionModel(BaseModel):
    """Base for a model whose schema is shipped to the language model.

    A class docstring must not reach that schema. Pydantic derives the schema's
    top-level ``description`` from ``__doc__``, so an ordinary documentation edit
    silently rewrites part of the instructions every role receives, on a contract that
    is otherwise held byte-compatible. That is not hypothetical: rewriting these
    classes for readability added a description to six submission schemas that had
    none and reworded three that had one, and nothing failed.

    Model-facing text is therefore declared explicitly in ``schema_description``, and
    docstrings are free to say whatever a reader needs.
    """

    schema_description: ClassVar[str | None] = None

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Emit only the description this class declares, never its docstring.

        Args:
            core_schema: The validated core schema.
            handler: The next schema generator.

        Returns:
            The JSON schema, with the docstring-derived description removed.
        """
        schema = handler(core_schema)
        schema.pop("description", None)
        if cls.schema_description is not None:
            schema["description"] = cls.schema_description
        return schema


class RootCause(SubmissionModel):
    """The diagnosis emitted by the root-cause analysis role."""

    fault_type: str = Field(
        description=(
            "Short fault category label. Examples: resource_stress, "
            "pod_crash, network_latency, misconfiguration, io_failure, "
            "node_failure."
        ),
    )
    affected_resource: str = Field(
        description=(
            "The object whose state or configuration changed to cause the incident, "
            "not the workload merely showing the symptom. Rule of thumb: if you are "
            "naming a pod that is crashing, restarting, or failing to connect, you "
            "are probably wrong — the crash is the symptom. Name what the pod "
            "depends on that changed (Service, Secret, ConfigMap, backing dependency, "
            "or deployment spec that was mutated). When a workload is unhealthy "
            "because a dependency rejected it (auth or permission denied, connection "
            "refused, empty endpoints, missing key), name the dependency object that "
            "changed, not the consumer. Use the most specific stable identifier: "
            "namespace, kind, and name."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Diagnosis confidence from 0.0 to 1.0 based on evidence strength and conflicts."
        ),
    )
    evidence: list[str] = Field(
        description=(
            "Short list of concrete observations supporting the diagnosis. "
            "Include command pattern, observed values, resource, and time window. "
            "Prefer numeric metrics, Kubernetes reasons, and concrete log/trace "
            "error classes."
        ),
    )
    summary: str = Field(description="One to three sentence diagnosis summary.")
    halt: bool = Field(
        default=False,
        description=(
            "Set to true ONLY when no truthful root cause can be derived from "
            "the available evidence and the agent wants to stop the incident "
            "flow. The normal path is false — Planning then designs the fix and "
            "Execution applies it. Never halt for safety, caution, or because "
            "remediation 'is not the RCA role'."
        ),
    )
    halt_reason: str | None = Field(
        default=None,
        description=(
            "Required when halt is true: explain why no diagnosis is reachable. "
            "Ignored when halt is false."
        ),
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _decode_evidence(cls, value: object) -> object:
        return decode_container(value)

    @field_validator("halt_reason", mode="before")
    @classmethod
    def _clean_halt_reason(cls, value: object) -> object:
        return normalize_nullish(value)


class PlanStep(SubmissionModel):
    """One remediation step in a plan."""

    action: str = Field(description="Declarative remediation action name.")
    target: str = Field(description="Kubernetes resource path or identifier.")
    params: dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters.")
    justification: str = Field(description="Why this step is appropriate.")

    @field_validator("params", mode="before")
    @classmethod
    def _decode_params(cls, value: object) -> object:
        return decode_container(value)


class Plan(SubmissionModel):
    """The typed remediation plan emitted by the planning role."""

    steps: list[PlanStep]
    expected_outcome: str = Field(description="What success should look like.")
    rollback: str | None = Field(default=None, description="Plain-language rollback guidance.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence (0.0–1.0) that the plan will achieve expected_outcome.",
    )
    halt: bool = Field(
        default=False,
        description=(
            "Set to true ONLY when no remediation step can be expressed at all "
            "and the agent wants to stop the incident flow. The normal path is "
            "false — Execution then applies the plan. Never halt for safety, "
            "risk, or low confidence."
        ),
    )
    halt_reason: str | None = Field(
        default=None,
        description=(
            "Required when halt is true: explain why no plan is possible. "
            "Ignored when halt is false."
        ),
    )

    @field_validator("steps", mode="before")
    @classmethod
    def _decode_steps(cls, value: object) -> object:
        return decode_container(value)

    @field_validator("halt_reason", mode="before")
    @classmethod
    def _clean_halt_reason(cls, value: object) -> object:
        return normalize_nullish(value)


class StepResult(SubmissionModel):
    """The observed outcome of applying one plan step."""

    step_index: int = Field(description="Zero-based index of the plan step this entry reports.")
    succeeded: bool = Field(description="Whether this step was applied and verified.")
    observation: str = Field(
        description=(
            "Command/check run, before value or precondition, mutation/denial "
            "result, after value, and verification verdict; for an abort, the "
            "drift, denial, or missing-resource reason."
        ),
    )


class ExecutionResult(SubmissionModel):
    """The resource-level outcome reported by the execution role."""

    succeeded: bool = Field(
        description=(
            "Whether the plan was APPLIED as intended at the resource level — the "
            "mutation took effect and the rollout completed (e.g. the Deployment "
            "generation advanced and Ready replicas match the spec). This is NOT a "
            "claim that the incident is resolved end to end; the Verify step decides "
            "system-level recovery. Set false only when the change could not be "
            "carried out (a command errored, the target was not shaped as the plan "
            "assumed, the write was denied, or the applied change did not take)."
        ),
    )
    steps: list[StepResult] = Field(
        description=(
            "Required, at least one entry — one StepResult per attempted step. Each "
            "observation names the command/check, the before value or precondition, "
            "the mutation or denial result, the after value, and the verification "
            "verdict; for an abort, the drift, denial, or missing-resource reason."
        ),
    )
    final_state_summary: str = Field(
        description=(
            "Required, non-empty. What was changed and what the resource-level "
            "verification showed (the patched field, rollout/Ready state), and why "
            "the apply is reported as success or failure. For an abort, state the "
            "drift, denial, or missing-resource reason. Do not assert end-to-end "
            "recovery — that is the Verify step's job."
        ),
    )
    drift_detected: bool = False
    needs_replan: bool = Field(
        default=False,
        description=(
            "Set true when the diagnosis still looks right but the PLAN could not be "
            "carried out as written and a different plan against the SAME root cause "
            "could work — a planned command errored or could not run, the target was "
            "not shaped as the plan assumed, a step was ambiguous/underspecified, or "
            "the applied change did not take and a different approach is needed. When "
            "true, also set succeeded=false and fill replan_reason — this routes the "
            "incident back to Planning for a revised plan. Use this instead of failing "
            "outright when the plan, not the diagnosis, is what went wrong. Leave false "
            "for a clean apply or a terminal failure (write denied by policy, unsafe "
            "drift, nothing actionable remains)."
        ),
    )
    replan_reason: str | None = Field(
        default=None,
        description=(
            "Required when needs_replan=true: name what in the plan could not be "
            "executed and what a revised plan should do differently, so Planning does "
            "not re-emit the same infeasible plan."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence (0.0–1.0) that the executed steps achieved the intended "
            "cluster state. Lower this when post-step observations partially "
            "contradict the expected outcome."
        ),
    )

    @field_validator("steps", mode="before")
    @classmethod
    def _decode_steps(cls, value: object) -> object:
        return decode_container(value)

    @model_validator(mode="after")
    def _require_handback_reason(self) -> "ExecutionResult":
        """Require a usable reason on a ``needs_replan`` hand-back.

        The submission schema marks ``replan_reason`` optional because a clean apply
        needs none, but a hand-back with an empty reason would route back to Planning
        with no context. Rejecting it here makes the loop re-prompt for a real reason
        instead of looping on a contentless retry.

        Returns:
            The validated model.

        Raises:
            ValueError: When a hand-back carries no reason.
        """
        if self.needs_replan and is_blank_reason(self.replan_reason):
            raise ValueError(
                "replan_reason is required and must be non-empty when "
                "needs_replan=true: name what in the plan could not be executed "
                "and what a revised plan should do differently."
            )
        return self


class VerifyResult(SubmissionModel):
    """End-to-end recovery verdict from the verification role.

    Verify runs after a clean apply and judges the incident at the system level: not
    whether a step applied, which Execution already confirmed, but whether the
    affected request path actually works now. Pods being Running and Ready is not
    enough; a request must be able to flow through. Verify owns the decision to end
    the pipeline or to hand back.
    """

    schema_description: ClassVar[str | None] = (
        "End-to-end recovery verdict from the Verify step.\n\nVerify runs AFTER a clean apply and judges the incident at the SYSTEM level —\nnot whether a step applied (Execution already confirmed that), but whether the\naffected request path actually works now. Pods being Running/Ready is not\nenough: a request must be able to flow through. Verify owns the decision to end\nthe pipeline (recovered) or hand back (replan / reassess)."
    )

    recovered: bool = Field(
        description=(
            "True ONLY when the incident is resolved at the SYSTEM level — the "
            "concrete firing alerts / degraded SLI have cleared AND the affected "
            "request path works end to end. Pods Running/Ready is NOT sufficient. "
            "False when the symptom persists or end-to-end recovery could not be "
            "confirmed."
        ),
    )
    verdict: Literal["recovered", "not_recovered", "inconclusive"] = Field(
        description=(
            "recovered: end-to-end checks confirm the system works and the "
            "concrete alerts cleared. not_recovered: the symptom still reproduces. "
            "inconclusive: "
            "telemetry/probes were unavailable or contradictory and recovery could "
            "not be established. Must be 'recovered' iff recovered=true."
        ),
    )
    checks: list[str] = Field(
        description=(
            "Required, at least one. The end-to-end checks you actually ran and what "
            "each showed — the re-evaluated alert query and its value, an "
            "error-rate/latency SLI, a trace or endpoint check, or a request probe. "
            "Name the command/query, the observed value, and the pass/fail verdict. "
            "Do not list pod-liveness checks alone."
        ),
    )
    summary: str = Field(
        description=(
            "Required, non-empty. What end-to-end state you observed and why the "
            "incident is or is not resolved at the system level."
        ),
    )
    needs_replan: bool = Field(
        default=False,
        description=(
            "Set true when the change was applied correctly but the system is still "
            "broken AND the diagnosed root cause still looks right — the plan was "
            "insufficient or incomplete and a fuller plan against the SAME cause "
            "could work. Also set recovered=false and fill replan_reason; routes back "
            "to Planning. Mutually exclusive with needs_reassessment."
        ),
    )
    replan_reason: str | None = Field(
        default=None,
        description=(
            "Required when needs_replan=true: name what the applied plan failed to "
            "achieve and what a fuller plan against the same cause should add."
        ),
    )
    needs_reassessment: bool = Field(
        default=False,
        description=(
            "Set true when the change was applied correctly but the system is still "
            "broken AND the evidence now points to a DIFFERENT or deeper root cause "
            "than the one diagnosed. Also set recovered=false and fill "
            "reassessment_reason; routes back to RCA. Mutually exclusive with "
            "needs_replan."
        ),
    )
    reassessment_reason: str | None = Field(
        default=None,
        description=(
            "Required when needs_reassessment=true: state what was already applied "
            "(so RCA does not repeat it) and why the persisting symptom points to a "
            "different root cause to investigate."
        ),
    )
    entry_point_status: str = Field(
        default="",
        description=(
            "Required, non-empty, when recovered=true. The observed health of the "
            "application's own user-facing entry point, established WITHOUT reference "
            "to RCA's causal chain — name what you probed (the Ingress-backed / "
            "LoadBalancer / load-generator-targeted Service) and the observed result. "
            "This is the incident-independent anchor: if RCA named the wrong symptom, "
            "every chain-derived check inherits that error, and only this one does "
            "not. When handing back, report whatever you observed, including that the "
            "probe was impossible and why."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence (0.0–1.0) in the recovery verdict, based on how directly the "
            "checks exercised the affected request path."
        ),
    )

    @field_validator("checks", mode="before")
    @classmethod
    def _decode_checks(cls, value: object) -> object:
        return decode_container(value)

    @model_validator(mode="after")
    def _validate(self) -> "VerifyResult":
        """Enforce the verdict's internal consistency and its evidentiary floor.

        Returns:
            The validated model.

        Raises:
            ValueError: When the verdict is inconsistent or insufficiently grounded.
        """
        # An entry-point observation taken independently of RCA's chain is what stops
        # a wrong diagnosis from certifying its own recovery: RCA names a symptom,
        # Planning fixes it, and a chain-derived check then confirms exactly the thing
        # RCA already believed. Blank here means the anchor was never established.
        #
        # Only recovered=true needs that anchor. Demanding it on a hand-back inverts
        # the incentive: the namespaces where a probe is impossible, entry Service
        # gone, every pod pending, DNS down, are exactly the ones worth handing back,
        # and rejecting the submission there discards a correct verdict. Measured on
        # the 2026-08 verify campaign: 1 of 141 verify runs died this way, on
        # rbac_misconfiguration, where the model had already concluded
        # recovered=false; the run was marked failed, the hand-back never happened,
        # and the cell graded as a mitigation failure.
        if self.recovered and is_blank_reason(self.entry_point_status):
            raise ValueError(
                "recovered=true requires a non-empty entry_point_status: probe the "
                "application's user-facing entry point and report what you observed, "
                "independently of RCA's causal chain."
            )
        # "Pods are 1/1 Ready with 0 restarts" is the single most common way a wrong
        # diagnosis passes verification: workloads are healthy precisely because the
        # fix was applied to the wrong object, while the incident's own path keeps
        # failing. Recovery therefore has to rest on at least one observation of
        # traffic or an SLI, not on the control plane's opinion of the pods.
        if self.recovered and not any(cites_request_path(c) for c in (self.checks or [])):
            raise ValueError(
                "recovered=true requires at least one check that exercises the affected "
                "request path or reads its SLI — a probe/curl/wget, a trace or endpoint "
                "check, or an error-rate/latency/probe_success query with its observed "
                "value. Pod, deployment and rollout status cannot establish recovery."
            )
        if self.recovered and self.verdict != "recovered":
            raise ValueError("recovered=true requires verdict='recovered'.")
        if not self.recovered and self.verdict == "recovered":
            raise ValueError("verdict='recovered' requires recovered=true.")
        if self.recovered and (self.needs_replan or self.needs_reassessment):
            raise ValueError("recovered=true cannot also set needs_replan/needs_reassessment.")
        if self.needs_replan and self.needs_reassessment:
            raise ValueError(
                "needs_replan and needs_reassessment are mutually exclusive: a "
                "persisting symptom is either an insufficient plan (same cause) or a "
                "wrong cause, not both."
            )
        # A not_recovered verdict with no hand-back silently ends the incident with
        # the symptom still live: the routing layer has nowhere to send it, so the
        # pipeline closes as if it were done. Measured on the loop-arm corpus, 6 of 21
        # not_recovered verdicts took this exit, a third of the arm's re-entry
        # opportunities, lost to an unset flag rather than to any judgement. Verify
        # has already decided the incident is unresolved; it must also say which way
        # back.
        if self.verdict == "not_recovered" and not (self.needs_replan or self.needs_reassessment):
            raise ValueError(
                "verdict='not_recovered' requires a hand-back: set needs_replan=true "
                "(the cause still looks right, the plan was insufficient) or "
                "needs_reassessment=true (the change applied but the cause was wrong), "
                "with the matching reason. Use verdict='inconclusive' only when the "
                "evidence genuinely could not establish recovery either way."
            )
        if self.needs_replan and is_blank_reason(self.replan_reason):
            raise ValueError(
                "replan_reason is required and must be non-empty when needs_replan=true."
            )
        if self.needs_reassessment and is_blank_reason(self.reassessment_reason):
            raise ValueError(
                "reassessment_reason is required and must be non-empty when "
                "needs_reassessment=true."
            )
        return self


class SingleDiagnosis(SubmissionModel):
    """Free-text diagnosis from the single-agent configuration.

    Deliberately unstructured. The single-agent configuration submits a
    natural-language root-cause description, exactly as a benchmark's own reference
    agents do, rather than the structured ``RootCause`` the multi-role pipeline
    produces. It occupies the same state slot; outcome resolution duck-types it, and
    since it has no ``halt`` field the pipeline treats it as a normal diagnosis.
    """

    schema_description: ClassVar[str | None] = (
        "Free-text diagnosis from the single-agent baseline (AGENT_MODE=single).\n\nDeliberately unstructured: the minimal baseline submits a natural-language\nroot-cause description — exactly as SREGym's reference agents do — rather than\nthe structured RootCause the multi-agent pipeline produces. Stored in the\n`root_cause` state slot; persistence and outcome resolution duck-type it (it\nhas no `halt`, so the pipeline treats it as a normal diagnosis)."
    )

    solution: str = Field(description="Natural-language root-cause description.")


class SingleResult(SubmissionModel):
    """Terminal result from the single-agent configuration's mitigation phase."""

    schema_description: ClassVar[str | None] = (
        "Terminal result from the single-agent baseline's mitigation phase.\n\nStored in the `execution_result` state slot. `succeeded` is the field\nresolve_pipeline_outcome reads; SREGym grades cluster health independently."
    )

    succeeded: bool = Field(description="Whether the agent believes its fix resolved the incident.")
    summary: str = Field(description="What was changed and what verification showed.")


class LearningResult(SubmissionModel):
    """The distillation an offline learning pass produces from a completed incident."""

    observations: list[str] = Field(
        min_length=1,
        description=(
            "Incident observations only: symptoms, exact Kubernetes identifiers, "
            "error strings, telemetry values, and status facts. Do not include a "
            "cause, hypothesis, remediation, or success claim."
        ),
    )
    attempted_hypothesis: str = Field(
        description="The diagnosis attempted by this session, whether right or wrong.",
    )
    attempted_action: str = Field(
        description="The remediation attempted, or why remediation halted.",
    )
    observed_outcome: str = Field(
        description=(
            "What was actually observed after the attempt. Distinguish recovery, "
            "continued failure, and unavailable/inconclusive verification."
        ),
    )
    case_negative_constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Case-specific conditions under which the attempted hypothesis or action "
            "must not be reused."
        ),
    )
    corrected_cause: str = Field(
        default="",
        description=(
            "Corrected cause only when an authoritative correction is present in the "
            "packet. Leave empty for scores-only feedback."
        ),
    )
    discriminator: str = Field(
        default="",
        description=(
            "The transferable part of this incident: which observation separated the "
            "correct cause from the plausible wrong one, and the concrete test that "
            "tells them apart on a future incident. State it so it applies to a "
            "different application and different resource names — no namespace, "
            "service, or workload names. Empty when nothing generalises."
        ),
    )
    summary: str = Field(
        description=(
            "Full incident summary, at most 500 words, following the section headings "
            "defined in the post-incident prompt. Returned on demand; not used as the "
            "retrieval source text."
        ),
    )
    fault_category: str = Field(
        default="general",
        description=(
            "Short label for the fault class or knowledge category "
            "(e.g. 'cpu_throttling', 'oom_kill', 'topology', 'general')."
        ),
    )
    verification_verdict: str = Field(
        default="",
        description=(
            "Independent live-recovery verdict from the learning pass's own re-check "
            "of the cluster. One of 'recovered', 'not_recovered', or 'inconclusive'. "
            "Leave empty only if live verification was not performed."
        ),
    )

    @field_validator("observations", "case_negative_constraints", mode="before")
    @classmethod
    def _decode_lists(cls, value: object) -> object:
        return decode_container(value)


class RoleTraceSummary(BaseModel):
    """A bounded summary of one role's process, retained for the learning pass."""

    role: str
    goal: str
    key_observations: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    errors_or_uncertainties: list[str] = Field(default_factory=list)
    output_summary: str = ""
    omitted_details_hint: str = ""


class LearningPacket(BaseModel):
    """What an offline learning pass receives about one completed incident.

    Assembled from stored records rather than from live state, because learning runs
    after the incident has closed.
    """

    session_id: str
    lifecycle_state: str
    alert: dict[str, Any]
    root_cause: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    verify_result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The system's own end-to-end recovery verdict, when verification ran. "
            "The prototype's packet omitted this, so the learning pass judged an "
            "incident's outcome without the one role that had actually checked "
            "whether the system recovered. Populating it changes what learning sees, "
            "so the role that consumes this decides how to use it."
        ),
    )
    trace_summaries: list[RoleTraceSummary] = Field(default_factory=list)
    operator_notes: str | None = None
    external_feedback: dict[str, Any] | None = Field(
        default=None,
        description=(
            "External evaluation of this session, when a source of truth is "
            "available, such as a benchmark oracle or a human grader. A neutral "
            "slot: the producer decides what goes here and the learning pass treats "
            "it as ground truth."
        ),
    )
