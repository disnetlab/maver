"""Typed artifacts exchanged between roles, and the audit event records.

These schemas are the frozen application binary interface of the system. They are
held constant across experiment configurations so that comparisons between them
remain single-variable. Changing a field is a research-validity decision, not a
refactor. See STYLE_GUIDE.md, "Typed artifacts are frozen".
"""

from maver.contracts.artifacts import (
    ExecutionResult,
    LearningPacket,
    LearningResult,
    Plan,
    PlanStep,
    RoleTraceSummary,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    StepResult,
    SubmissionModel,
    VerifyResult,
)
from maver.contracts.events import (
    AuditEvent,
    EventPayload,
    InterventionEvent,
    ModelCallEvent,
    ModelCallStartedEvent,
    PolicyDecisionEvent,
    SecondPassEvent,
    StageEvent,
    SubmissionEvent,
    ToolCallEvent,
)
from maver.contracts.lifecycle import LifecycleState, RunStatus
from maver.contracts.state import IncidentState, TargetStage

SCHEMA_VERSION = 2
"""Version of the submission schemas and the event contract, together.

One number for both, because both are read back from stored results and neither moves
without the other's readers noticing. It is recorded in every run's configuration
snapshot, so a stored result says which schemas produced it rather than leaving a later
reader to infer them from the date.

It moves when a field is added to, removed from, or given a new meaning in an artifact
under :mod:`maver.contracts.artifacts` or an event under :mod:`maver.contracts.events`.
Bumping it is step 2 of the procedure in STYLE_GUIDE.md, "Typed artifacts are frozen";
step 3 is the entry in CHANGELOG.md that says which published comparisons the change
invalidates.
"""

__all__ = [
    "SCHEMA_VERSION",
    "AuditEvent",
    "EventPayload",
    "ExecutionResult",
    "IncidentState",
    "InterventionEvent",
    "LearningPacket",
    "LearningResult",
    "LifecycleState",
    "ModelCallEvent",
    "ModelCallStartedEvent",
    "Plan",
    "PlanStep",
    "PolicyDecisionEvent",
    "RoleTraceSummary",
    "RootCause",
    "RunStatus",
    "SecondPassEvent",
    "SingleDiagnosis",
    "SingleResult",
    "StageEvent",
    "StepResult",
    "SubmissionEvent",
    "SubmissionModel",
    "TargetStage",
    "ToolCallEvent",
    "VerifyResult",
]
