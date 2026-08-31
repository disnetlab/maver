"""Typed artifacts exchanged between roles, and the audit event records.

These schemas are the frozen application binary interface of the system. They are
held constant across experiment configurations so that comparisons between them
remain single-variable. Changing a field is a research-validity decision, not a
refactor. See STYLE_GUIDE.md, "Typed artifacts are frozen".
"""

from kira.contracts.artifacts import (
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
from kira.contracts.events import (
    AuditEvent,
    EventPayload,
    InterventionEvent,
    ModelCallEvent,
    PolicyDecisionEvent,
    StageEvent,
    SubmissionEvent,
    ToolCallEvent,
)
from kira.contracts.lifecycle import LifecycleState, RunStatus
from kira.contracts.state import IncidentState, TargetStage

__all__ = [
    "AuditEvent",
    "EventPayload",
    "ExecutionResult",
    "IncidentState",
    "InterventionEvent",
    "LearningPacket",
    "LearningResult",
    "LifecycleState",
    "ModelCallEvent",
    "Plan",
    "PlanStep",
    "PolicyDecisionEvent",
    "RoleTraceSummary",
    "RootCause",
    "RunStatus",
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
