"""Typed artifacts exchanged between roles, and the audit event records.

These schemas are the frozen application binary interface of the system. They are
held constant across experiment configurations so that comparisons between them
remain single-variable. Changing a field is a research-validity decision, not a
refactor. See STYLE_GUIDE.md, "Typed artifacts are frozen".
"""

from kira.contracts.artifacts import (
    ExecutionResult,
    LearningResult,
    Plan,
    PlanStep,
    RoleTraceSummary,
    RootCause,
    SingleDiagnosis,
    SingleResult,
    StepResult,
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
from kira.contracts.state import IncidentState, TargetStage

__all__ = [
    "AuditEvent",
    "EventPayload",
    "ExecutionResult",
    "IncidentState",
    "InterventionEvent",
    "LearningResult",
    "ModelCallEvent",
    "Plan",
    "PlanStep",
    "PolicyDecisionEvent",
    "RoleTraceSummary",
    "RootCause",
    "SingleDiagnosis",
    "SingleResult",
    "StageEvent",
    "StepResult",
    "SubmissionEvent",
    "TargetStage",
    "ToolCallEvent",
    "VerifyResult",
]
