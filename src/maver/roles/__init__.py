"""The roles that make up the pipeline, and the loop they run in.

Importing this package registers every shipped role. Registration is import-time, so a
role module nobody imports is a role the graph cannot find — and the failure is a missing
node rather than an error, which is why the imports are here rather than left to a caller.
"""

from maver.roles import (  # noqa: F401 - imported for their registrations
    execution,
    learning,
    planning,
    rca,
    single,
    verify,
)
from maver.roles.base import RoleContext, RoleOutcome, RoleSpec, StateDelta
from maver.roles.loop import LoopResult, SubmitArgsNormalizer, SubmitGuard, ToolFn, run_until_submit
from maver.roles.policy import LoopLimits, RepeatDecision, RepeatedCallTracker, repair_submission
from maver.roles.registry import (
    RoleRegistryError,
    get_role,
    load_plugin_roles,
    registered_roles,
    role,
    validate_roles,
)
from maver.roles.trace_summary import build_trace_summary

__all__ = [
    "LoopLimits",
    "LoopResult",
    "RepeatDecision",
    "RepeatedCallTracker",
    "RoleContext",
    "RoleOutcome",
    "RoleRegistryError",
    "RoleSpec",
    "StateDelta",
    "SubmitArgsNormalizer",
    "SubmitGuard",
    "ToolFn",
    "build_trace_summary",
    "get_role",
    "load_plugin_roles",
    "registered_roles",
    "repair_submission",
    "role",
    "run_until_submit",
    "validate_roles",
]
