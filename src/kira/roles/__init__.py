"""The roles that make up the pipeline, and the loop they run in.

Importing this package registers every shipped role. Registration is import-time, so a
role module nobody imports is a role the graph cannot find — and the failure is a missing
node rather than an error, which is why the imports are here rather than left to a caller.
"""

from kira.roles import planning, rca  # noqa: F401 - imported for their registrations
from kira.roles.base import RoleContext, RoleOutcome, RoleSpec, StateDelta
from kira.roles.loop import LoopResult, SubmitGuard, ToolFn, run_until_submit
from kira.roles.policy import LoopLimits, RepeatDecision, RepeatedCallTracker, repair_submission
from kira.roles.registry import (
    RoleRegistryError,
    get_role,
    load_plugin_roles,
    registered_roles,
    role,
    validate_roles,
)

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
    "SubmitGuard",
    "ToolFn",
    "get_role",
    "load_plugin_roles",
    "registered_roles",
    "repair_submission",
    "role",
    "run_until_submit",
    "validate_roles",
]
