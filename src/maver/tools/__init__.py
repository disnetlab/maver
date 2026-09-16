"""The tools each role may call, and the rules about them.

Importing this package imports every capability module, which is what runs their
registrations, and then checks every rule about them. That check is deliberately at
import: a description missing for a role, a requirement naming a capability no transport
offers, a write-capable tool that cannot classify its own calls — each of those leaves
the model either uninstructed or short a tool, and none of them raises anything at
incident time.
"""

from maver.tools import (  # noqa: F401 - imported for their registrations
    execution,
    host,
    kubernetes,
    logs,
    memory,
    metrics,
    snapshot,
    submit,
    traces,
)
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.context import ToolContext
from maver.tools.registry import (
    Authority,
    BoundTool,
    ToolRegistryError,
    ToolSafety,
    ToolSpec,
    definition_for,
    description_for,
    load_descriptions,
    load_entry_point_tools,
    registered,
    tool,
    validate_registry,
)
from maver.tools.schema import RATIONALE_ARG, normalise_schema, strip_rationale, with_rationale

__all__ = [
    "RATIONALE_ARG",
    "Authority",
    "BoundTool",
    "ToolContext",
    "ToolRegistryError",
    "ToolSafety",
    "ToolSpec",
    "definition_for",
    "description_for",
    "load_entry_point_tools",
    "normalise_schema",
    "registered",
    "strip_rationale",
    "tool",
    "validate_registry",
    "with_rationale",
]

submit.register_all()
validate_registry(descriptions=load_descriptions(), allowlists=ROLE_TOOLSETS)
