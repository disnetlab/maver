"""The tools each role may call, and the rules about them."""

from kira.tools.context import ToolContext
from kira.tools.registry import (
    Authority,
    BoundTool,
    ToolRegistryError,
    ToolSafety,
    ToolSpec,
    definition_for,
    description_for,
    load_entry_point_tools,
    registered,
    tool,
    validate_registry,
)
from kira.tools.schema import RATIONALE_ARG, normalise_schema, strip_rationale, with_rationale

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
