"""What a tool needs that the model did not supply.

Every dispatch takes this as its first argument. The prototype passed the equivalent
through four separate context variables, which made a tool impossible to call in a test
without a fixture that set all four, and made the set of things a tool could reach
invisible at its call site.

The context grows as the subsystems it carries land; it is deliberately built from what
exists rather than stubbed ahead of them.
"""

from dataclasses import dataclass

from kira.audit.journal import Journal
from kira.tools.transport.base import ToolTransport

__all__ = ["ToolContext"]


@dataclass(frozen=True)
class ToolContext:
    """One role's run, from a tool's point of view."""

    transport: ToolTransport
    role: str
    """The calling role. This is the stage the approval gate keys on."""
    session_id: str
    journal: Journal
