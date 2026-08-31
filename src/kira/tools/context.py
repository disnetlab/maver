"""What a tool needs that the model did not supply.

Every dispatch takes this as its first argument. The prototype passed the equivalent
through four separate context variables, which made a tool impossible to call in a test
without a fixture that set all four, and made the set of things a tool could reach
invisible at its call site.

The context grows as the subsystems it carries land; it is deliberately built from what
exists rather than stubbed ahead of them.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kira.audit.journal import Journal
from kira.memory.ports import RunStore, SessionStore
from kira.memory.retrieval import Retriever
from kira.settings.groups import ToolSettings
from kira.tools.transport.base import ToolTransport
from kira.tools.workspace import SessionWorkspace

__all__ = ["EvidencePolicy", "NullEvidencePolicy", "ToolContext"]


@runtime_checkable
class EvidencePolicy(Protocol):
    """Removes from a tool result anything the environment should not have revealed.

    A benchmark can leak its own fault injection into the telemetry it serves — a
    chaos-injector's own labels appearing in the metrics an agent queries, for instance,
    which turns a diagnosis into a lookup. Filtering belongs to the adapter that knows
    its environment, not to the tools, which is why this is a port and not a function.
    """

    def filter_result(self, content: str) -> str:
        """Redact anything the model should not see.

        Args:
            content: A tool result.

        Returns:
            The result to hand to the model.
        """
        ...

    def filter_query(self, query: str) -> str:
        """Rewrite a query so it cannot reach what the environment is hiding.

        Filtering a result is not enough on its own: a model that can *ask* for a
        hidden namespace learns it exists from the shape of the answer. The rewrite
        and the redaction are the same policy seen from two directions, so they belong
        to the same object.

        Args:
            query: The query the model wrote.

        Returns:
            The query to actually run.
        """
        ...


class NullEvidencePolicy:
    """Filters nothing. The default, for any environment with nothing to hide."""

    def filter_result(self, content: str) -> str:
        """Return the content unchanged.

        Args:
            content: A tool result.

        Returns:
            The same content.
        """
        return content

    def filter_query(self, query: str) -> str:
        """Return the query unchanged.

        Args:
            query: The query the model wrote.

        Returns:
            The same query.
        """
        return query


@dataclass(frozen=True)
class ToolContext:
    """One role's run, from a tool's point of view."""

    transport: ToolTransport
    workspace: SessionWorkspace
    role: str
    """The calling role. This is the stage the approval gate keys on."""
    session_id: str
    journal: Journal
    settings: ToolSettings = field(default_factory=ToolSettings)
    """Tool-layer limits, injected rather than read from the process.

    A tool that reaches for global settings behaves differently depending on process
    state its call site cannot see, which is how the prototype ended up with five
    environment reads hidden inside its toolset builder.
    """
    evidence: EvidencePolicy = field(default_factory=NullEvidencePolicy)
    memory: Retriever | None = None
    """The experience store, or None when this configuration has none."""
    sessions: SessionStore | None = None
    runs: RunStore | None = None
    """Role runs and their traces, when this deployment records them."""
    memory_exclude: tuple[str, ...] = ()
    """Session ids retrieval must withhold, supplied per run by an evaluation harness."""
    memory_aliases: dict[str, str] = field(default_factory=dict)
    """Maps the short handles shown to the model back to real session ids.

    Per run rather than global: a handle means "the first result of *my* search", and
    two concurrent incidents would otherwise resolve each other's.
    """
