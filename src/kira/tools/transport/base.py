"""What a tool needs from the environment, and how an environment says what it offers.

A tool never talks to a cluster or a telemetry backend directly. It asks a transport,
and an environment adapter supplies the transport. That is the seam that keeps the core
free of environment branching: the prototype opened its central call path with
``if aiopslab_bridge_enabled():``, so adding a second benchmark meant editing the module
every tool depended on.

Capabilities make an environment's gaps explicit. A benchmark that deploys no log
backend does not cause its log tool to fail at incident time; the tool is dropped from
every role's toolset before the run starts, and the drop is recorded. The difference
matters because a tool that errors looks to the model like a fault it should work around,
and a tool that was never offered does not.
"""

from typing import Any, Literal, Protocol, runtime_checkable

__all__ = ["CAPABILITIES", "Capability", "ToolTransport"]

Capability = Literal[
    "kubernetes",
    "kubernetes.write",
    "kubernetes.rollback",
    "metrics",
    "logs",
    "traces",
    "host",
]
"""What a transport can serve.

Split finely enough that a real gap is expressible. ``kubernetes.rollback`` is separate
from ``kubernetes.write`` because replaying an inverse command needs a journal of what
was applied, which a transport talking straight to an API server does not keep.
"""

CAPABILITIES: frozenset[str] = frozenset(
    {
        "kubernetes",
        "kubernetes.write",
        "kubernetes.rollback",
        "metrics",
        "logs",
        "traces",
        "host",
    }
)
"""The declared vocabulary, used to reject a typo in a tool's ``requires`` at import.

Without this check a misspelt requirement would silently disable its tool everywhere,
which reads in the results exactly like a model that chose not to call it.
"""


@runtime_checkable
class ToolTransport(Protocol):
    """One environment's implementation of what tools need.

    Implementations are per-run and may hold connection state. A transport says what it
    can do rather than being asked to do something it cannot: ``capabilities`` is read
    once when the toolset is built.
    """

    @property
    def name(self) -> str:
        """Which transport this is, for the audit record.

        Returns:
            A short identifier, e.g. ``direct`` or ``mcp``.
        """
        ...

    @property
    def capabilities(self) -> frozenset[str]:
        """What this transport can serve.

        Returns:
            A subset of :data:`CAPABILITIES`.
        """
        ...

    async def kubectl(self, command: str, *, namespace: str | None = None) -> str:
        """Run one kubectl command.

        Authority is decided before this is reached; a transport is not a policy layer.

        Args:
            command: The full command string as the model wrote it.
            namespace: Namespace to scope the command to, when the caller knows one.

        Returns:
            Combined output, as the model should see it.
        """
        ...

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        """Run one PromQL query.

        Args:
            query: The PromQL expression.
            window_minutes: How far back to look, when the query is a range query.

        Returns:
            The result, rendered for the model.
        """
        ...

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        """Run one log query.

        Args:
            query: The backend's query expression.
            last_n_minutes: How far back to look.

        Returns:
            Matching log lines, rendered for the model.
        """
        ...

    async def query_traces(self, operation: str, **kwargs: Any) -> str:
        """Read distributed traces.

        Args:
            operation: Which trace query to run, e.g. ``services`` or ``dependencies``.
            **kwargs: Operation-specific arguments.

        Returns:
            The result, rendered for the model.
        """
        ...
