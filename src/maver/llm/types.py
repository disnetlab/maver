"""The types the loop and the client exchange.

These are provider-neutral. Everything a specific provider does differently is resolved
inside the client, so nothing above this layer imports the provider library or reaches
into a provider object.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from maver.settings.profiles import SamplingSet

__all__ = [
    "Completion",
    "Message",
    "ModelRequest",
    "ModelResponse",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "ToolSpec",
    "Usage",
]

Message = dict[str, Any]
"""One chat message in the provider's schema."""

ToolChoice = str | dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as the model sees it."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    """The provider's own argument string, echoed back verbatim.

    Kept so the next request carries what the provider produced rather than a
    re-encoding of it, which is what makes identifier normalisation safe: the
    synthesised identifier appears in the echoed message and in the matching result.
    """
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returned, and what the audit log should know about it."""

    tool_call_id: str
    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    """Audit-only. Never serialised into the transcript the model sees."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts as the endpoint reported them.

    Every field may be None, and None means "not reported" rather than zero. A backend
    running prefix caching without reporting it produces a null cached count, and
    treating that as zero silently invents a measurement.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    uncached_prompt_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One completion request, before any provider-specific assembly."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice | None = None
    sampling: SamplingSet | None = None
    """Overrides the profile's set. Normally None: the profile is the source."""
    lite_reasoning: bool = False
    """A rescue turn, which applies the profile's rescue quirks."""
    suppress_reasoning: bool = False
    """Ask the server for no reasoning channel on this call, whatever the profile says.

    Separate from ``lite_reasoning`` because the two answer different questions. A
    rescue turn asks the profile whether suppressing helps this model; a helper that
    must return one JSON object inside a few hundred tokens cannot afford a reasoning
    channel on any model, and the measured runs set it unconditionally on all three.
    """


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One completion, normalised."""

    text: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    reasoning: str | None
    usage: Usage
    duration_ms: int
    attempts: int
    estimated_cost_usd: float | None = None
    notes: tuple[str, ...] = ()
    """What the client repaired or normalised, recorded on the audit event."""
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class Completion(Protocol):
    """What the loop depends on. A test substitutes a fake and nothing else changes."""

    async def complete(
        self,
        request: ModelRequest,
        *,
        caller: str,
    ) -> ModelResponse:
        """Run one completion.

        Args:
            request: The messages, tools and sampling for this call.
            caller: What issued the call: a role name, or the mechanism name for a
                helper such as a quality gate, nominator or selector. Required, with
                no default, so that no call can escape cost accounting. The prototype
                counted only role-issued calls and had to publish its cost figures with
                that gap disclosed.

        Returns:
            The normalised response.

        Raises:
            ModelCallError: When the call failed and retrying did not help.
        """
        ...


def serialise_tool_results(results: Sequence[ToolResult]) -> list[Message]:
    """Render tool results as transcript messages.

    ``metadata`` is deliberately not serialised: it is audit information a tool attached
    to its result, and echoing it back would put content into the model's context that
    no one wrote for the model.

    Args:
        results: Results to render, in call order.

    Returns:
        One message per result.
    """
    return [{"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content} for r in results]
