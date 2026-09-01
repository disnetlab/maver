"""The hook protocol and the stack that composes hooks around the model/tool loop.

The vocabulary deliberately mirrors LangChain 1.x middleware: ``before_model``,
``after_model``, ``wrap_model_call``, ``wrap_tool_call``. We adopt the pattern and not
the API, because that framework's middleware exists only inside its own agent factory,
which builds a graph we do not use. Borrowing the names costs nothing and makes the
architecture legible to anyone arriving from it.

The test for whether something belongs here: would you want to turn it off for one
experiment configuration, or reuse it across roles? Then it is a hook. The prototype's
loop reached 1,269 lines because the answer was always yes and the code always said
otherwise.

What must **not** be a hook is equally important, and is enforced by keeping it out of
this module: the turn budget, the submission contract that terminates a run, the
cross-turn repeat tracker, the counters and their interlocks, and the rescue admission
policy. Each of those reads state spanning several turns, and a hook that can only see
one turn cannot implement them correctly.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from kira.llm.types import Message, ModelRequest, ModelResponse, ToolResult

__all__ = [
    "AgentHook",
    "Correction",
    "MiddlewareStack",
    "ModelHandler",
    "ModelWrappingHook",
    "NamedHook",
    "ReviewingHook",
    "ShapingHook",
    "ToolHandler",
    "ToolRequest",
    "ToolWrappingHook",
    "TurnContext",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnContext:
    """What a hook may know about the turn it is acting on.

    Deliberately narrow. A hook sees the current turn and the role's identity, not the
    loop's counters, because a hook that could read the counters would start making
    decisions that belong to the loop.
    """

    role: str
    session_id: str
    turn: int
    caller: str
    """What will be recorded as the issuer of any model call in this turn."""
    scratch: dict[str, Any] = field(default_factory=dict)
    """Per-run space a hook may use for its own state. Not persisted."""


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """One tool invocation, as it reaches the hooks."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    role: str
    session_id: str
    rationale: str | None = None


@dataclass(frozen=True, slots=True)
class Correction:
    """A hook's decision to intervene after a turn.

    The loop appends ``message`` and takes another turn, unless ``end_run`` is set.
    ``control`` and ``detail`` are recorded as an intervention in the event log, so an
    operator can see afterwards that the system, and not the model, produced that turn.
    """

    message: Message | None
    control: str
    detail: str
    end_run: bool = False


ModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]
ToolHandler = Callable[[ToolRequest], Awaitable[ToolResult]]


@runtime_checkable
class NamedHook(Protocol):
    """The only thing every hook must have.

    A hook is identified by name in the audit log and in the warning written when one
    misbehaves, so the name is what the stack genuinely requires. Everything else is
    checked for at the point of use.
    """

    name: str


@runtime_checkable
class ShapingHook(NamedHook, Protocol):
    """Shapes a request before it is sent."""

    def before_model(self, ctx: TurnContext, request: ModelRequest) -> ModelRequest | None:
        """Shape the request from loop state, before it is sent.

        Args:
            ctx: The turn being prepared.
            request: The request as assembled so far.

        Returns:
            A replacement request, or None to leave it unchanged.
        """
        ...


@runtime_checkable
class ReviewingHook(NamedHook, Protocol):
    """Inspects a completed turn and may intervene."""

    def after_model(self, ctx: TurnContext, response: ModelResponse) -> Correction | None:
        """Inspect a completed turn and decide whether to intervene.

        Runs before any tool is dispatched, so a hook can stop an unproductive turn from
        costing tool calls as well as tokens.

        Args:
            ctx: The turn that just completed.
            response: What the model returned.

        Returns:
            A correction to apply, or None to proceed.
        """
        ...


@runtime_checkable
class ModelWrappingHook(NamedHook, Protocol):
    """Owns the model round-trip."""

    async def wrap_model_call(
        self, ctx: TurnContext, request: ModelRequest, handler: ModelHandler
    ) -> ModelResponse:
        """Own the request and response round-trip.

        Args:
            ctx: The turn being run.
            request: The request to send.
            handler: The next layer. Call it to proceed.

        Returns:
            The response.
        """
        ...


@runtime_checkable
class ToolWrappingHook(NamedHook, Protocol):
    """Owns one tool invocation."""

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> ToolResult:
        """Own one tool invocation.

        This is where authority lives: a hook may refuse the call, pause for a decision,
        or bound how long it may run.

        Args:
            request: The invocation.
            handler: The next layer. Call it to proceed.

        Returns:
            The result.
        """
        ...


AgentHook = ShapingHook | ReviewingHook | ModelWrappingHook | ToolWrappingHook
"""A hook implements at least one of the four phases, and the stack skips the rest.

Declared as a union rather than as one protocol with four optional methods, because
Python has no way to say a protocol member is optional: a single protocol naming all
four would be satisfied by no real hook, which is what it was. The stack checks for each
method at the point it would call it, so a hook that implements one phase is a valid
hook. A hook that raises is logged and skipped rather than ending the run, because a run
cut short by a misbehaving accessory is worse than a run that lost one accessory.
"""


_PHASES = (ShapingHook, ReviewingHook, ModelWrappingHook, ToolWrappingHook)


def _check(hook: object) -> None:
    """Refuse a hook that would be silently inert.

    Each phase is dispatched only to the hooks that implement it, so a hook missing a
    name, or implementing no phase at all, would simply never run. That is the failure
    a stack is least likely to notice: nothing errors, and the concern the hook exists
    to enforce is quietly absent. It is a startup error instead.

    Args:
        hook: The candidate.

    Raises:
        TypeError: When the hook has no name, or implements no phase.
    """
    if not isinstance(hook, NamedHook):
        raise TypeError(f"{type(hook).__name__} is not a hook: it has no name")
    if not any(isinstance(hook, phase) for phase in _PHASES):
        raise TypeError(f"hook {hook.name!r} implements no phase and would never run")


class MiddlewareStack:
    """Composes hooks in declared order.

    Ordering is onion-style for the wrapping hooks: the first hook in the list is the
    outermost, so it sees the call before every later hook and the result after them.
    The shaping hooks run first-to-last for ``before_model`` and stop at the first
    correction for ``after_model``, because two hooks correcting the same turn would
    produce two injected messages for one problem.
    """

    def __init__(
        self,
        hooks: Sequence[AgentHook] = (),
        *,
        on_error: Callable[[str, str, BaseException], None] | None = None,
    ) -> None:
        """Build a stack.

        Args:
            hooks: The hooks, outermost first.
            on_error: Notified when a hook raises. Receives the hook name, the phase,
                and the exception. Defaults to logging.
        """
        self.hooks = list(hooks)
        self._on_error = on_error or self._log_error
        for hook in self.hooks:
            _check(hook)

    def _report(self, hook: NamedHook, phase: str, exc: BaseException) -> None:
        self._on_error(getattr(hook, "name", type(hook).__name__), phase, exc)

    @staticmethod
    def _log_error(hook: str, phase: str, exc: BaseException) -> None:
        _log.warning("hook %s failed in %s and was skipped: %s", hook, phase, exc, exc_info=exc)

    def _guard(self, hook: NamedHook, phase: str, call: Callable[[], Any]) -> Any:
        try:
            return call()
        except Exception as exc:
            self._on_error(getattr(hook, "name", type(hook).__name__), phase, exc)
            return None

    def before_model(self, ctx: TurnContext, request: ModelRequest) -> ModelRequest:
        """Run every shaping hook in order.

        Args:
            ctx: The turn being prepared.
            request: The assembled request.

        Returns:
            The request after every hook has had a chance to shape it.
        """
        for hook in self.hooks:
            if not isinstance(hook, ShapingHook):
                continue
            try:
                shaped = hook.before_model(ctx, request)
            except Exception as exc:
                self._report(hook, "before_model", exc)
                continue
            if shaped is not None:
                request = shaped
        return request

    def after_model(self, ctx: TurnContext, response: ModelResponse) -> Correction | None:
        """Ask each hook whether this turn needs correcting, stopping at the first.

        Args:
            ctx: The turn that completed.
            response: What the model returned.

        Returns:
            The first correction requested, or None.
        """
        for hook in self.hooks:
            if not isinstance(hook, ReviewingHook):
                continue
            try:
                correction = hook.after_model(ctx, response)
            except Exception as exc:
                self._report(hook, "after_model", exc)
                continue
            if correction is not None:
                return correction
        return None

    async def run_model(
        self, ctx: TurnContext, request: ModelRequest, handler: ModelHandler
    ) -> ModelResponse:
        """Invoke the model through every wrapping hook.

        Args:
            ctx: The turn being run.
            request: The request to send.
            handler: The innermost call.

        Returns:
            The response.
        """
        wrapped = handler
        for hook in reversed(self.hooks):
            if not isinstance(hook, ModelWrappingHook):
                continue
            wrapped = self._wrap_model(hook, ctx, wrapped)
        return await wrapped(request)

    @staticmethod
    def _wrap_model(hook: ModelWrappingHook, ctx: TurnContext, nxt: ModelHandler) -> ModelHandler:
        async def _call(request: ModelRequest) -> ModelResponse:
            return await hook.wrap_model_call(ctx, request, nxt)

        return _call

    async def run_tool(self, request: ToolRequest, handler: ToolHandler) -> ToolResult:
        """Invoke a tool through every wrapping hook.

        A hook that raises here is **not** swallowed. Unlike the shaping hooks, a
        wrapping hook that fails has already taken responsibility for the call, and
        continuing as though it had not is how an authority check gets bypassed.

        Args:
            request: The invocation.
            handler: The innermost dispatch.

        Returns:
            The result.
        """
        wrapped = handler
        for hook in reversed(self.hooks):
            if not isinstance(hook, ToolWrappingHook):
                continue
            wrapped = self._wrap_tool(hook, wrapped)
        return await wrapped(request)

    @staticmethod
    def _wrap_tool(hook: ToolWrappingHook, nxt: ToolHandler) -> ToolHandler:
        async def _call(request: ToolRequest) -> ToolResult:
            return await hook.wrap_tool_call(request, nxt)

        return _call

    def with_hook(self, hook: AgentHook) -> "MiddlewareStack":
        """Return a stack with one more hook appended.

        Args:
            hook: The hook to add, innermost.

        Returns:
            A new stack. This one is unchanged.
        """
        return MiddlewareStack([*self.hooks, hook], on_error=self._on_error)


def replace_request(request: ModelRequest, **changes: Any) -> ModelRequest:
    """Build a modified copy of a request.

    Requests are frozen so a hook cannot mutate one another hook has already seen.

    Args:
        request: The request to copy.
        **changes: Fields to replace.

    Returns:
        The modified copy.
    """
    return replace(request, **changes)
