"""Optional correlation for nested I/O inside an audited call.

Context variables carry identifiers across async transport boundaries; tools still
receive their functional dependencies explicitly through ToolContext.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolObservation:
    """Measurements collected during one tool attempt, never added to its prompt."""

    status: str | None = None
    incomplete: bool = False
    unavailable: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    truncated: bool = False
    upstream_truncated: bool = False
    output_artifacts: list[dict[str, Any]] = field(default_factory=list)
    full_result_chars: int | None = None
    transport: str | None = None
    result: dict[str, Any] = field(default_factory=dict)
    retrieval: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CallContext:
    """Identity shared by one logical tool attempt and its nested backend requests."""

    invocation_id: str
    call_id: str | None
    run_id: str | None
    stage: str
    session_id: str
    observation: ToolObservation
    journal: Any
    tool: str = ""
    dispatching: bool = False


_current: ContextVar[CallContext | None] = ContextVar("maver_audit_call", default=None)


def current_call() -> CallContext | None:
    """Return the active audit correlation, or None outside a tool attempt."""
    return _current.get()


@contextmanager
def bind_call(context: CallContext) -> Iterator[CallContext]:
    """Bind correlation to this async task and restore its parent on every exit."""
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


_journal: ContextVar[Any | None] = ContextVar("maver_audit_journal", default=None)


def current_journal() -> Any | None:
    """Return the journal of the active role or its nested tool attempt."""
    call = current_call()
    return call.journal if call is not None else _journal.get()


@contextmanager
def bind_journal(journal: Any) -> Iterator[None]:
    """Keep auxiliary observations on the current role's audit stream."""
    token = _journal.set(journal)
    try:
        yield
    finally:
        _journal.reset(token)
