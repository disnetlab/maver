"""Context-local model-call observations, without changing the Completion protocol."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any

Observer = Callable[[str, dict[str, Any]], None]
_current: ContextVar[Observer | None] = ContextVar("maver_model_observer", default=None)


@contextmanager
def observe_calls(observer: Observer) -> Iterator[None]:
    """Bind one call's observer, isolated from concurrent incidents and nested calls."""
    token = _current.set(observer)
    try:
        yield
    finally:
        _current.reset(token)


def observe(kind: str, **detail: Any) -> None:
    """Report a provider boundary to the current logical call, when recorded."""
    observer = _current.get()
    if observer is not None:
        observer(kind, detail)


def serialisable(value: Any) -> Any:
    """Project provider values to JSON without serialising private transport internals."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): serialisable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serialisable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return serialisable(asdict(value))
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return serialisable(dump(mode="json"))
    # Provider fakes and SDK objects without model_dump still expose these public
    # response fields. Never fall back to repr: it can contain request credentials.
    names = (
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "index",
        "message",
        "finish_reason",
        "logprobs",
        "role",
        "content",
        "reasoning",
        "reasoning_content",
        "tool_calls",
        "function",
        "name",
        "arguments",
        "type",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cached_tokens",
        "reasoning_tokens",
    )
    return {name: serialisable(getattr(value, name)) for name in names if hasattr(value, name)}


def public_response(raw: Any) -> dict[str, Any]:
    """Keep the full public model response body, excluding SDK/private HTTP metadata."""
    result = serialisable(raw)
    if not isinstance(result, dict):
        return {"body": result}
    return {
        key: value
        for key, value in result.items()
        if not key.startswith("_")
        and key.lower()
        not in {
            "headers",
            "request",
            "api_key",
            "authorization",
            "cookies",
        }
    }


def error_record(exc: BaseException, *, secret: str | None = None) -> dict[str, Any]:
    """Describe a failure without retaining an HTTP request or credential value."""
    message = str(exc)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return {
        "type": type(exc).__name__,
        "message": message,
        "status_code": getattr(exc, "status_code", None),
        "error_class": getattr(exc, "error_class", None),
        "attempts": getattr(exc, "attempts", None),
    }
