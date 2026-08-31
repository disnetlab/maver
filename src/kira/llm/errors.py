"""Model-call failures, classified by what a caller should do about them."""

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["ErrorClass", "ModelCallError", "classify"]


class ErrorClass(StrEnum):
    """What kind of failure a model call hit."""

    TRANSIENT = "transient"
    """A connection reset, read timeout or server error. Retrying is reasonable."""

    RATE_LIMIT = "rate_limit"
    """The endpoint is shedding load. Retry, with a longer schedule."""

    CONTEXT_OVERFLOW = "context_overflow"
    """The request does not fit the context window.

    Held apart from the other terminal failures because it is the one whose correct
    response is to send less rather than to send again. Retrying it unchanged is
    guaranteed to fail and costs a full prefill each time.
    """

    BAD_REQUEST = "bad_request"
    """A parameter or schema the provider refuses. Retrying cannot help."""

    AUTH = "auth"
    """Credentials are missing, wrong, or lack access to the model."""

    TERMINAL = "terminal"
    """Anything else. Not retried, because we do not know that retrying is safe."""


RETRYABLE = frozenset({ErrorClass.TRANSIENT, ErrorClass.RATE_LIMIT})


@dataclass(frozen=True, slots=True)
class ModelCallError(Exception):
    """A model call that failed after exhausting whatever retries applied."""

    error_class: ErrorClass
    provider_type: str
    message: str
    attempts: int
    status_code: int | None = None

    def __str__(self) -> str:
        """Render the failure for a log line.

        Returns:
            A one-line description naming the class, the provider's own exception type
            and the attempt count.
        """
        code = f" [{self.status_code}]" if self.status_code is not None else ""
        return (
            f"{self.error_class}{code} after {self.attempts} attempt(s): "
            f"{self.provider_type}: {self.message}"
        )


_CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
)


def classify(exc: BaseException, status_code: int | None = None) -> ErrorClass:
    """Decide how a provider exception should be treated.

    Classification reads the status code first and the message only as a fallback,
    because provider exception hierarchies differ while status codes do not. Context
    overflow is detected from the message because providers report it as a plain bad
    request.

    Args:
        exc: The exception the provider raised.
        status_code: Its HTTP status, when one is available.

    Returns:
        The class of failure.
    """
    code = status_code if status_code is not None else getattr(exc, "status_code", None)
    text = str(exc).lower()

    if any(marker in text for marker in _CONTEXT_MARKERS):
        return ErrorClass.CONTEXT_OVERFLOW
    if code == 429:
        return ErrorClass.RATE_LIMIT
    if code in (401, 403):
        return ErrorClass.AUTH
    if code == 400:
        return ErrorClass.BAD_REQUEST
    if isinstance(code, int) and 500 <= code < 600:
        return ErrorClass.TRANSIENT
    if isinstance(exc, TimeoutError | ConnectionError):
        return ErrorClass.TRANSIENT
    name = type(exc).__name__.lower()
    if any(
        marker in name for marker in ("timeout", "connection", "apiconnection", "internalserver")
    ):
        return ErrorClass.TRANSIENT
    if "ratelimit" in name:
        return ErrorClass.RATE_LIMIT
    if "authentication" in name or "permission" in name:
        return ErrorClass.AUTH
    return ErrorClass.TERMINAL
