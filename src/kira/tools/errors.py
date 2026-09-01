"""How a tool reports failure, and how much of a result it may return.

A tool returns a string and never raises. Every failure is a return value whose first
characters are ``error:``, because far more of the system reads that prefix than reads
any exception: the loop's success heuristic, the repeat breaker, the namespace
snapshot's read-failure detection and the trace tools' safe-passthrough screen all key
on it. A tool that raised instead would be counted as a successful turn by every one of
them.

The output limits live here for the same reason the prefix does. They are part of the
model-facing interface — a prompt was written against results of this size — so they are
constants with a provenance, not a parameter each call site picks for itself.
"""

__all__ = [
    "EXEC_OUTPUT_LIMIT",
    "OUTPUT_LIMIT",
    "call_failed",
    "format_exit_result",
    "tool_error",
]

OUTPUT_LIMIT = 16384
"""Characters a cluster or telemetry read may return before it is saved to a file."""

EXEC_OUTPUT_LIMIT = 8192
"""The same, for the sandboxed execution tools, whose output is far less structured."""


def tool_error(message: str) -> str:
    """Build a model-facing failure string.

    Args:
        message: What went wrong, in the words the model should read.

    Returns:
        The message behind the ``error:`` prefix everything downstream recognises.
    """
    return f"error: {message}"


def format_exit_result(returncode: int, output: str) -> str:
    """Render a subprocess result, spelling out the silent cases.

    A bare ``exit_code=0`` with nothing after it is the single most repeated tool
    result in the VM3 A/B — 50 of ~400 ``bash_exec`` results, and ``bash_exec`` and
    ``code_exec`` together account for 70% of all repeat-breaker trips. An empty
    result gives a greedy decoder nothing to react to, so it reissues the same
    command. Naming the emptiness, and saying where to look instead, costs one line.

    Args:
        returncode: The process exit status.
        output: Its combined output.

    Returns:
        The result as the model should read it.
    """
    body = output.strip()
    if body:
        return f"exit_code={returncode}\n{output}"
    if returncode == 0:
        return (
            "exit_code=0 (no output)\n"
            "The command ran and printed nothing. Re-running it returns this same "
            "empty result. If you expected output, the path, resource, or selector "
            "you named matched nothing — widen the query or check one level up."
        )
    return (
        f"exit_code={returncode} (no output)\n"
        "The command failed silently. Re-running it returns this same empty result. "
        "Run it again with stderr shown, or check that the command and its inputs exist."
    )


def call_failed(content: str) -> bool:
    """Say whether a tool result records a failure.

    Two shapes, because there are two kinds of failure and the audit log conflated
    them. A tool that could not act at all says so with the prefix above. A command
    that ran and exited non-zero says so with its exit code, and reading only the
    prefix recorded that as a successful call — so a transport whose backend reports
    the exit code and one whose backend flags the error disagreed about the same
    failure, in the log that exists to be compared across them.

    Args:
        content: The tool result, as the model will read it.

    Returns:
        True when the result records a failure.
    """
    if content.startswith("error:"):
        return True
    head = content.split("\n", 1)[0]
    return head.startswith("exit_code=") and not head.startswith("exit_code=0")
