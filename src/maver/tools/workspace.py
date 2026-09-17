"""The per-session scratch directory tools write into.

A tool result that runs to tens of thousands of characters is worse than useless in a
transcript: it crowds out the reasoning that has to fit alongside it, and the model
cannot grep it. So a large result is written to a file and the model is handed a head
plus a path, which it can then process with the sandbox tools instead of re-running an
expensive call.

An object rather than the prototype's context variable. The variable made a tool
impossible to call in a test without setting it first, and made the session directory an
invisible dependency of every tool that happened to save anything.
"""

import ast
import itertools
import json
import os
import re
from pathlib import Path

__all__ = ["SessionWorkspace"]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

_PASS_THROUGH = frozenset({"PATH", "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TZ"})


def _normalise_for_file(content: str) -> tuple[str, str]:
    """Canonicalise a structured payload to JSON for the saved file.

    Some backends return Python reprs — single quotes, ``True``, ``None`` — so a model's
    natural ``json.load`` on the saved file raises and it burns turns falling back to
    ``ast.literal_eval``. When the payload parses as either, rewrite it as canonical JSON.

    Args:
        content: The raw result.

    Returns:
        The body to write and the file extension to use.
    """
    if content.lstrip()[:1] in ("[", "{"):
        for parse in (json.loads, ast.literal_eval):
            try:
                return json.dumps(parse(content), indent=2, default=str), "json"
            except (ValueError, SyntaxError):
                continue
    return content, "log"


class SessionWorkspace:
    """One incident's scratch directory."""

    def __init__(self, path: str | Path) -> None:
        """Open a workspace.

        Args:
            path: The session directory. Created if absent.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._sequence = itertools.count(1)

    @property
    def outputs(self) -> Path:
        """The directory saved tool outputs go in.

        Returns:
            The path, created on first access.
        """
        out = self.path / "tool_outputs"
        out.mkdir(exist_ok=True)
        return out

    def save_tool_output(self, content: str, tool_name: str, limit: int) -> str:
        """Return the content, or save it and return a head plus a pointer.

        Content within the limit is returned **unchanged**. That is not an optimisation:
        adding an envelope unconditionally would change the text of every prompt the
        published results were measured on.

        Args:
            content: The tool's result.
            tool_name: Used in the saved file's name.
            limit: Characters above which the result is saved instead.

        Returns:
            What the model should see.
        """
        if len(content) <= limit:
            return content
        body, ext = _normalise_for_file(content)
        while True:
            name = f"{tool_name}_{next(self._sequence):04d}.{ext}"
            try:
                with (self.outputs / name).open("x", encoding="utf-8") as output:
                    output.write(body)
                break
            except FileExistsError:
                continue
        hint = " (valid JSON — json.load it)" if ext == "json" else ""
        return (
            content[:limit] + f"\n\n[Output truncated ({len(content)} chars). "
            f"Full output saved to tool_outputs/{name}{hint}]"
        )

    def save_named_output(self, content: str, name: str, head_limit: int = 1500) -> str:
        """Write a result to a file the model named, and return a head plus the path.

        The name is reduced to a safe basename, so it cannot escape the outputs
        directory however it was written.

        Args:
            content: The tool's result.
            name: The model's requested filename.
            head_limit: Characters of the head to return.

        Returns:
            What the model should see.
        """
        safe = _SAFE_NAME.sub("_", Path(name).name).strip("._") or "output"
        if not Path(safe).suffix:
            safe += ".log"
        (self.outputs / safe).write_text(content, encoding="utf-8")
        rel = f"tool_outputs/{safe}"
        more = (
            ""
            if len(content) <= head_limit
            else f"\n[... {len(content) - head_limit} more chars in file]"
        )
        return (
            f"[Saved {len(content)} chars to {rel}. Read/process it with code_exec "
            f"(your current directory) or bash_exec — do NOT run kubectl inside them.]\n"
            f"--- head of {rel} ---\n{content[:head_limit]}{more}"
        )

    def subprocess_env(self) -> dict[str, str]:
        """Build a restricted environment for a sandbox subprocess.

        ``HOME`` and ``TMPDIR`` are pointed at the session directory so a command cannot
        write to the real home directory or leave files outside the workspace. No cluster
        credentials are passed, and none are passed by anything layered on top of this
        either: a deployment whose transport offers a credential-free proxy adds that
        endpoint and the calling stage in :func:`maver.tools.execution.subprocess_env`,
        where the stage is known, and what a command reaches through it is screened by
        the same stage policy the approval gate applies.

        Returns:
            The environment to run with, before any route-specific wiring.
        """
        env = {k: os.environ[k] for k in _PASS_THROUGH if k in os.environ}
        env.setdefault("PATH", os.defpath)
        env["SESSION_DIR"] = str(self.path)
        env["HOME"] = str(self.path)
        env["TMPDIR"] = str(self.path)
        return env
