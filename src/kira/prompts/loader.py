"""Loading and composing prompt assets.

Prompts are ``.txt`` files shipped as package data and read through
``importlib.resources``, never by filesystem path. This is a deliberate deviation
from every major agent framework, which keep prompts in Python. Prompt content is an
experimental variable here: files diff cleanly in review, can be swapped without
touching code, and can be read by someone assessing the artifact who does not want to
read our code. See STYLE_GUIDE.md, "Prompts".

A role prompt is a template containing ``{{MARKER}}`` slots filled from composable
blocks. Templating is strict in both directions: a marker with no value raises, and a
value with no marker raises. Renaming a marker therefore cannot silently leave a
prompt half-wired, which is the failure this strictness exists to prevent.
"""

import re
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from pathlib import Path

__all__ = [
    "PromptError",
    "compose_role_prompt",
    "read_asset",
    "read_override",
    "render",
]

_MARKER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")

_ROLE_MARKERS: dict[str, str] = {
    "KUBECTL_GUIDANCE": "kubectl",
    "MEMORY_GUIDANCE": "memory",
}
"""Marker name to the block family that fills it."""

# Which kubectl guidance a role receives. Roles absent from this map hold no
# cluster tools and must not declare the marker.
_KUBECTL_BLOCK: dict[str, str] = {
    "rca": "read_only",
    "planning": "planning",
    "execution": "execution",
    "verify": "read_only",
    "learning": "read_only",
    "single": "single",
}


class PromptError(RuntimeError):
    """A prompt asset is missing, or a template and its values disagree."""


@cache
def read_asset(relative_path: str) -> str:
    """Read a prompt asset shipped inside the package.

    Args:
        relative_path: Path under ``kira/prompts``, using forward slashes.

    Returns:
        The asset's text.

    Raises:
        PromptError: When the asset does not exist.
    """
    resource = files("kira.prompts")
    for part in relative_path.split("/"):
        resource = resource / part
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise PromptError(f"prompt asset not found: {relative_path}") from exc


def read_override(path: Path, *, root: Path | None = None) -> str:
    """Read an operator-supplied prompt from outside the package.

    This is the single supported way to change a prompt. Editing a file inside the
    installed package is not, because it does not survive an upgrade and leaves no
    record of what was run.

    Args:
        path: The override file.
        root: Optional directory the path must stay within.

    Returns:
        The override's text.

    Raises:
        PromptError: When the file is missing or escapes ``root``.
    """
    resolved = path.expanduser().resolve()
    if root is not None:
        base = root.expanduser().resolve()
        if not resolved.is_relative_to(base):
            raise PromptError(f"prompt override escapes {base}: {path}")
    try:
        return resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"cannot read prompt override {path}: {exc}") from exc


def markers(template: str) -> set[str]:
    """Return the marker names a template declares.

    Args:
        template: Prompt text.

    Returns:
        Every distinct ``{{MARKER}}`` name found.
    """
    return set(_MARKER.findall(template))


def render(template: str, values: Mapping[str, str]) -> str:
    """Substitute every marker, rejecting any mismatch in either direction.

    Args:
        template: Prompt text containing ``{{MARKER}}`` slots.
        values: Marker name to replacement text.

    Returns:
        The rendered prompt, byte-identical to what the prototype's loader produced.
        Substitution is literal: no stripping and no collapsing of blank runs, because
        the composed string is a held-constant input and tidying it would change what
        the measured baseline was measured on.

    Raises:
        PromptError: When a marker has no value, or a value has no marker.
    """
    declared = markers(template)
    supplied = set(values)
    if missing := declared - supplied:
        raise PromptError(f"no value supplied for prompt marker(s): {sorted(missing)}")
    if unused := supplied - declared:
        raise PromptError(f"value supplied for absent prompt marker(s): {sorted(unused)}")

    def _replace(match: re.Match[str]) -> str:
        return values[match.group(1)]

    return _MARKER.sub(_replace, template)


def _block_for(marker: str, role: str, *, retrieval_enabled: bool) -> str:
    family = _ROLE_MARKERS.get(marker)
    if family == "kubectl":
        variant = _KUBECTL_BLOCK.get(role)
        if variant is None:
            raise PromptError(f"role {role!r} declares {{{{{marker}}}}} but holds no cluster tools")
        return read_asset(f"blocks/kubectl/{variant}.txt")
    if family == "memory":
        state = "enabled" if retrieval_enabled else "disabled"
        # An intentionally empty block is valid: the memory-off variant of a role
        # may add nothing rather than say something. Emptiness is carried over
        # deliberately, because inserting guidance here would change the behaviour
        # of the configuration the published baseline was measured on.
        return read_asset(f"blocks/memory/{role}.{state}.txt")
    raise PromptError(f"unknown prompt marker: {marker}")


def compose_role_prompt(
    role: str,
    *,
    retrieval_enabled: bool,
    override: str | None = None,
) -> str:
    """Build a role's system prompt from its template and the applicable blocks.

    Args:
        role: The role name, matching a file under ``prompts/roles``.
        retrieval_enabled: Whether the role can actually retrieve a past incident.
            This is a conjunction, not the memory setting alone: the memory-on block
            prescribes the retrieval tools by name, so emitting it for a role whose
            toolset withholds them tells the model to call a tool it does not have.
            The prototype gated on memory being on **and** the retrieval tool being
            enabled, and the configuration where those disagree is a real one: the
            second-pass control deliberately runs with memory on and no retrieval.
        override: Replacement template text, supplied by configuration.

    Returns:
        The fully rendered system prompt.

    Raises:
        PromptError: When an asset is missing or a marker cannot be filled.
    """
    template = override if override is not None else read_asset(f"roles/{role}.txt")
    values = {
        marker: _block_for(marker, role, retrieval_enabled=retrieval_enabled)
        for marker in markers(template)
    }
    return render(template, values)
