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
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

__all__ = [
    "PROMPT_PACKAGE",
    "PromptError",
    "compose_role_prompt",
    "overridden_roles",
    "override_for",
    "read_asset",
    "read_override",
    "read_runtime",
    "render",
]

_MARKER = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")

_ROLE_MARKERS: dict[str, str] = {
    "KUBECTL_GUIDANCE": "kubectl",
    "MEMORY_GUIDANCE": "memory",
}
"""Marker name to the block family that fills it."""

FILLABLE_MARKERS: frozenset[str] = frozenset(_ROLE_MARKERS)
"""Markers the loader can fill. A role prompt declaring any other raises ``PromptError``
at composition, and ``validate_roles`` raises it at startup before any incident."""

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

# The same question under the route where a role runs kubectl itself, through a
# credential-free proxy on PATH, rather than through the cluster tool. Three blocks
# rather than four: the archived deployment gave Planning the read-only text, not a
# Planning-specific one, and the composed prompts it ran under say so.
_KUBECTL_PROXY_BLOCK: dict[str, str] = {
    "rca": "proxy_exec.read_only",
    "planning": "proxy_exec.read_only",
    "execution": "proxy_exec.execution",
    "verify": "proxy_exec.read_only",
    "learning": "proxy_exec.read_only",
    "single": "proxy_exec.single",
}

_KUBECTL_ROUTE_BLOCKS: dict[str, dict[str, str]] = {
    "tool": _KUBECTL_BLOCK,
    "proxy_exec": _KUBECTL_PROXY_BLOCK,
}
"""How a role reaches the cluster, and therefore which guidance it is given.

``tool`` is the cluster tool with the sandbox refusing kubectl; ``proxy_exec`` is a real
kubectl on the role's PATH, pointed at a credential-free proxy and screened per stage by
the shim it resolves to. The two are mutually exclusive and a deployment is in exactly
one of them, decided by whether its bound transport offers a proxy URL — which is what
:func:`kira.tools.context.kubectl_route` answers. Keyed by the route name rather than
guarded by a separate set of them, so a route this cannot fill is not a route it claims
to know.
"""


class PromptError(RuntimeError):
    """A prompt asset is missing, or a template and its values disagree."""


PROMPT_PACKAGE = "kira.prompts"
"""Where a prompt asset is read from unless the caller names another package."""


@cache
def read_asset(relative_path: str, *, package: str = PROMPT_PACKAGE) -> str:
    """Read a prompt asset shipped inside a package.

    Args:
        relative_path: Path under the package, using forward slashes.
        package: The package the path is anchored at. An environment adapter ships its
            own prompts inside its own package and reads them through this function
            rather than a second one, so the containment check above and the strict
            rendering below apply to them exactly as they do to the core's.

    Returns:
        The asset's text.

    Raises:
        PromptError: When the path leaves the package, or the asset does not exist.
    """
    resource = files(package)
    for part in relative_path.split("/"):
        # Containment lives here rather than in the three readers layered above,
        # because this is the one they all funnel through and the one a third-party
        # RoleSpec.prompt reaches: an escaping name is read straight into a system
        # prompt.
        if not part or part == ".." or part.startswith("."):
            raise PromptError(f"prompt path escapes {package}: {relative_path!r}")
        resource = resource / part
    try:
        # Stripped on load, matching the prototype's own loader. Every asset ships with
        # a trailing newline, and a block substituted into a template unstripped adds a
        # blank line the measured prompts did not have. Composition is a held-constant
        # input, so this single .strip() is load-bearing rather than cosmetic.
        return resource.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise PromptError(f"prompt asset not found: {relative_path}") from exc


def read_runtime(role: str, name: str) -> str:
    """Read a per-turn runtime asset belonging to a role.

    Runtime assets are the blocks a role emits into the conversation rather than into
    its system prompt: task statements, re-entry headers, conditional context blocks
    and the validator rejections returned when a submission is malformed. They live
    under ``prompts/runtime/<role>/`` and are read, then filled with :func:`render`.

    Args:
        role: The owning role, or ``"shared"`` for a block several roles emit.
        name: The asset's file name, including its suffix.

    Returns:
        The asset's text.

    Raises:
        PromptError: When either argument tries to traverse, or the asset is missing.
    """
    for part in (role, name):
        if not part or "/" in part or "\\" in part or part.startswith("."):
            raise PromptError(f"invalid runtime prompt path: {role}/{name}")
    return read_asset(f"runtime/{role}/{name}")


def _roles_dir(directory: Path) -> Path:
    """Resolve the override directory the one way every reader of it must.

    A ``~`` is written literally in a ConfigMap and a compose ``environment:`` block,
    and pydantic keeps it. Expanded by the digester and not by the reader, the snapshot
    recorded overrides that never ran.

    Args:
        directory: The configured override directory.

    Returns:
        The directory holding one file per replaced role.
    """
    return directory.expanduser() / "roles"


def override_for(role: str, directory: Path | None) -> str | None:
    """Read a deployment's replacement for one role's prompt, if it supplied one.

    A role with no file in the directory keeps the packaged prompt, so a deployment can
    replace one prompt and inherit the rest rather than having to fork the whole set.

    Args:
        role: The role whose prompt is being composed.
        directory: The configured override directory, or None.

    Returns:
        The replacement template, or None to use the packaged asset.

    Raises:
        PromptError: When the role name would escape the directory, or the file is
            there and cannot be used. A file that is absent is not an error; a file
            that is present and unreadable, or empty, is, because silently falling back
            to the packaged prompt would run something other than what the deployment
            asked for and record no sign of it — and an empty one does not even fall
            back: it sends the role no system prompt at all.
    """
    if directory is None:
        return None
    if not role or "/" in role or "\\" in role or role.startswith("."):
        raise PromptError(f"invalid role name for a prompt override: {role!r}")
    path = _roles_dir(directory) / f"{role}.txt"
    if not path.exists():
        return None
    template = read_override(path, root=directory)
    if not template.strip():
        raise PromptError(
            f"prompt override {path} is empty; a template rendered to nothing would "
            "leave the role with no system prompt"
        )
    return template


def overridden_roles(directory: Path | None) -> dict[str, str]:
    """Digest every role prompt a deployment replaced.

    For the run's configuration snapshot. A result produced under a prompt nobody
    recorded is not attributable, and the directory path alone does not say what was in
    it at the time.

    Args:
        directory: The configured override directory, or None.

    Returns:
        Role name to the SHA-256 of its replacement, empty when nothing is overridden.
    """
    if directory is None:
        return {}
    roles = _roles_dir(directory)
    if not roles.is_dir():
        return {}
    digests: dict[str, str] = {}
    for path in sorted(roles.glob("*.txt")):
        try:
            digests[path.stem] = sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PromptError(f"cannot read prompt override {path}: {exc}") from exc
    return digests


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


def _block_for(marker: str, role: str, *, retrieval_enabled: bool, kubectl_route: str) -> str:
    family = _ROLE_MARKERS.get(marker)
    if family == "kubectl":
        variant = _KUBECTL_ROUTE_BLOCKS[kubectl_route].get(role)
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
    override_dir: Path | None = None,
    kubectl_route: str = "tool",
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
        override_dir: Where a deployment keeps its own role prompts, or None. A role
            with no file there keeps the packaged one.
        kubectl_route: How this deployment's roles reach the cluster, as
            :func:`kira.tools.context.kubectl_route` answers it. The default is the cluster tool, which is what an
            unbound run and every direct-transport deployment uses; the caller passes
            ``proxy_exec`` when the bound transport offers a proxy the roles run their
            own kubectl against, because the two routes prescribe opposite things about
            typing kubectl into the sandbox.

    Returns:
        The fully rendered system prompt.

    Raises:
        PromptError: When an asset is missing, the route is not a known one, or a
            marker cannot be filled.
    """
    override = override_for(role, override_dir)
    template = override if override is not None else read_asset(f"roles/{role}.txt")
    if kubectl_route not in _KUBECTL_ROUTE_BLOCKS:
        raise PromptError(
            f"unknown kubectl route {kubectl_route!r}; "
            f"expected one of {sorted(_KUBECTL_ROUTE_BLOCKS)}"
        )
    values = {
        marker: _block_for(
            marker, role, retrieval_enabled=retrieval_enabled, kubectl_route=kubectl_route
        )
        for marker in markers(template)
    }
    return render(template, values)
