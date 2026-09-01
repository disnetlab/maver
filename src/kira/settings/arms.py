"""Experiment arms: named, frozen overlays on settings, with provenance.

An arm is the unit a published result is measured on. It pins a model profile by name
so the pairing is explicit, deep-merges a partial settings tree over the defaults, and
records why it exists and when it was registered. It never contains an endpoint or a
secret.

Two properties make an arm file worth more than a shell script full of exports. An arm
marked frozen has had a published number measured on it, so changing the file changes
what that number means and continuous integration should refuse the diff. And an arm
declares what it holds constant, so comparing two arms can assert that only one thing
varies rather than trusting that it did.
"""

from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Arm",
    "Registration",
    "arm_digest",
    "compare_constants",
    "deep_merge",
    "load_arm",
]


class Registration(BaseModel):
    """When and why an arm was created."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    campaign: str
    hypothesis: str


class Arm(BaseModel):
    """A named configuration a result can be attributed to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    name: str
    label: str = Field(description="The name this configuration is published under.")
    description: str
    model_profile: str
    frozen: bool = Field(
        default=False,
        description=(
            "True once a published number was measured on this arm. A change to a "
            "frozen arm changes what that number means."
        ),
    )
    registered: Registration
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="A partial settings tree, deep-merged over the defaults.",
    )
    holds_constant: list[str] = Field(
        default_factory=list,
        description=(
            "Dotted setting paths, wildcards allowed, that a comparison against "
            "another arm must find identical."
        ),
    )
    notes: str = ""

    @model_validator(mode="after")
    def _label_and_name_are_both_present(self) -> Self:
        """Reject an arm that cannot be cited.

        Returns:
            The validated arm.

        Raises:
            ValueError: When the arm has no name or no label.
        """
        if not self.name.strip() or not self.label.strip():
            raise ValueError("an arm needs both a name and a published label")
        return self

    def apply_to(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Overlay this arm's settings on a default settings tree.

        Args:
            defaults: The settings tree the arm modifies.

        Returns:
            A new tree with the arm's values merged in.
        """
        return deep_merge(defaults, self.settings)

    def is_held_constant(self, path: str) -> bool:
        """Report whether a dotted setting path is one this arm claims not to vary.

        Args:
            path: A dotted setting path.

        Returns:
            True when any held-constant pattern matches.
        """
        return any(fnmatch(path, pattern) for pattern in self.holds_constant)

    def contradicted_by(self, resolved: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
        """Find settings this arm declares that the resolved configuration does not carry.

        The environment outranks an arm file by design, so a difference here is not by
        itself a fault: an operator overriding one key without editing a frozen file is
        the deployment contract working. It is a fault on a path the arm claims to hold
        constant, because that claim is what licenses reading a two-arm gap as
        single-variable, and nothing else checks it.

        Args:
            resolved: The settings tree the process actually resolved.

        Returns:
            Dotted path to the declared and resolved pair, empty when the file applied
            as written.
        """
        actual = _flatten(resolved)
        return {
            path: (declared, actual[path])
            for path, declared in _flatten(self.settings).items()
            if path in actual and actual[path] != declared
        }

    def constant_values(self, resolved: dict[str, Any]) -> dict[str, Any]:
        """Extract the settings this arm claims to hold constant.

        Args:
            resolved: The arm's fully resolved settings tree.

        Returns:
            Dotted path to value, for every path matching a held-constant pattern.
        """
        flat = _flatten(resolved)
        flat["model_profile"] = self.model_profile
        return {path: value for path, value in flat.items() if self.is_held_constant(path)}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge one mapping into another, recursing into nested mappings.

    Args:
        base: The mapping being overlaid.
        overlay: Values that win where they are present.

    Returns:
        A new mapping. Neither argument is modified.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Render a nested mapping as dotted paths.

    Args:
        tree: The mapping to flatten.
        prefix: Path accumulated from enclosing levels.

    Returns:
        Dotted path to leaf value.
    """
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def compare_constants(
    left: Arm, left_resolved: dict[str, Any], right: Arm, right_resolved: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Find settings two arms both claim to hold constant but do not.

    A paired contrast is only single-variable if everything outside the treatment
    matches. This turns that from an assumption into a check.

    Args:
        left: One arm.
        left_resolved: Its resolved settings tree.
        right: The other arm.
        right_resolved: Its resolved settings tree.

    Returns:
        Dotted path to the differing pair of values, empty when the claim holds.
    """
    a = left.constant_values(left_resolved)
    b = right.constant_values(right_resolved)
    shared = set(a) & set(b)
    return {path: (a[path], b[path]) for path in sorted(shared) if a[path] != b[path]}


def load_arm(name: str, directory: Path) -> Arm:
    """Read and validate one arm definition.

    Args:
        name: The arm's stem, matching its filename.
        directory: Where arm files live.

    Returns:
        The validated arm.

    Raises:
        FileNotFoundError: When no such arm exists.
        ValueError: When the file's declared name disagrees with its filename.
        pydantic.ValidationError: When the arm is malformed.
    """
    path = directory / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.yaml")) if directory.is_dir() else []
        raise FileNotFoundError(
            f"no arm {name!r} in {directory}" + (f"; available: {available}" if available else "")
        )
    arm = Arm.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if arm.name != name:
        raise ValueError(
            f"arm in {path.name} declares name {arm.name!r}; the filename is what a "
            "run record cites, so the two must agree"
        )
    return arm


def arm_digest(name: str, directory: Path) -> str:
    """Hash one arm file's bytes, so a frozen arm's drift is detectable after the fact.

    An arm marked frozen has a published number measured on it. Recording the digest in
    every run's configuration snapshot is what lets a later reader tell a rerun of that
    configuration from a rerun of a file that has since been edited.

    Args:
        name: The arm's stem, matching its filename.
        directory: Where arm files live.

    Returns:
        The file's SHA-256, hex encoded.

    Raises:
        OSError: When the file cannot be read.
    """
    return sha256((directory / f"{name}.yaml").read_bytes()).hexdigest()
