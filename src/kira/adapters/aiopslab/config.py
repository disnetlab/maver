"""The AIOpsLab adapter's own slice of ``Settings.adapters``.

Every AIOpsLab fact is data here rather than a constant in code. The prototype kept the
bridge's address in four accessor functions and the applications' deployment facts in a
module-level dict inside its runner, so the values a run was measured under were only
recoverable by reading the source at that commit.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from kira.settings.root import Settings

__all__ = ["ADAPTER_NAME", "MIN_TOKEN_CHARS", "AiopsLabConfig"]

_log = logging.getLogger(__name__)

ADAPTER_NAME = "aiopslab"

MIN_TOKEN_CHARS = 16
"""Shortest bridge token this adapter will start with.

The prototype accepted any non-empty string, including a truncated read of the token
file, and a truncated token disabled every cluster tool without an error: the token was
read into an empty string, the bridge counted as unconfigured, and the roles were handed
a toolset with nothing in it. A length floor turns that into a startup failure.
"""

CONNECTIVITY_CONTRACTS: dict[str, dict[str, Any]] = {
    # Static deployment facts for the benchmark applications. These are deliberately
    # keyed only by namespace and contain no task, fault, or diagnosis information:
    # Verify needs a reliable user-facing Service to probe, but must remain blind to
    # the incident's causal answer.
    "test-social-network": {
        "application": "Social Network",
        "namespace": "test-social-network",
        "http_entry_point": {"service": "nginx-thrift", "port": 8080, "path": "/"},
    },
    "test-hotel-reservation": {
        "application": "Hotel Reservation",
        "namespace": "test-hotel-reservation",
        "http_entry_point": {"service": "frontend", "port": 5000, "path": "/"},
    },
    "astronomy-shop": {
        "application": "OpenTelemetry Demo Astronomy Shop",
        "namespace": "astronomy-shop",
        "http_entry_point": {"service": "frontend-proxy", "port": 8080, "path": "/"},
    },
}
"""What Verify is told to probe, per namespace. Carried verbatim from the prototype."""


class AiopsLabConfig(BaseModel):
    """How this process reaches the AIOpsLab driver, and what it knows about the apps.

    Unknown keys are refused rather than ignored, so a typo inside the slice is a
    startup failure instead of a default that quietly changes what an arm means.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    """Whether this process is serving an AIOpsLab run and therefore needs the bridge.

    Distinct from being selected in ``intake.adapters``, which is what binds the adapter
    at all. This is what makes the bridge credential mandatory: without it a deployment
    that meant to run under AIOpsLab and mistyped the token path would start, lose every
    cluster tool, and report a model that could not investigate."""

    bridge_url: str = "http://127.0.0.1:1819"
    """Where the driver serves the scoped action bridge. Loopback by default, because
    the agent and the driver share a host and the bridge is not a network service."""
    bridge_token: SecretStr | None = None
    bridge_token_file: Path | None = None
    """Read once, at startup. The resolved value lands in :attr:`bridge_token`."""
    bridge_timeout_s: float = Field(default=120.0, ge=1.0)

    detection_max_turns: int = Field(default=10, ge=1)
    """The turn budget the detection task's diagnosing run is narrowed to."""
    detection_dropped_tools: frozenset[str] = frozenset(
        {"bash_exec", "code_exec", "get_operations"}
    )
    """What the detection task withholds from the diagnosing role."""

    connectivity_contracts: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in CONNECTIVITY_CONTRACTS.items()}
    )
    """Namespace to its user-facing entry point. Deployment metadata, not evidence."""

    mongodb_services: frozenset[str] = frozenset({"mongodb-rate", "mongodb-geo"})
    mongodb_access_repairs: frozenset[str] = frozenset({"privileges", "user"})
    """The two axes of the MongoDB access repair the remediating role is offered."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_token(cls, values: Any) -> Any:
        """Read the bridge token from wherever it was supplied, before anything runs.

        Explicit wins over a file, because a deployment that names both means the one
        it typed. A file that cannot be read, or holds less than
        :data:`MIN_TOKEN_CHARS`, is a startup failure naming the path rather than an
        empty token: the prototype swallowed the read error and returned ``""``, which
        made the bridge count as unconfigured and silently removed every cluster tool
        from every role, and the run then burned its whole turn budget.

        Args:
            values: The raw slice.

        Returns:
            The slice with ``bridge_token`` resolved.

        Raises:
            ValueError: When this process is serving an AIOpsLab run and no usable
                token could be resolved.
        """
        if not isinstance(values, dict):
            return values
        resolved = dict(values)
        enabled = bool(resolved.get("enabled", False))
        token = resolved.get("bridge_token")
        secret = token.get_secret_value() if isinstance(token, SecretStr) else token
        path = resolved.get("bridge_token_file")

        if not (secret or "").strip() and path:
            secret = _read_token(Path(path), enabled=enabled)
        secret = (secret or "").strip()

        if len(secret) < MIN_TOKEN_CHARS:
            where = f" from {path}" if path and not token else ""
            if enabled:
                raise ValueError(
                    f"the aiopslab bridge token{where} is "
                    f"{'missing' if not secret else f'{len(secret)} characters'}; at "
                    f"least {MIN_TOKEN_CHARS} are required. A short or unreadable token "
                    "disables every cluster tool without an error, so this is refused "
                    "at startup instead."
                )
            secret = ""
        resolved["bridge_token"] = SecretStr(secret) if secret else None
        return resolved

    @classmethod
    def resolve(cls, settings: Settings, env: Mapping[str, str] | None = None) -> "AiopsLabConfig":
        """Build the configuration, honouring the names a prototype deployment uses.

        Args:
            settings: The resolved settings, whose ``adapters`` slice takes precedence.
            env: Environment to read legacy names from. Defaults to the process's.

        Returns:
            The validated configuration.

        Raises:
            pydantic.ValidationError: When the slice carries an unknown or malformed
                key, or the bridge token cannot be resolved.
        """
        environ = os.environ if env is None else env
        values = dict(settings.adapters.get(ADAPTER_NAME, {}))
        for field, value in _legacy_values(environ).items():
            values.setdefault(field, value)
        return cls.model_validate(values)


def _read_token(path: Path, *, enabled: bool) -> str:
    """Read the token file, failing loudly where the prototype failed silently.

    Args:
        path: The file to read.
        enabled: Whether this process is serving an AIOpsLab run.

    Returns:
        The file's contents, stripped, or an empty string when the adapter is not
        serving a run and the file is unreadable.

    Raises:
        ValueError: When the file cannot be read and the bridge is required.
    """
    try:
        return path.expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        if enabled:
            raise ValueError(
                f"the aiopslab bridge token file {path} cannot be read: {exc}"
            ) from exc
        _log.warning("the aiopslab bridge token file %s cannot be read: %s", path, exc)
        return ""


def _legacy_values(env: Mapping[str, str]) -> dict[str, Any]:
    """Read the prototype's own names for this adapter's settings.

    Each is reported once, like every other legacy name, so an operator can see which
    spelling decided a value.

    Args:
        env: The environment to read.

    Returns:
        Field name to value, for the names that are actually set.
    """
    found: dict[str, Any] = {}
    for name, field in (
        ("AIOPSLAB_BRIDGE_URL", "bridge_url"),
        ("AIOPSLAB_BRIDGE_TOKEN", "bridge_token"),
        ("AIOPSLAB_BRIDGE_TOKEN_FILE", "bridge_token_file"),
        ("AIOPSLAB_BRIDGE_TIMEOUT_SECONDS", "bridge_timeout_s"),
    ):
        value = env.get(name, "").strip()
        if value:
            found[field] = value
            _log.info("legacy name in use: %s -> adapters.aiopslab.%s", name, field)
    return found
