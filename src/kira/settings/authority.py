"""Who may authorise a write, and under what conditions.

Two defaults differ from the prototype, both deliberately. The mode defaults to
``prompt`` rather than ``auto``: the prototype shipped automatic approval on a service
holding cluster write permissions, and an artifact other people deploy should not. And
the timeout has a real default, so a pending request cannot park a session forever.

An evaluation harness that wants unattended behaviour sets ``mode: auto`` explicitly,
which every existing run script already does — and which then lands in the run's
configuration snapshot, so the results say what authority they were produced under
instead of leaving it to be inferred.
"""

from fnmatch import fnmatch
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from kira.settings.base import NameSet, SettingsGroup, group_config, setting

__all__ = ["AuthorityRule", "AuthoritySettings", "Mode"]

Mode = Literal["auto", "prompt", "deny"]


class AuthorityRule(SettingsGroup):
    """One policy override, matched against a resolved tool call.

    Rules exist because a deployment's real policy is rarely uniform. It wants routine
    scaling to proceed unattended while a human sees anything touching a secret, and
    neither a global mode nor a list of exempt tool names can say that.
    """

    model_config = SettingsConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Stable identifier, recorded in the audit event.")
    tools: tuple[str, ...] = ("*",)
    """Globs matched against the tool name."""
    stages: tuple[str, ...] = ("*",)
    """Globs matched against the calling stage."""
    selectors: tuple[str, ...] = ()
    """Globs that must each match at least one of the call's selectors.

    Empty matches every call. This is what expresses "anything touching a secret"
    without the gate having to know what a secret is.
    """
    mode: Mode
    on_timeout: Literal["deny", "approve"] | None = None
    """What an expired request resolves to.

    Only ever set per rule, never globally. Approving on expiry is a footgun, and making
    an operator name the narrow class of actions that are safe unattended is the whole
    safety property. An unmatched call always denies on timeout.
    """

    def matches(self, *, tool: str, stage: str, selectors: tuple[str, ...]) -> bool:
        """Report whether this rule governs a call.

        Args:
            tool: The tool name.
            stage: The calling stage.
            selectors: The call's own selectors.

        Returns:
            True when the rule applies.
        """
        if not any(fnmatch(tool, pattern) for pattern in self.tools):
            return False
        if not any(fnmatch(stage, pattern) for pattern in self.stages):
            return False
        return all(
            any(fnmatch(selector, pattern) for selector in selectors) for pattern in self.selectors
        )


class AuthoritySettings(SettingsGroup):
    """The approval gate's configuration. Fails closed on every axis."""

    model_config = group_config("authority")

    mode: Mode = setting(
        "prompt",
        "KIRA_AUTHORITY__MODE",
        "HITL_MODE",
        description="Approve writes automatically, ask a human, or refuse them all.",
    )
    rules: tuple[AuthorityRule, ...] = setting(
        (),
        "KIRA_AUTHORITY__RULES",
        description="Ordered policy overrides. First match wins; the rest fall to mode.",
    )
    timeout_s: int | None = setting(
        1800,
        "KIRA_AUTHORITY__TIMEOUT_S",
        "HITL_TIMEOUT_SECONDS",
        description="How long a request waits for a human. None waits forever.",
    )
    poll_interval_s: int = setting(
        15,
        "KIRA_AUTHORITY__POLL_INTERVAL_S",
        description="How often a decision made elsewhere is noticed.",
    )
    additional_write_stages: NameSet = setting(
        frozenset(),
        "KIRA_AUTHORITY__ADDITIONAL_WRITE_STAGES",
        description=(
            "Stages allowed to write beyond each tool's own declared set. Widening this "
            "is how a diagnostic role gets write access, so it is deliberately explicit."
        ),
    )
    webhook_url: AnyHttpUrl | None = setting(
        None,
        "KIRA_AUTHORITY__WEBHOOK_URL",
        description="Notified when a request is raised. Failure never blocks the pause.",
    )
    webhook_timeout_s: float = setting(
        5.0,
        "KIRA_AUTHORITY__WEBHOOK_TIMEOUT_S",
        description="How long to wait on the notification before giving up on it.",
    )

    @field_validator("timeout_s")
    @classmethod
    def _timeout_is_a_wait(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError(
                "authority.timeout_s must be at least 1 second; a request that expires "
                "the instant it is raised is never seen by the human it is raised for. "
                "Waiting forever is spelled null."
            )
        return value

    @model_validator(mode="after")
    def _rule_names_are_unique(self) -> "AuthoritySettings":
        """Keep rule names distinct, since they are what an audit event records.

        Returns:
            The validated group.

        Raises:
            ValueError: When two rules share a name.
        """
        names = [rule.name for rule in self.rules]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"authority rule names must be unique; repeated: {duplicates}")
        return self

    def rule_for(
        self, *, tool: str, stage: str, selectors: tuple[str, ...]
    ) -> AuthorityRule | None:
        """Find the rule governing one call.

        Args:
            tool: The tool name.
            stage: The calling stage.
            selectors: The call's own selectors.

        Returns:
            The first matching rule, or None to fall back to ``mode``.
        """
        for rule in self.rules:
            if rule.matches(tool=tool, stage=stage, selectors=selectors):
                return rule
        return None
