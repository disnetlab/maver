"""Sampling resolution, and the assertions that keep a pinned set pinned.

The failure this module exists to prevent: a configuration change moved one sampling
knob and left the rest at their previous values, the run record showed only the knob
that was read, and every result on either side of the change became incomparable
without anyone noticing for weeks.

The defence is that the whole set is resolved in one place, asserted at startup, and
recorded together on every call. A partial record is indistinguishable from a complete
one after the fact, so there is no such thing as recording "the parameter that changed".
"""

import hashlib
import json
import logging
from typing import Any

from maver.settings.profiles import EXTRA_BODY_PROVIDERS, ModelProfile, SamplingSet
from maver.settings.root import Settings

__all__ = [
    "SamplingResolution",
    "applied_sampling",
    "assert_sampling_is_pinned",
    "is_token_limit_finish",
    "resolve_sampling",
    "sampling_digest",
]

_log = logging.getLogger(__name__)

_TOKEN_LIMIT_FINISHES = frozenset({"length", "max_tokens", "max_token", "max_tokens_exceeded"})


def is_token_limit_finish(reason: str | None) -> bool:
    """Report whether a generation stopped because it hit its cap.

    Defined once and imported by both the client and the runtime-control hook. The
    prototype had this function byte-identical in two modules on opposite sides of the
    truncation contract, which is a divergence waiting to happen.

    Args:
        reason: The provider's finish reason.

    Returns:
        True when the generation was cut off at the cap.
    """
    if reason is None:
        return False
    return str(reason).strip().lower() in _TOKEN_LIMIT_FINISHES


class SamplingResolution:
    """A profile's sampling set after any overrides, split for the request."""

    __slots__ = ("extra_body", "overrides", "profile_name", "top_level", "values")

    def __init__(
        self,
        profile_name: str,
        values: dict[str, Any],
        top_level: dict[str, Any],
        extra_body: dict[str, Any],
        overrides: dict[str, Any],
    ) -> None:
        """Hold a resolved sampling configuration.

        Args:
            profile_name: The profile the values came from.
            values: The complete set, including nulls.
            top_level: Fields sent as ordinary request parameters.
            extra_body: Fields routed through the provider's native extension.
            overrides: Knobs displaced from the profile, empty in the normal case.
        """
        self.profile_name = profile_name
        self.values = values
        self.top_level = top_level
        self.extra_body = extra_body
        self.overrides = overrides

    @property
    def digest(self) -> str:
        """A short stable hash of the complete set.

        Returns:
            Twelve hex characters, enough to compare two runs at a glance.
        """
        return sampling_digest(self.values)

    def as_record(self) -> dict[str, Any]:
        """Render the set for the audit event.

        Returns:
            The complete set plus its digest and any overrides, so a reader can tell a
            profile run from an overridden one without consulting anything else.
        """
        return {
            "profile": self.profile_name,
            "digest": self.digest,
            "overrides": sorted(self.overrides),
            **self.values,
        }


def sampling_digest(values: dict[str, Any]) -> str:
    """Hash a sampling set so two runs can be compared by one string.

    Args:
        values: The complete sampling set.

    Returns:
        Twelve hex characters.
    """
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def resolve_sampling(profile: ModelProfile, settings: Settings) -> SamplingResolution:
    """Merge any environment overrides over a profile and split for the request.

    Args:
        profile: The model profile, which is the source of the complete set.
        settings: Settings carrying any per-knob overrides.

    Returns:
        The resolved sampling configuration.
    """
    values = profile.sampling.model_dump()
    overrides = {k: v for k, v in settings.sampling_overrides().items() if k in values}
    if overrides:
        # Warn rather than fail. A run script exporting a flat name is the reason
        # overrides exist, and refusing it would break exactly the deployments this
        # compatibility layer is for. What must never happen is that it goes unrecorded.
        _log.warning(
            "sampling override(s) displace profile %r: %s. The merged set is recorded "
            "on every call; prefer a new profile file over an override.",
            profile.name,
            ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items())),
        )
        values.update(overrides)

    extra_fields = profile.quirks.extra_body_fields
    top_level: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            continue
        (extra_body if name in extra_fields else top_level)[name] = value
    return SamplingResolution(profile.name, values, top_level, extra_body, overrides)


def split_sampling(sampling: SamplingSet, profile: ModelProfile) -> SamplingResolution:
    """Split an explicit sampling set for the request, without consulting settings.

    Used for an auxiliary call that supplies its own set — a quality gate, a nominator,
    a selector — so it runs on what it asked for rather than on the agent's profile.
    Environment overrides deliberately do not apply: they exist so a run script can
    displace the *agent's* sampling, and letting them reach a gate would make the gate's
    determinism depend on a variable set for something else.

    Args:
        sampling: The complete set to apply.
        profile: The profile, consulted only for which fields the provider needs routed
            through its native extension.

    Returns:
        The resolved sampling configuration.
    """
    values = sampling.model_dump()
    extra_fields = profile.quirks.extra_body_fields
    top_level: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            continue
        (extra_body if name in extra_fields else top_level)[name] = value
    return SamplingResolution(profile.name, values, top_level, extra_body, {})


def applied_sampling(
    request_sampling: SamplingSet | None,
    profile: ModelProfile,
    resolved: SamplingResolution,
) -> SamplingResolution:
    """Which set this one call actually runs on.

    One rule, imported by the client that sends the call and by the wrapper that records
    it. Written out separately in each, they disagreed in both directions at once: the
    client reported the agent's sampling for every auxiliary call, and the recorder
    reported the profile file for every agent call, missing any override. Two logs, two
    numbers, and neither of them the set that was sent.

    Args:
        request_sampling: The set this call carries, when it brought its own.
        profile: The profile, for which fields the provider needs routed natively.
        resolved: The agent's set, already resolved against settings.

    Returns:
        The resolution that applies to this call.
    """
    if request_sampling is None:
        return resolved
    return split_sampling(request_sampling, profile)


def assert_sampling_is_pinned(profile: ModelProfile, resolution: SamplingResolution) -> None:
    """Check at startup what would otherwise fail at the first call, or not at all.

    Args:
        profile: The profile in use.
        resolution: Its resolved sampling configuration.

    Raises:
        ValueError: When the resolved configuration cannot be sent to this provider.
    """
    missing = set(profile.sampling.model_dump()) - set(resolution.values)
    if missing:
        raise ValueError(
            f"resolved sampling set is incomplete, missing {sorted(missing)}; the whole "
            "set is pinned together or not at all"
        )
    if resolution.extra_body and profile.provider not in EXTRA_BODY_PROVIDERS:
        raise ValueError(
            f"provider {profile.provider!r} accepts no native extension fields, but "
            f"{sorted(resolution.extra_body)} would be routed there. Dropping "
            "unsupported parameters does not police the native extension, so this "
            "would reach the provider and fail the whole request."
        )
    budget = resolution.values.get("thinking_token_budget")
    cap = resolution.values.get("max_tokens")
    if budget is not None and cap is not None and budget >= cap:
        raise ValueError(
            f"thinking_token_budget ({budget}) must be below max_tokens ({cap}); "
            "otherwise the generation is spent reasoning and stops before it can emit "
            "a tool call"
        )
    _log.info(
        "sampling pinned for profile %s: digest=%s%s",
        profile.name,
        resolution.digest,
        f" overrides={sorted(resolution.overrides)}" if resolution.overrides else "",
    )
