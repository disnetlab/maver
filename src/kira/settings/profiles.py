"""Model profiles: one file fully determines how a model is called.

A profile carries the endpoint, the provider route, the **complete** sampling set, the
per-model behavioural quirks, and the serving provenance. Every sampling field is
required: ``null`` means "omit this from the request" and never "I forgot". That
distinction is the whole point of the file. Pinning one knob and assuming the rest
followed made months of runs incomparable, and the record showed only the knob that
was read.

Sampling parameters belong to the model, not to the harness or the role. Different
endpoints accept different parameters, and several have no field in the OpenAI schema
at all, so they must be routed through the provider's native extension. Keeping them
in one global namespace is what turned a working local configuration into a hard
failure the moment a run pointed at a hosted route.

Secrets never appear in a profile. ``api_key_env`` names an environment variable;
``api_key_literal`` exists only for keyless local servers and is checked against a
small allowlist.
"""

import logging
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_log = logging.getLogger(__name__)

__all__ = [
    "EXTRA_BODY_PROVIDERS",
    "Endpoint",
    "ModelProfile",
    "Quirks",
    "SamplingSet",
    "Serving",
    "load_profile",
]

EXTRA_BODY_PROVIDERS = frozenset({"vllm", "ollama"})
"""Providers whose native sampler accepts fields absent from the OpenAI schema."""

_KEYLESS_LITERALS = frozenset({"EMPTY", "sk-no-key-required"})

_NATIVE_ONLY = ("top_k", "min_p", "repetition_penalty", "no_repeat_ngram_size")
"""Sampling knobs with no OpenAI-schema field. They must travel via extra_body."""

Provider = Literal["openai", "anthropic", "gemini", "ollama", "vllm"]
ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Endpoint(_Frozen):
    """Where the model is served and how the request is authorised."""

    base_url: str | None = Field(description="Null uses the provider's default endpoint.")
    api_key_env: str | None = Field(description="Environment variable holding the key.")
    api_key_literal: str | None = Field(
        default=None,
        description="Placeholder for a keyless local server. Never a real secret.",
    )

    @model_validator(mode="after")
    def _literal_is_a_placeholder(self) -> Self:
        """Refuse to carry anything that could be a real credential.

        Returns:
            The validated endpoint.

        Raises:
            ValueError: When the literal is not a recognised placeholder.
        """
        if self.api_key_literal is not None and self.api_key_literal not in _KEYLESS_LITERALS:
            raise ValueError(
                f"api_key_literal must be one of {sorted(_KEYLESS_LITERALS)}; a real "
                "credential belongs in the environment variable named by api_key_env"
            )
        return self


class SamplingSet(_Frozen):
    """The complete sampling configuration. Every field is required.

    ``null`` means the field is omitted from the request. Because nothing has a
    default, a profile that forgets a knob fails to load rather than silently
    inheriting whatever the provider happens to do.
    """

    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    repetition_penalty: float | None
    no_repeat_ngram_size: int | None
    seed: int | None
    reasoning_effort: ReasoningEffort | None
    thinking_token_budget: int | None

    @model_validator(mode="after")
    def _thinking_fits_within_the_cap(self) -> Self:
        """A reasoning budget at or above the cap leaves no room for an answer.

        Returns:
            The validated sampling set.

        Raises:
            ValueError: When the reasoning budget does not fit under max_tokens.
        """
        if self.thinking_token_budget is not None and self.thinking_token_budget >= self.max_tokens:
            raise ValueError(
                f"thinking_token_budget ({self.thinking_token_budget}) must be below "
                f"max_tokens ({self.max_tokens}); otherwise the generation is spent "
                "reasoning and stops before it can emit a tool call"
            )
        return self

    def request_fields(
        self, extra_body_fields: frozenset[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split the set into top-level request fields and native extension fields.

        Args:
            extra_body_fields: Knobs this provider accepts only via its extension.

        Returns:
            The top-level parameters and the extension parameters, each omitting
            anything set to null.
        """
        top: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if value is None:
                continue
            (extra if name in extra_body_fields else top)[name] = value
        return top, extra


class Quirks(_Frozen):
    """Per-model behaviours the loop has to accommodate."""

    tool_choice: Literal["required", "auto", "omit"]
    keep_reasoning: bool
    rescue_disable_thinking: bool = False
    rescue_reasoning_effort: ReasoningEffort | None = None
    drop_unsupported_params: bool = True
    extra_body_fields: frozenset[str] = frozenset()
    reports_cached_tokens: bool = Field(
        default=False,
        description=(
            "When false, a null cached-token count is expected rather than a missing "
            "measurement. Several servers run prefix caching without reporting it."
        ),
    )


class Serving(_Frozen):
    """How the weights are actually served.

    Copied verbatim into every recorded model call. Model identity is not checkable
    from the model string: a mid-campaign precision change kept the same alias and
    every assertion still passed.
    """

    runtime: str
    tool_parser: str | None = None
    dtype: str | None = None
    tensor_parallel: int | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    weights_digest: str | None = None


class ModelProfile(_Frozen):
    """Everything needed to call one model, in one file."""

    schema_version: Literal[1]
    name: str
    model: str = Field(description="The provider route passed to the client library.")
    provider: Provider
    endpoint: Endpoint
    sampling: SamplingSet
    aux_sampling: SamplingSet | None = Field(
        default=None,
        description=(
            "Sampling for short structured classifications. Derived from `sampling` "
            "when a file omits it; never absent by the time a call is made."
        ),
    )
    limits: dict[str, int] = Field(
        description="request_timeout_s and num_retries.",
    )
    quirks: Quirks
    serving: Serving
    known_issues: list[str] = Field(
        default_factory=list,
        description="Behavioural facts a reader must know before trusting a result.",
    )

    @model_validator(mode="after")
    def _auxiliary_sampling_is_always_resolved(self) -> Self:
        """Fill the auxiliary set when the file leaves it out.

        The deployment's profile is tuned for open-ended tool-use generation. A short
        structured classification wants none of it: sampling makes the same input yield
        different answers run to run, and a presence penalty distorts a fifty-token JSON
        reply. Deriving it rather than leaving it absent means every helper call is made
        under a complete, recorded set instead of under whichever knobs the primary
        profile happened to carry.

        Note this buys stability, not reproducibility: a batching server is not
        bit-deterministic at any temperature.

        Returns:
            The profile, with an auxiliary set either declared or derived.
        """
        if self.aux_sampling is None:
            object.__setattr__(self, "aux_sampling", _derive_aux(self.sampling))
            _log.info(
                "profile %s declares no aux_sampling; derived one from its primary set",
                self.name,
            )
        return self

    @model_validator(mode="after")
    def _native_knobs_match_the_route(self) -> Self:
        """Refuse a profile that would send native-only knobs to a hosted provider.

        A hosted route rejects the whole request when it receives a field outside the
        OpenAI schema, and dropping unsupported parameters does not police the native
        extension. This is the most common way a profile fails at the first call
        rather than at load.

        Returns:
            The validated profile.

        Raises:
            ValueError: When the sampling set and the route disagree.
        """
        fields = self.quirks.extra_body_fields
        if fields and self.provider not in EXTRA_BODY_PROVIDERS:
            raise ValueError(
                f"provider {self.provider!r} accepts no native extension fields, but "
                f"extra_body_fields is {sorted(fields)}; a hosted route rejects the "
                "entire request when it receives one"
            )
        for knob in _NATIVE_ONLY:
            if getattr(self.sampling, knob) is not None and knob not in fields:
                raise ValueError(
                    f"sampling.{knob} is set but is not listed in "
                    f"quirks.extra_body_fields; it has no field in the OpenAI schema, "
                    "so it would be silently dropped or rejected"
                )
        missing = {"request_timeout_s", "num_retries"} - set(self.limits)
        if missing:
            raise ValueError(f"limits is missing {sorted(missing)}")
        return self


def load_profile(name: str, directory: Path) -> ModelProfile:
    """Read and validate one model profile.

    Args:
        name: The profile's stem, matching its filename.
        directory: Where profiles live.

    Returns:
        The validated profile.

    Raises:
        FileNotFoundError: When no such profile exists.
        ValueError: When the file's declared name disagrees with its filename.
        pydantic.ValidationError: When the profile is malformed or inconsistent.
    """
    path = directory / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.yaml")) if directory.is_dir() else []
        raise FileNotFoundError(
            f"no model profile {name!r} in {directory}"
            + (f"; available: {available}" if available else "")
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = ModelProfile.model_validate(data)
    if profile.name != name:
        raise ValueError(
            f"profile in {path.name} declares name {profile.name!r}; the filename is "
            "what a run record cites, so the two must agree"
        )
    return profile


def _derive_aux(primary: SamplingSet) -> SamplingSet:
    """Build an auxiliary set from a primary one.

    Temperature to zero and every diversity or repetition knob off, because each of them
    exists to shape a long generation and each of them distorts a short one. The
    reasoning knobs go with them, for the same reason and one more: an auxiliary call has
    no tools, so its whole answer has to arrive in the content, and with thinking on the
    model spends the budget reasoning and returns nothing. That is measured — every
    selector call in one rerun parsed to empty and fell back to the first opinion — and
    :mod:`kira.secondpass.select` states it as a property of this set. It was not one.

    The seed is kept. It says which behaviour is being used rather than how widely the
    model may wander, which is the test the rest of these fail.

    Args:
        primary: The profile's open-ended set.

    Returns:
        The derived set.
    """
    return SamplingSet(
        max_tokens=primary.max_tokens,
        temperature=0.0,
        top_p=None,
        top_k=None,
        min_p=None,
        presence_penalty=None,
        frequency_penalty=None,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        seed=primary.seed,
        reasoning_effort=None,
        thinking_token_budget=None,
    )
