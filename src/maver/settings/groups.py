"""The settings groups.

Ten groups replace the prototype's 1,272-line module of 105 bare accessor functions
over ``os.environ``, plus roughly 55 more variables read directly elsewhere. Of those
105, 47 existed only to ablate something; that ratio, not the line count, is what made
the original unreadable.

Every field names the modern ``MAVER_GROUP__FIELD`` variable and any flat name the
prototype used, so the existing evaluation run scripts keep working unchanged. Defaults
fail closed.
"""

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, AnyHttpUrl, Field, field_validator, model_validator

from maver.settings.base import Credential, NameSet, SettingsGroup, group_config, setting

__all__ = [
    "IntakeSettings",
    "KafkaSettings",
    "MemorySettings",
    "ModelSettings",
    "ObservabilitySettings",
    "PipelineSettings",
    "PromptSettings",
    "RuntimeSettings",
    "SecondPassDelivery",
    "SecondPassSettings",
    "ServerSettings",
    "ToolSettings",
    "TransportSettings",
    "VerifySettings",
]

SecondPassDelivery = Literal["context", "assignment", "none"]


class PipelineSettings(SettingsGroup):
    """The graph, its shape, and its retry budgets."""

    model_config = group_config("pipeline")

    mode: Literal["multi", "single"] = setting(
        "multi",
        "MAVER_PIPELINE__MODE",
        "AGENT_MODE",
        description="Three typed roles, or one goal-only agent over a single transcript.",
    )
    engine: Literal["langgraph", "sequential"] = setting(
        "langgraph",
        "MAVER_PIPELINE__ENGINE",
        description=(
            "Which executor walks the routing. Both run the same nodes and the same "
            "predicates; sequential imports no framework and exists so the containment "
            "claim is a supported configuration rather than a fixture."
        ),
    )
    max_turns: int = setting(
        100,
        "MAVER_PIPELINE__MAX_TURNS",
        "MAX_TURNS",
        description=(
            "Model/tool turns a role may take before its run is ended. One hundred is "
            "what every measured campaign ran: sixteen of sixteen archived environments "
            "set it, and none set anything else. The prototype's default was twenty, "
            "which its own deployment overlay raised on every host — so twenty was a "
            "value no result was ever produced under."
        ),
    )
    remediation_retries: int = setting(
        2,
        "MAVER_PIPELINE__REMEDIATION_RETRIES",
        "MAX_REMEDIATION_RETRIES",
        "MAX_REASSESSMENTS",
        description=(
            "One budget shared by every back-edge, so replan and re-diagnose cycles "
            "cannot compound into independent budgets."
        ),
    )
    auto_snapshot: bool = setting(
        True,
        "MAVER_PIPELINE__AUTO_SNAPSHOT",
        "RCA_AUTO_SNAPSHOT",
        # A deploy script sets RCA_AUTO_SNAPSHOT_ENABLED while the prototype reads
        # RCA_AUTO_SNAPSHOT, so asking for the snapshot to be off left it on and
        # nobody noticed. Accepting both names is the compatible fix.
        "RCA_AUTO_SNAPSHOT_ENABLED",
        description="Attach a namespace overview to the diagnosing role's first message.",
    )
    reassess_mode: Literal["fresh", "talkback"] = setting(
        "fresh",
        "MAVER_PIPELINE__REASSESS_MODE",
        "RCA_REASSESS_MODE",
        description="Whether a re-diagnosis starts clean or carries the prior transcript.",
    )
    talkback_max_chars: int = setting(
        400_000,
        "MAVER_PIPELINE__TALKBACK_MAX_CHARS",
        "RCA_TALKBACK_MAX_CHARS",
        description="Ceiling on a carried-forward transcript.",
    )

    @field_validator("max_turns")
    @classmethod
    def _turns_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_turns must be at least 1")
        return value

    @field_validator("remediation_retries", "talkback_max_chars")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must not be negative")
        return value


class SamplingOverrides(SettingsGroup):
    """Per-knob overrides of the model profile's sampling set.

    These exist so an existing run script exporting a flat ``LLM_*`` name keeps
    working. Every override is warned about at startup and recorded in the run's
    config snapshot, because a partial, unrecorded sampling change is precisely the
    failure that made months of runs incomparable: temperature moved and nothing else
    did, and the record showed only the knob that was read.

    Prefer a new profile file over an override.
    """

    model_config = group_config("model__overrides")

    model: str | None = setting(
        None, "MAVER_MODEL__OVERRIDES__MODEL", "LLM_MODEL", description="Provider route."
    )
    max_tokens: int | None = setting(
        None, "MAVER_MODEL__OVERRIDES__MAX_TOKENS", "LLM_MAX_TOKENS", description="Generation cap."
    )
    temperature: float | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__TEMPERATURE",
        "LLM_TEMPERATURE",
        description="Sampling temperature.",
    )
    top_p: float | None = setting(
        None, "MAVER_MODEL__OVERRIDES__TOP_P", "LLM_TOP_P", description="Nucleus sampling mass."
    )
    top_k: int | None = setting(
        None, "MAVER_MODEL__OVERRIDES__TOP_K", "LLM_TOP_K", description="Top-k cutoff."
    )
    min_p: float | None = setting(
        None, "MAVER_MODEL__OVERRIDES__MIN_P", "LLM_MIN_P", description="Minimum token probability."
    )
    presence_penalty: float | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__PRESENCE_PENALTY",
        "LLM_PRESENCE_PENALTY",
        description="Presence penalty.",
    )
    frequency_penalty: float | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__FREQUENCY_PENALTY",
        "LLM_FREQUENCY_PENALTY",
        description="Frequency penalty.",
    )
    repetition_penalty: float | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__REPETITION_PENALTY",
        "LLM_REPETITION_PENALTY",
        description="Repetition penalty.",
    )
    no_repeat_ngram_size: int | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__NO_REPEAT_NGRAM_SIZE",
        "LLM_NO_REPEAT_NGRAM_SIZE",
        description="Banned repeat n-gram length.",
    )
    seed: int | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__SEED",
        "LLM_SEED",
        description="Sampling seed, where the server honours one.",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__REASONING_EFFORT",
        "LLM_REASONING_EFFORT",
        description="Reasoning effort, where supported.",
    )
    thinking_token_budget: int | None = setting(
        None,
        "MAVER_MODEL__OVERRIDES__THINKING_TOKEN_BUDGET",
        "LLM_THINKING_TOKEN_BUDGET",
        description="Reasoning-channel budget.",
    )

    def active(self) -> dict[str, object]:
        """Return only the knobs actually set.

        Returns:
            Field name to value, for every override that is not None.
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ModelSettings(SettingsGroup):
    """Which model profile to use, and the narrow escapes from it."""

    model_config = group_config("model")

    profile: str = setting(
        "qwen3.6-35b-a3b",
        "MAVER_MODEL__PROFILE",
        description="Stem of a file under the profile directory.",
    )
    profile_dir: Path = setting(
        Path("configs/models"),
        "MAVER_MODEL__PROFILE_DIR",
        description="Where model profiles are read from.",
    )
    endpoint_override: str | None = setting(
        None,
        "MAVER_MODEL__ENDPOINT_OVERRIDE",
        "OPENAI_API_BASE",
        description="Replaces the profile's base URL. For pointing a run at another server.",
    )
    api_key: Credential = setting(
        None,
        "MAVER_MODEL__API_KEY",
        description="Overrides the key named by the profile's api_key_env.",
    )
    overrides: SamplingOverrides = Field(default_factory=SamplingOverrides)
    request_timeout_s: int = setting(
        1800,
        "MAVER_MODEL__REQUEST_TIMEOUT_S",
        "LLM_TIMEOUT_SECONDS",
        description=(
            "Per-request ceiling. The prototype's 300 s default abandoned long "
            "generations that were still running, so the work was paid for and thrown away."
        ),
    )
    num_retries: int = setting(
        3, "MAVER_MODEL__NUM_RETRIES", "LLM_NUM_RETRIES", description="Retries per model call."
    )


class RuntimeSettings(SettingsGroup):
    """Controls that keep an open-weight model's generation loop productive.

    These are the interventions a published description of the system names directly:
    a generation cap, bounded nudges after a turn that called no tool, a repeat breaker
    that warns then refuses then ends the run, and a single finaliser rescue. They are
    plumbing, not an experimental arm, and they apply to every configuration.
    """

    model_config = group_config("runtime")

    no_tool_nudges: int = setting(
        2,
        "MAVER_RUNTIME__NO_TOOL_NUDGES",
        "LLM_NO_TOOL_NUDGE_LIMIT",
        description="Protocol reminders appended after a turn that called no tool.",
    )
    submit_guard_rejections: int = setting(
        2,
        "MAVER_RUNTIME__SUBMIT_GUARD_REJECTIONS",
        "LLM_SUBMIT_GUARD_LIMIT",
        description="Times a guard may refuse a premature submission.",
    )
    submit_validation_rejections: int = setting(
        3,
        "MAVER_RUNTIME__SUBMIT_VALIDATION_REJECTIONS",
        "LLM_SUBMIT_VALIDATION_LIMIT",
        description="Times a schema-invalid submission is handed back for repair.",
    )
    repeat_call_soft: int = setting(
        3,
        "MAVER_RUNTIME__REPEAT_CALL_SOFT",
        "LLM_REPEAT_CALL_SOFT_LIMIT",
        description="Identical repeats before the loop warns.",
    )
    repeat_call_hard: int = setting(
        6,
        "MAVER_RUNTIME__REPEAT_CALL_HARD",
        "LLM_REPEAT_CALL_HARD_LIMIT",
        description="Identical repeats before the loop refuses the call.",
    )
    repeat_call_stop: int = setting(
        9,
        "MAVER_RUNTIME__REPEAT_CALL_STOP",
        "LLM_REPEAT_CALL_STOP_LIMIT",
        description="Identical repeats before the run is ended.",
    )
    max_tool_calls_per_turn: int = setting(
        8,
        "MAVER_RUNTIME__MAX_TOOL_CALLS_PER_TURN",
        "LLM_MAX_TOOL_CALLS_PER_TURN",
        description="Tool calls honoured from a single turn.",
    )
    keep_reasoning_max_chars: int = setting(
        6000,
        "MAVER_RUNTIME__KEEP_REASONING_MAX_CHARS",
        "LLM_KEEP_REASONING_MAX_CHARS",
        description="Ceiling on reasoning text carried between turns, when the profile keeps it.",
    )
    finalizer_rescues: int = setting(
        1,
        "MAVER_RUNTIME__FINALIZER_RESCUES",
        "LLM_FINALIZER_RESCUE_LIMIT",
        description="Rescue attempts per run after a protocol stall.",
    )
    finalizer_min_content_chars: int = setting(
        800,
        "MAVER_RUNTIME__FINALIZER_MIN_CONTENT_CHARS",
        "LLM_FINALIZER_RESCUE_MIN_CONTENT_CHARS",
        description="Evidence a transcript must hold before a rescue is worth attempting.",
    )
    finalizer_tool_results: int = setting(
        8,
        "MAVER_RUNTIME__FINALIZER_TOOL_RESULTS",
        "LLM_FINALIZER_RESCUE_TOOL_RESULTS",
        description="Tool results kept when compacting a transcript for rescue.",
    )
    finalizer_tool_result_chars: int = setting(
        1600,
        "MAVER_RUNTIME__FINALIZER_TOOL_RESULT_CHARS",
        "LLM_FINALIZER_RESCUE_TOOL_RESULT_CHARS",
        description="Characters kept per retained tool result.",
    )
    finalizer_assistant_chars: int = setting(
        6000,
        "MAVER_RUNTIME__FINALIZER_ASSISTANT_CHARS",
        "LLM_FINALIZER_RESCUE_ASSISTANT_CHARS",
        description="Characters kept of the role's own prior output.",
    )

    @model_validator(mode="after")
    def _repeat_limits_ascend(self) -> "RuntimeSettings":
        """Warn, then refuse, then stop. Out of order, a stage is unreachable.

        Returns:
            The validated group.

        Raises:
            ValueError: When the three repeat thresholds are not strictly increasing.
        """
        if not self.repeat_call_soft < self.repeat_call_hard < self.repeat_call_stop:
            raise ValueError(
                "repeat_call_soft < repeat_call_hard < repeat_call_stop must hold: "
                f"got {self.repeat_call_soft}, {self.repeat_call_hard}, {self.repeat_call_stop}"
            )
        return self


class SecondPassSettings(SettingsGroup):
    """The gated second diagnostic opinion.

    A subgroup of memory rather than a group of its own, because it is one of the two
    ways a past incident reaches a present one. The other is ``role_retrieval``, where a
    role calls a retrieval tool itself. Splitting them across two top-level groups made
    the pair look independent, and the pair is the mechanism.
    """

    model_config = group_config("memory__second_pass")

    enabled: bool = setting(
        False,
        "MAVER_MEMORY__SECOND_PASS__ENABLED",
        "MEMORY_SECOND_OPINION_ENABLED",
        description="Allow a second diagnostic run informed by retrieved precedent.",
    )
    gate: Literal["quality", "nominator", "always"] = setting(
        "quality",
        "MAVER_MEMORY__SECOND_PASS__GATE",
        "MEMORY_SECOND_OPINION_GATE",
        description="What decides whether a second pass is warranted.",
    )
    delivery: SecondPassDelivery = setting(
        "context",
        "MAVER_MEMORY__SECOND_PASS__DELIVERY",
        "MEMORY_SECOND_OPINION_MODE",
        description=(
            "What the second run receives: precedent text, a fault-family assignment "
            "with no text, or nothing at all, which is the resampling control."
        ),
    )
    force: bool = setting(
        False,
        "MAVER_MEMORY__SECOND_PASS__FORCE",
        "MEMORY_SECOND_OPINION_FORCE",
        description=(
            "Deliver an assignment even when the nominator finds no differing family, "
            "turning the second run into a blind verification pass. Lifts the "
            "nominator's veto only; it does not override the quality gate."
        ),
    )
    drop_boilerplate: bool = setting(
        False,
        "MAVER_MEMORY__SECOND_PASS__DROP_BOILERPLATE",
        "MEMORY_EVIDENCE_QUERY_DROP_BOILERPLATE",
        description=(
            "Strip section headers and empty-state markers from the pre-anchor "
            "observation query before it is searched. Off is the prototype's default "
            "and how the published cells were retrieved; the September campaign ran it "
            "on two of its four hosts and off on the other two, so neither value is "
            "the campaign's and an arm reproducing one host's cells pins it."
        ),
    )
    max_precedents: int = setting(
        6,
        "MAVER_MEMORY__SECOND_PASS__MAX_PRECEDENTS",
        description=(
            "Ceiling on the merged union of the two observation-keyed queries. Six "
            "reproduces the measured configuration: the campaign ran three hits per "
            "query with 0.51 mean overlap, for a merged mean of 5.49, so a cap of "
            "three would deliver roughly half the dose the published cells received."
        ),
    )
    gate_max_tokens: int = setting(
        600,
        "MAVER_MEMORY__SECOND_PASS__GATE_MAX_TOKENS",
        "MEMORY_SECOND_OPINION_QUALITY_MAX_TOKENS",
        description="Generation cap for the quality gate call.",
    )
    nominator_max_tokens: int = setting(
        900,
        "MAVER_MEMORY__SECOND_PASS__NOMINATOR_MAX_TOKENS",
        "MEMORY_SECOND_OPINION_NOMINATOR_MAX_TOKENS",
        description="Generation cap for the nominator call.",
    )
    selector_max_tokens: int = setting(
        1600,
        "MAVER_MEMORY__SECOND_PASS__SELECTOR_MAX_TOKENS",
        "MEMORY_COUNTERFACTUAL_SELECTOR_MAX_TOKENS",
        description=(
            "Generation cap for the selector call. Deliberately well under the "
            "run-wide cap: an unbounded reasoning budget under a frequency penalty "
            "degenerates rather than deciding better."
        ),
    )
    selector_timeout_s: int = setting(
        180,
        "MAVER_MEMORY__SECOND_PASS__SELECTOR_TIMEOUT_S",
        "MEMORY_COUNTERFACTUAL_SELECTOR_TIMEOUT",
        description=("Wall-clock bound on the selector; a timeout falls back to the first answer."),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_delivery_names(cls, data: object) -> object:
        """Translate the prototype's delivery vocabulary.

        The old names actively misled: ``memory`` and ``context`` were both about
        memory, and ``resample`` named a mode that deliberately used none. Both old
        values are still accepted.

        Args:
            data: The raw input mapping.

        Returns:
            The input with any legacy delivery value translated.
        """
        if not isinstance(data, dict):
            return data
        renamed = {"memory": "assignment", "resample": "none"}
        for key in (
            "delivery",
            "MEMORY_SECOND_OPINION_MODE",
            "MAVER_MEMORY__SECOND_PASS__DELIVERY",
            "KIRA_MEMORY__SECOND_PASS__DELIVERY",
        ):
            value = data.get(key)
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            if lowered == "critique":
                raise ValueError(
                    "secondpass.delivery='critique' is not a shipped mechanism: it "
                    "passed precedent text to the second pass verbatim, which breaks "
                    "the invariant the published second-pass configurations rest on. "
                    "Use 'context' for precedent text, 'assignment' for a fault-family "
                    "assignment, or 'none' for the resampling control."
                )
            if lowered in renamed:
                data[key] = renamed[lowered]
        return data


class MemorySettings(SettingsGroup):
    """The experience store and the offline pass that fills it."""

    model_config = group_config("memory")

    enabled: bool = setting(
        True,
        "MAVER_MEMORY__ENABLED",
        "MEMORY_ENABLED",
        description="Whether the experience store is readable at all.",
    )
    role_retrieval: bool = setting(
        True,
        "MAVER_MEMORY__ROLE_RETRIEVAL",
        description=(
            "Whether a role may call a retrieval tool. Off with the store enabled is a "
            "real configuration: the second pass reads precedents in the harness while "
            "every agent run stays retrieval-free."
        ),
    )
    second_pass: SecondPassSettings = Field(default_factory=SecondPassSettings)
    """The gated second opinion, the mechanism's other adoption path."""
    exclude_record_ids: NameSet = setting(
        frozenset(),
        "MAVER_MEMORY__EXCLUDE_RECORD_IDS",
        "MEMORY_EXCLUDE_SESSION_IDS",
        description=(
            "Records withheld from every retrieval, as a process-wide floor. An "
            "environment adapter's own per-incident set is unioned with this rather "
            "than replacing it, so a deployment's promise about the corpus and an "
            "evaluation protocol's leave-one-out exclusion both hold at once."
        ),
    )
    learning_enabled: bool = setting(
        True,
        "MAVER_MEMORY__LEARNING_ENABLED",
        "LEARNING_ENABLED",
        description="Whether completed incidents are distilled into new records.",
    )
    learning_delay_s: int = setting(
        300,
        "MAVER_MEMORY__LEARNING_DELAY_S",
        "LEARNING_DELAY_SECONDS",
        description="Wait before distilling, so a late outcome is visible.",
    )
    learning_timeout_s: int = setting(
        600,
        "MAVER_MEMORY__LEARNING_TIMEOUT_S",
        "LEARNING_TIMEOUT_SECONDS",
        description=(
            "Ceiling on a distillation an environment adapter runs inline, while it "
            "still holds the environment. Bounded because the release waits on it, and "
            "a hung pass would strand a deployed application."
        ),
    )
    learning_verify: bool = setting(
        False,
        "MAVER_MEMORY__LEARNING_VERIFY",
        "LEARNING_VERIFY_ENABLED",
        description=(
            "Let the distilling pass re-check the incident's real symptoms instead of "
            "trusting the remediation's own report. Only useful while the environment "
            "is still up, and suppressed anyway when an external evaluation is present."
        ),
    )
    learning_config_dir: Path = setting(
        Path("configs/learning"),
        "MAVER_MEMORY__LEARNING_CONFIG_DIR",
        description="Where the learning trigger configuration is read from.",
    )
    backend: Literal["embedded", "postgres"] = setting(
        "embedded",
        "MAVER_MEMORY__BACKEND",
        "MEMORY_MODE",
        description=(
            "Embedded SQLite, or a PostgreSQL server. The two are one retriever: both "
            "index and query the terms of the same Python normaliser, so the same "
            "corpus and query produce the same candidate pool on either. That is a "
            "reproducibility requirement rather than a convenience — the published "
            "numbers were measured on the server backend, and an adopter who will not "
            "run PostgreSQL has to be able to reproduce them. Choose on operational "
            "grounds: a server for concurrent writes and a durable campaign, the "
            "embedded file for a single process and no service to run."
        ),
    )
    postgres_dsn: Credential = setting(
        None,
        "MAVER_MEMORY__POSTGRES_DSN",
        "MEMORY_POSTGRES_DSN",
        description="Connection string. Required by the postgres backend.",
    )
    postgres_schema: str = setting(
        "kira_memory",
        "MAVER_MEMORY__POSTGRES_SCHEMA",
        "MEMORY_POSTGRES_SCHEMA",
        description="Schema holding the tables.",
    )
    corpus_id: str = setting(
        "default",
        "MAVER_MEMORY__CORPUS_ID",
        "MEMORY_EXPERIMENT_ID",
        description=(
            "Names this campaign's corpus. The partition is the database itself, so a "
            "run configured for one corpus cannot read another's."
        ),
    )
    top_k: int = setting(
        3,
        "MAVER_MEMORY__TOP_K",
        "MEMORY_SEARCH_TOP_K",
        description=(
            "Records returned by a retrieval. Three rather than the prototype's code "
            "default of five: the archived environment of the campaigns this build "
            "reproduces sets MEMORY_SEARCH_TOP_K=3, and the frozen arm files pin three. "
            "Recall and precision pull against each other here — over the offline "
            "corpus 1 to 20 results moves useful-precedent recall 0.39 to 0.59 while "
            "precision falls 0.39 to 0.04 — and a small model that anchors easily is "
            "served the precision end of that trade. The earlier MA+R arm, whose "
            "campaign ran at five, pins five in its own file."
        ),
    )
    candidate_pool: int = setting(
        500,
        "MAVER_MEMORY__CANDIDATE_POOL",
        description=(
            "Ceiling on the candidate pool a backend hands the ranker. It bounds the "
            "cost of a retrieval and, because BM25's inverse document frequency is "
            "computed over the pool it is given, it is also the population a score is "
            "relative to. The pool is taken in write order rather than by rank, so that "
            "the ranker rather than the database decides relevance — which means a "
            "corpus larger than this drops its newest records from every retrieval. "
            "Raise it above the corpus size for a deployment that has outgrown it."
        ),
    )
    min_score_per_token: float = setting(
        0.0,
        "MAVER_MEMORY__MIN_SCORE_PER_TOKEN",
        "MEMORY_MIN_TEXT_RANK_PER_TOKEN",
        description=(
            "Relevance floor under a hit, as its lexical score over the query's token "
            "count, which is what lets a retrieval answer that nothing is comparable. "
            "Zero or less disables it and returns k records whatever they score, rather "
            "than setting the floor at zero: a BM25 score is not bounded below, so a "
            "zero threshold would drop a whole small or homogeneous pool. Off by "
            "default because no measured run set it: the prototype shipped it off and "
            "no host's environment named it. The offline screen that favoured 0.5 is "
            "a reason to try it, not a measured configuration to inherit."
        ),
    )
    mask_identifiers: bool = setting(
        False,
        "MAVER_MEMORY__MASK_IDENTIFIERS",
        description=(
            "Replace an incident's own resource and namespace names with type "
            "placeholders before a record is indexed, so a lesson about one application "
            "can be retrieved for the same fault on another. Screened over three corpus "
            "snapshots: recall@5 for cross-application precedents rose 0.455 to 0.475, "
            "0.433 to 0.448 and 0.417 to 0.500, at a cost of 0.005-0.027 same-family "
            "recall. Off, because it changes every stored record and no published "
            "measurement was taken with it on."
        ),
    )
    mask_namespaces: NameSet = setting(
        frozenset(),
        "MAVER_MEMORY__MASK_NAMESPACES",
        description=(
            "Further namespaces masked alongside the incident's own, for a deployment "
            "running more than one application. Only read when mask_identifiers is on. "
            "Supplied rather than known: the prototype carried its benchmark's three "
            "namespaces as a constant inside the memory layer."
        ),
    )
    ranker: str = setting(
        "bm25",
        "MAVER_MEMORY__RANKER",
        description=(
            "Which registered ranker scores the candidate pool. `bm25` ships; another "
            "is added by registering a factory in the `maver.rankers` entry-point group, "
            "with no fork. A name nothing registered is a startup failure rather than a "
            "fall back to the default, because retrieving with a ranker other than the "
            "one a run reports is how a result stops meaning what it says."
        ),
    )
    lifecycle_weights: dict[str, float] | None = setting(
        None,
        "MAVER_MEMORY__LIFECYCLE_WEIGHTS",
        description=(
            "Retrieval weight per lifecycle state. None uses the shipped table "
            "(invalidated 0.3, tentative_success 1.0, human_verified 1.25, else 0.9). "
            "Present so a frozen arm can pin the exact ranking a past campaign used."
        ),
    )

    @model_validator(mode="after")
    def _postgres_needs_a_dsn(self) -> "MemorySettings":
        """Fail at startup rather than at the first incident.

        Returns:
            The validated group.

        Raises:
            ValueError: When the postgres backend has no connection string.
        """
        if self.backend == "postgres" and self.postgres_dsn is None:
            raise ValueError(
                "memory.backend='postgres' requires memory.postgres_dsn "
                "(MAVER_MEMORY__POSTGRES_DSN or MEMORY_POSTGRES_DSN)"
            )
        return self

    @model_validator(mode="after")
    def _an_adoption_path_needs_a_readable_store(self) -> "MemorySettings":
        """Refuse a configuration that delivers experience it cannot read.

        Both paths out of the store are checked in one place because they draw on one
        store. The resampling control is the exception it looks like: it deliberately
        uses no memory, which is what makes it the attribution control for the other.

        Returns:
            The validated group.

        Raises:
            ValueError: When a path is on with nothing behind it.
        """
        if self.second_pass.enabled and self.second_pass.delivery != "none" and not self.enabled:
            raise ValueError(
                "memory.second_pass.enabled with delivery != 'none' requires "
                "memory.enabled: the second pass has no precedent to deliver. Use "
                "delivery='none' for the resampling control, which is the "
                "configuration that deliberately uses no memory."
            )
        if self.learning_enabled and not self.enabled:
            raise ValueError(
                "memory.learning_enabled requires memory.enabled: distilling records "
                "that nothing can retrieve is work with no consumer."
            )
        return self


class VerifySettings(SettingsGroup):
    """The independent end-to-end recovery check and its hand-back loop.

    Verification is unconditionally blind to the diagnosis, so its independence is
    structural rather than requested. The prototype made that a flag; a flag implies the
    sighted mode is a supported configuration, and a sighted verification derives its
    checks from the causal chain it is supposed to be checking, which lets a wrong
    diagnosis certify its own recovery. The cost of the decision is that the routing
    precision of blindness can no longer be re-measured here.
    """

    model_config = group_config("verify")

    enabled: bool = setting(
        False,
        "MAVER_VERIFY__ENABLED",
        "VERIFY_STEP_ENABLED",
        description="Run a read-only recovery check after a clean apply.",
    )
    health_gate: bool = setting(
        False,
        "MAVER_VERIFY__HEALTH_GATE",
        "VERIFY_HEALTH_GATE",
        description="Enforce a computed namespace-health floor before accepting recovery.",
    )
    handback_retries: Annotated[int | None, Field(ge=0)] = setting(
        None,
        "MAVER_VERIFY__HANDBACK_RETRIES",
        description=(
            "Maximum Verify returns to RCA or Planning, within the shared remediation "
            "budget. Unset inherits pipeline.remediation_retries; zero allows recovery "
            "checks but no Verify hand-backs. Execution replans and inconclusive "
            "rechecks do not spend this allowance."
        ),
    )
    inconclusive_retries: int = setting(
        1,
        "MAVER_VERIFY__INCONCLUSIVE_RETRIES",
        "VERIFY_INCONCLUSIVE_RETRIES",
        description=(
            "Re-checks after an inconclusive verdict. Budgeted separately from "
            "remediation because verification is read-only: a second look costs a few "
            "telemetry reads, while closing on 'could not tell' ends the incident with "
            "recovery unconfirmed."
        ),
    )


class KafkaSettings(SettingsGroup):
    """Optional alert consumer. Off by default; needs the ``kafka`` extra."""

    model_config = group_config("intake__kafka")

    enabled: bool = setting(
        False,
        "MAVER_INTAKE__KAFKA__ENABLED",
        "KAFKA_ENABLED",
        description="Consume alerts from a topic in addition to the HTTP endpoint.",
    )
    bootstrap_servers: str | None = setting(
        None,
        "MAVER_INTAKE__KAFKA__BOOTSTRAP_SERVERS",
        "KAFKA_BOOTSTRAP_SERVERS",
        description="Broker list.",
    )
    topic: str = setting(
        "alerts", "MAVER_INTAKE__KAFKA__TOPIC", "KAFKA_TOPIC", description="Topic to consume."
    )
    group_id: str = setting(
        "maver", "MAVER_INTAKE__KAFKA__GROUP_ID", "KAFKA_GROUP_ID", description="Consumer group."
    )

    @model_validator(mode="after")
    def _enabled_needs_brokers(self) -> "KafkaSettings":
        """Refuse to start a consumer with nowhere to consume from.

        Returns:
            The validated group.

        Raises:
            ValueError: When enabled without a broker list.
        """
        if self.enabled and not self.bootstrap_servers:
            raise ValueError("intake.kafka.enabled requires intake.kafka.bootstrap_servers")
        return self


def _expanded(value: Path | None) -> Path | None:
    """Expand a leading ``~``, which pydantic keeps literal.

    Args:
        value: The configured path.

    Returns:
        The path with the home directory resolved, or None.
    """
    return None if value is None else value.expanduser()


class PromptSettings(SettingsGroup):
    """Where a deployment's own role prompts come from, when it has any."""

    model_config = group_config("prompts")

    override_dir: Annotated[Path | None, AfterValidator(_expanded)] = setting(
        None,
        "MAVER_PROMPTS__OVERRIDE_DIR",
        description=(
            "Directory holding replacement role prompts, as `roles/<role>.txt`. A file "
            "that is not there leaves that role on the packaged prompt, so a deployment "
            "can replace one role and inherit the rest. This is the supported way to "
            "change a prompt: editing a file inside the installed package does not "
            "survive an upgrade and leaves no record of what was run. Overridden roles "
            "and the digest of each are written into the run's configuration snapshot, "
            "because a result produced under a prompt nobody recorded is not "
            "attributable. A leading `~` is expanded here, so every reader of it "
            "resolves the same directory."
        ),
    )


class IntakeSettings(SettingsGroup):
    """How incidents reach the system."""

    model_config = group_config("intake")

    http_enabled: bool = setting(
        True,
        "MAVER_INTAKE__HTTP_ENABLED",
        "HTTP_INTAKE_ENABLED",
        description="Accept incidents over the HTTP endpoint.",
    )
    adapters: NameSet = setting(
        frozenset(),
        "MAVER_INTAKE__ADAPTERS",
        description=(
            "Environment adapters to enable, by registered name. Each supplies the "
            "transport its tools speak and may mount routes of its own. Empty means "
            "none: the service runs on the direct transport and its generic intake."
        ),
    )
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)


class ToolSettings(SettingsGroup):
    """Tool-layer limits and kill switches."""

    model_config = group_config("tools")

    disabled: NameSet = setting(
        frozenset(),
        "MAVER_TOOLS__DISABLED",
        description="Tool names withheld from every role. Submission tools are exempt.",
    )
    code_exec_timeout_s: int = setting(
        120,
        "MAVER_TOOLS__CODE_EXEC_TIMEOUT_S",
        "CODE_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a sandboxed code execution.",
    )
    shell_exec_timeout_s: int = setting(
        120,
        "MAVER_TOOLS__SHELL_EXEC_TIMEOUT_S",
        "SHELL_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a sandboxed shell command.",
    )
    kubectl_timeout_s: int = setting(
        120,
        "MAVER_TOOLS__KUBECTL_TIMEOUT_S",
        "KUBECTL_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a cluster command.",
    )
    fs_sandbox: bool = setting(
        True,
        "MAVER_TOOLS__FS_SANDBOX",
        "TOOL_FS_SANDBOX_ENABLED",
        description="Confine code and shell execution to the incident directory.",
    )
    blind_namespaces: NameSet = setting(
        frozenset(),
        "MAVER_TOOLS__BLIND_NAMESPACES",
        "BLIND_NAMESPACES",
        description="Namespaces removed from telemetry results before a role sees them.",
    )
    annotate_stale_namespaces: bool = setting(
        False,
        "MAVER_TOOLS__ANNOTATE_STALE_NAMESPACES",
        "STALE_ALERT_ANNOTATION_ENABLED",
        description=(
            "Check each firing alert's namespace against the cluster and mark the "
            "ones whose namespace no longer exists. Off by default: it costs one "
            "cluster read per distinct namespace on every alert query."
        ),
    )
    host_recovery_nodes: NameSet = setting(
        frozenset(),
        "MAVER_TOOLS__HOST_RECOVERY_NODES",
        "HOST_RECOVERY_ALLOWED_NODES",
        description="Nodes reachable for host-level recovery. Empty disables it.",
    )
    kubectl_deny_all_namespaces: bool = setting(
        True,
        "MAVER_TOOLS__KUBECTL_DENY_ALL_NAMESPACES",
        description=(
            "Refuse cluster-wide kubectl reads. This is incident scoping and not a "
            "safety property — a cluster-wide read is routine for a real operator — so "
            "it defaults to the behaviour every published measurement was taken under "
            "and an operator can turn it off."
        ),
    )
    kubectl_denied_argument_pattern: str | None = setting(
        "chaos",
        "MAVER_TOOLS__KUBECTL_DENIED_ARGUMENT_PATTERN",
        description=(
            "Case-insensitive regular expression searched in every argument of every "
            "kubectl call, before the verb is read. An evaluation-integrity knob, not a "
            "safety rule: a benchmark that injects faults with a named controller leaks "
            "the answer to anyone who can query that controller's resources, which "
            "turns diagnosis into a lookup. It refuses reads and writes alike, so a "
            "cluster whose own namespaces match it cannot be operated through this "
            "agent at all — set it empty there. The default is the pattern every "
            "published measurement was taken under."
        ),
    )

    @field_validator("kubectl_denied_argument_pattern", mode="after")
    @classmethod
    def _pattern_compiles(cls, value: str | None) -> str | None:
        """Reject an unusable expression at startup rather than at the first kubectl.

        Args:
            value: The configured pattern.

        Returns:
            The pattern, or None where it is empty.

        Raises:
            ValueError: When the expression will not compile.
        """
        if not value:
            return None
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"kubectl_denied_argument_pattern is not a regexp: {exc}") from exc
        return value


class TransportSettings(SettingsGroup):
    """Where the default transport finds the cluster and its telemetry backends.

    An unset endpoint is a description of the deployment, not an error. The transport
    reports what it was given as its capabilities, and a tool whose backend is missing
    is dropped before the run with the reason recorded, rather than failing the first
    time a role calls it.
    """

    model_config = group_config("transport")

    kubectl_binary: str = setting(
        "kubectl",
        "MAVER_TRANSPORT__KUBECTL_BINARY",
        description="The kubectl executable, looked up on PATH unless given as a path.",
    )
    kubeconfig: Path | None = setting(
        None,
        "MAVER_TRANSPORT__KUBECONFIG",
        description=(
            "Credentials for the cluster. Unset leaves kubectl its own discovery, which "
            "is the mounted service account when the agent runs inside the cluster."
        ),
    )
    prometheus_url: AnyHttpUrl | None = setting(
        None,
        "MAVER_TRANSPORT__PROMETHEUS_URL",
        description="Base URL of the metrics backend. Unset withdraws the metrics tools.",
    )
    loki_url: AnyHttpUrl | None = setting(
        None,
        "MAVER_TRANSPORT__LOKI_URL",
        description="Base URL of the log backend. Unset withdraws the log tools.",
    )
    jaeger_url: AnyHttpUrl | None = setting(
        None,
        "MAVER_TRANSPORT__JAEGER_URL",
        description="Base URL of the trace backend. Unset withdraws the trace tools.",
    )
    timeout_s: int = setting(
        120,
        "MAVER_TRANSPORT__TIMEOUT_S",
        description=(
            "How long one cluster or telemetry call may take. A blocking kubectl is "
            "bounded ten seconds under this, so it reports its own timeout rather than "
            "being killed with nothing to show for the wait. Must be positive: zero "
            "means no limit in curl, docker and kubectl's own --request-timeout, and "
            "here it would kill every call on a deadline already past."
        ),
    )

    @field_validator("timeout_s")
    @classmethod
    def _timeout_is_a_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("transport.timeout_s must be at least 1 second")
        return value


class ObservabilitySettings(SettingsGroup):
    """Tracing, logging, and the audit event log."""

    model_config = group_config("observability")

    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = setting(
        "INFO",
        "MAVER_OBSERVABILITY__LOG_LEVEL",
        "LOG_LEVEL",
        description="Root log level, applied to the root logger when the server starts.",
    )
    otlp_endpoint: str | None = setting(
        None,
        "MAVER_OBSERVABILITY__OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "COLLECTOR_SERVICE_ADDR",
        description="Collector address. Unset disables export without disabling instrumentation.",
    )
    otlp_insecure: bool = setting(
        True,
        "MAVER_OBSERVABILITY__OTLP_INSECURE",
        "OTEL_EXPORTER_OTLP_INSECURE",
        description="Whether the collector connection skips TLS.",
    )
    metric_interval_ms: int = setting(
        30_000,
        "MAVER_OBSERVABILITY__METRIC_INTERVAL_MS",
        "OTEL_METRIC_EXPORT_INTERVAL_MS",
        description="Metric export period.",
    )
    deployment_id: str | None = setting(
        None,
        "MAVER_OBSERVABILITY__DEPLOYMENT_ID",
        "AGENT_DEPLOYMENT_ID",
        description="Identifies this deployment in traces and in the config snapshot.",
    )
    event_log_enabled: bool = setting(
        True,
        "MAVER_OBSERVABILITY__EVENT_LOG_ENABLED",
        "JSONL_EVENT_LOG_ENABLED",
        description=(
            "Legacy compatibility field; ignored by the service composition root. "
            "The service always records audit evidence under server.workdir/audit."
        ),
    )
    event_log_path: Path | None = setting(
        None,
        "MAVER_OBSERVABILITY__EVENT_LOG_PATH",
        "JSONL_EVENT_LOG_PATH",
        description=(
            "Legacy compatibility field; ignored. The service audit directory is "
            "server.workdir/audit; custom applications select their own sink."
        ),
    )


class ServerSettings(SettingsGroup):
    """Process and filesystem placement."""

    model_config = group_config("server")

    # API_HOSTNAME and API_PORT are deliberately not accepted here. In the evaluation
    # harness's own deployment files those two names are the *conductor's* address, and
    # honouring them made this service bind the conductor's port while the harness
    # posted to 8080 — a run that fails as a health-check timeout naming nothing. They
    # reach the adapter that gives them their other meaning instead.
    host: str = setting("0.0.0.0", "MAVER_SERVER__HOST", description="Bind address.")
    port: int = setting(8080, "MAVER_SERVER__PORT", description="Bind port.")
    workdir: Path = setting(
        Path(".local-workspace"),
        "MAVER_SERVER__WORKDIR",
        "WORKDIR",
        description="Root for incident directories and embedded stores.",
    )
    environment: Literal["local", "production"] = setting(
        "local",
        "MAVER_SERVER__ENVIRONMENT",
        "ENVIRONMENT",
        description=(
            "Selects production hardening: the API documentation endpoints are "
            "withheld, and a deployment that approves its own cluster writes "
            "(authority.mode='auto') is refused at startup."
        ),
    )
    drain_timeout_s: float = setting(
        30.0,
        "MAVER_SERVER__DRAIN_TIMEOUT_S",
        description=(
            "How long a shutdown waits for incidents already in flight. An incident may "
            "have changed the cluster already, so killing one mid-remediation leaves it "
            "half-modified with nobody recording what was done."
        ),
    )
    max_concurrent_incidents: int = setting(
        4,
        "MAVER_SERVER__MAX_CONCURRENT_INCIDENTS",
        description=(
            "Incidents processed at once. A private model endpoint serves a small "
            "number of concurrent sequences, so unbounded intake degrades every run "
            "in flight rather than queueing."
        ),
    )
