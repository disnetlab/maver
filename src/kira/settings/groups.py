"""The settings groups.

Ten groups replace the prototype's 1,272-line module of 105 bare accessor functions
over ``os.environ``, plus roughly 55 more variables read directly elsewhere. Of those
105, 47 existed only to ablate something; that ratio, not the line count, is what made
the original unreadable.

Every field names the modern ``KIRA_GROUP__FIELD`` variable and any flat name the
prototype used, so the existing evaluation run scripts keep working unchanged. Defaults
fail closed.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from kira.settings.base import NameSet, SettingsGroup, group_config, setting

__all__ = [
    "AuthoritySettings",
    "IntakeSettings",
    "KafkaSettings",
    "MemorySettings",
    "ModelSettings",
    "ObservabilitySettings",
    "PipelineSettings",
    "RuntimeSettings",
    "SecondPassDelivery",
    "SecondPassSettings",
    "ServerSettings",
    "ToolSettings",
    "VerifySettings",
]

SecondPassDelivery = Literal["context", "assignment", "none", "critique"]


class PipelineSettings(SettingsGroup):
    """The graph, its shape, and its retry budgets."""

    model_config = group_config("pipeline")

    mode: Literal["multi", "single"] = setting(
        "multi",
        "KIRA_PIPELINE__MODE",
        "AGENT_MODE",
        description="Three typed roles, or one goal-only agent over a single transcript.",
    )
    max_turns: int = setting(
        20,
        "KIRA_PIPELINE__MAX_TURNS",
        "MAX_TURNS",
        description="Model/tool turns a role may take before its run is ended.",
    )
    remediation_retries: int = setting(
        2,
        "KIRA_PIPELINE__REMEDIATION_RETRIES",
        "MAX_REMEDIATION_RETRIES",
        "MAX_REASSESSMENTS",
        description=(
            "One budget shared by every back-edge, so replan and re-diagnose cycles "
            "cannot compound into independent budgets."
        ),
    )
    auto_snapshot: bool = setting(
        True,
        "KIRA_PIPELINE__AUTO_SNAPSHOT",
        "RCA_AUTO_SNAPSHOT",
        # A deploy script sets RCA_AUTO_SNAPSHOT_ENABLED while the prototype reads
        # RCA_AUTO_SNAPSHOT, so asking for the snapshot to be off left it on and
        # nobody noticed. Accepting both names is the compatible fix.
        "RCA_AUTO_SNAPSHOT_ENABLED",
        description="Attach a namespace overview to the diagnosing role's first message.",
    )
    reassess_mode: Literal["fresh", "talkback"] = setting(
        "fresh",
        "KIRA_PIPELINE__REASSESS_MODE",
        "RCA_REASSESS_MODE",
        description="Whether a re-diagnosis starts clean or carries the prior transcript.",
    )
    talkback_max_chars: int = setting(
        400_000,
        "KIRA_PIPELINE__TALKBACK_MAX_CHARS",
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
        None, "KIRA_MODEL__OVERRIDES__MODEL", "LLM_MODEL", description="Provider route."
    )
    max_tokens: int | None = setting(
        None, "KIRA_MODEL__OVERRIDES__MAX_TOKENS", "LLM_MAX_TOKENS", description="Generation cap."
    )
    temperature: float | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__TEMPERATURE",
        "LLM_TEMPERATURE",
        description="Sampling temperature.",
    )
    top_p: float | None = setting(
        None, "KIRA_MODEL__OVERRIDES__TOP_P", "LLM_TOP_P", description="Nucleus sampling mass."
    )
    top_k: int | None = setting(
        None, "KIRA_MODEL__OVERRIDES__TOP_K", "LLM_TOP_K", description="Top-k cutoff."
    )
    min_p: float | None = setting(
        None, "KIRA_MODEL__OVERRIDES__MIN_P", "LLM_MIN_P", description="Minimum token probability."
    )
    presence_penalty: float | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__PRESENCE_PENALTY",
        "LLM_PRESENCE_PENALTY",
        description="Presence penalty.",
    )
    frequency_penalty: float | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__FREQUENCY_PENALTY",
        "LLM_FREQUENCY_PENALTY",
        description="Frequency penalty.",
    )
    repetition_penalty: float | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__REPETITION_PENALTY",
        "LLM_REPETITION_PENALTY",
        description="Repetition penalty.",
    )
    no_repeat_ngram_size: int | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__NO_REPEAT_NGRAM_SIZE",
        "LLM_NO_REPEAT_NGRAM_SIZE",
        description="Banned repeat n-gram length.",
    )
    seed: int | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__SEED",
        "LLM_SEED",
        description="Sampling seed, where the server honours one.",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__REASONING_EFFORT",
        "LLM_REASONING_EFFORT",
        description="Reasoning effort, where supported.",
    )
    thinking_token_budget: int | None = setting(
        None,
        "KIRA_MODEL__OVERRIDES__THINKING_TOKEN_BUDGET",
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
        "KIRA_MODEL__PROFILE",
        description="Stem of a file under the profile directory.",
    )
    profile_dir: Path = setting(
        Path("configs/models"),
        "KIRA_MODEL__PROFILE_DIR",
        description="Where model profiles are read from.",
    )
    endpoint_override: str | None = setting(
        None,
        "KIRA_MODEL__ENDPOINT_OVERRIDE",
        "OPENAI_API_BASE",
        description="Replaces the profile's base URL. For pointing a run at another server.",
    )
    api_key: SecretStr | None = setting(
        None,
        "KIRA_MODEL__API_KEY",
        description="Overrides the key named by the profile's api_key_env.",
    )
    overrides: SamplingOverrides = Field(default_factory=SamplingOverrides)
    request_timeout_s: int = setting(
        1800,
        "KIRA_MODEL__REQUEST_TIMEOUT_S",
        "LLM_TIMEOUT_SECONDS",
        description=(
            "Per-request ceiling. The prototype's 300 s default abandoned long "
            "generations that were still running, so the work was paid for and thrown away."
        ),
    )
    num_retries: int = setting(
        3, "KIRA_MODEL__NUM_RETRIES", "LLM_NUM_RETRIES", description="Retries per model call."
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
        "KIRA_RUNTIME__NO_TOOL_NUDGES",
        "LLM_NO_TOOL_NUDGE_LIMIT",
        description="Protocol reminders appended after a turn that called no tool.",
    )
    submit_guard_rejections: int = setting(
        2,
        "KIRA_RUNTIME__SUBMIT_GUARD_REJECTIONS",
        "LLM_SUBMIT_GUARD_LIMIT",
        description="Times a guard may refuse a premature submission.",
    )
    submit_validation_rejections: int = setting(
        3,
        "KIRA_RUNTIME__SUBMIT_VALIDATION_REJECTIONS",
        "LLM_SUBMIT_VALIDATION_LIMIT",
        description="Times a schema-invalid submission is handed back for repair.",
    )
    repeat_call_soft: int = setting(
        3,
        "KIRA_RUNTIME__REPEAT_CALL_SOFT",
        "LLM_REPEAT_CALL_SOFT_LIMIT",
        description="Identical repeats before the loop warns.",
    )
    repeat_call_hard: int = setting(
        6,
        "KIRA_RUNTIME__REPEAT_CALL_HARD",
        "LLM_REPEAT_CALL_HARD_LIMIT",
        description="Identical repeats before the loop refuses the call.",
    )
    repeat_call_stop: int = setting(
        9,
        "KIRA_RUNTIME__REPEAT_CALL_STOP",
        "LLM_REPEAT_CALL_STOP_LIMIT",
        description="Identical repeats before the run is ended.",
    )
    max_tool_calls_per_turn: int = setting(
        8,
        "KIRA_RUNTIME__MAX_TOOL_CALLS_PER_TURN",
        "LLM_MAX_TOOL_CALLS_PER_TURN",
        description="Tool calls honoured from a single turn.",
    )
    keep_reasoning_max_chars: int = setting(
        6000,
        "KIRA_RUNTIME__KEEP_REASONING_MAX_CHARS",
        "LLM_KEEP_REASONING_MAX_CHARS",
        description="Ceiling on reasoning text carried between turns, when the profile keeps it.",
    )
    finalizer_rescues: int = setting(
        1,
        "KIRA_RUNTIME__FINALIZER_RESCUES",
        "LLM_FINALIZER_RESCUE_LIMIT",
        description="Rescue attempts per run after a protocol stall.",
    )
    finalizer_min_content_chars: int = setting(
        800,
        "KIRA_RUNTIME__FINALIZER_MIN_CONTENT_CHARS",
        "LLM_FINALIZER_RESCUE_MIN_CONTENT_CHARS",
        description="Evidence a transcript must hold before a rescue is worth attempting.",
    )
    finalizer_tool_results: int = setting(
        8,
        "KIRA_RUNTIME__FINALIZER_TOOL_RESULTS",
        "LLM_FINALIZER_RESCUE_TOOL_RESULTS",
        description="Tool results kept when compacting a transcript for rescue.",
    )
    finalizer_tool_result_chars: int = setting(
        1600,
        "KIRA_RUNTIME__FINALIZER_TOOL_RESULT_CHARS",
        "LLM_FINALIZER_RESCUE_TOOL_RESULT_CHARS",
        description="Characters kept per retained tool result.",
    )
    finalizer_assistant_chars: int = setting(
        6000,
        "KIRA_RUNTIME__FINALIZER_ASSISTANT_CHARS",
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


class MemorySettings(SettingsGroup):
    """The experience store and the offline pass that fills it."""

    model_config = group_config("memory")

    enabled: bool = setting(
        True,
        "KIRA_MEMORY__ENABLED",
        "MEMORY_ENABLED",
        description="Whether roles may retrieve past incidents.",
    )
    learning_enabled: bool = setting(
        True,
        "KIRA_MEMORY__LEARNING_ENABLED",
        "LEARNING_ENABLED",
        description="Whether completed incidents are distilled into new records.",
    )
    learning_delay_s: int = setting(
        300,
        "KIRA_MEMORY__LEARNING_DELAY_S",
        "LEARNING_DELAY_SECONDS",
        description="Wait before distilling, so a late outcome is visible.",
    )
    backend: Literal["embedded", "postgres"] = setting(
        "embedded",
        "KIRA_MEMORY__BACKEND",
        "MEMORY_MODE",
        description="Embedded SQLite, or a PostgreSQL server.",
    )
    postgres_dsn: SecretStr | None = setting(
        None,
        "KIRA_MEMORY__POSTGRES_DSN",
        "MEMORY_POSTGRES_DSN",
        description="Connection string. Required by the postgres backend.",
    )
    postgres_schema: str = setting(
        "kira_memory",
        "KIRA_MEMORY__POSTGRES_SCHEMA",
        "MEMORY_POSTGRES_SCHEMA",
        description="Schema holding the tables.",
    )
    corpus_id: str = setting(
        "default",
        "KIRA_MEMORY__CORPUS_ID",
        "MEMORY_EXPERIMENT_ID",
        description="Partitions records so separate campaigns cannot read each other's.",
    )
    top_k: int = setting(
        5,
        "KIRA_MEMORY__TOP_K",
        "MEMORY_SEARCH_TOP_K",
        description="Records returned by a retrieval.",
    )
    playbook: bool = setting(
        False,
        "KIRA_MEMORY__PLAYBOOK",
        "PLAYBOOK_ENABLED",
        description=(
            "Cross-cutting operational notes injected into role prompts. Off by "
            "default: it is absent from the published system description, and it "
            "shipped on and undocumented in the prototype."
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
                "(KIRA_MEMORY__POSTGRES_DSN or MEMORY_POSTGRES_DSN)"
            )
        return self


class VerifySettings(SettingsGroup):
    """The independent end-to-end recovery check and its hand-back loop."""

    model_config = group_config("verify")

    enabled: bool = setting(
        False,
        "KIRA_VERIFY__ENABLED",
        "VERIFY_STEP_ENABLED",
        description="Run a read-only recovery check after a clean apply.",
    )
    blind_to_diagnosis: bool = setting(
        True,
        "KIRA_VERIFY__BLIND_TO_DIAGNOSIS",
        "VERIFY_BLIND_TO_DIAGNOSIS",
        description=(
            "Withhold the diagnosis, so a wrong cause cannot certify its own recovery "
            "through checks derived from it."
        ),
    )
    health_gate: bool = setting(
        False,
        "KIRA_VERIFY__HEALTH_GATE",
        "VERIFY_HEALTH_GATE",
        description="Enforce a computed namespace-health floor before accepting recovery.",
    )
    inconclusive_retries: int = setting(
        1,
        "KIRA_VERIFY__INCONCLUSIVE_RETRIES",
        "VERIFY_INCONCLUSIVE_RETRIES",
        description=(
            "Re-checks after an inconclusive verdict. Budgeted separately from "
            "remediation because verification is read-only: a second look costs a few "
            "telemetry reads, while closing on 'could not tell' ends the incident with "
            "recovery unconfirmed."
        ),
    )


class SecondPassSettings(SettingsGroup):
    """The gated second diagnostic opinion."""

    model_config = group_config("secondpass")

    enabled: bool = setting(
        False,
        "KIRA_SECONDPASS__ENABLED",
        "MEMORY_SECOND_OPINION_ENABLED",
        description="Allow a second diagnostic run informed by retrieved precedent.",
    )
    gate: Literal["quality", "nominator", "always"] = setting(
        "quality",
        "KIRA_SECONDPASS__GATE",
        "MEMORY_SECOND_OPINION_GATE",
        description="What decides whether a second pass is warranted.",
    )
    delivery: SecondPassDelivery = setting(
        "context",
        "KIRA_SECONDPASS__DELIVERY",
        "MEMORY_SECOND_OPINION_MODE",
        description=(
            "What the second run receives: precedent text, a fault-family assignment "
            "with no text, nothing at all (the resampling control), or a critique."
        ),
    )
    force: bool = setting(
        False,
        "KIRA_SECONDPASS__FORCE",
        "MEMORY_SECOND_OPINION_FORCE",
        description="Run the second pass regardless of the gate. For dose measurement.",
    )
    max_precedents: int = setting(
        3,
        "KIRA_SECONDPASS__MAX_PRECEDENTS",
        description="Precedents retrieved for the second run.",
    )
    gate_max_tokens: int = setting(
        600,
        "KIRA_SECONDPASS__GATE_MAX_TOKENS",
        "MEMORY_SECOND_OPINION_QUALITY_MAX_TOKENS",
        description="Generation cap for the quality gate call.",
    )
    nominator_max_tokens: int = setting(
        900,
        "KIRA_SECONDPASS__NOMINATOR_MAX_TOKENS",
        "MEMORY_SECOND_OPINION_NOMINATOR_MAX_TOKENS",
        description="Generation cap for the nominator call.",
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
        for key in ("delivery", "MEMORY_SECOND_OPINION_MODE", "KIRA_SECONDPASS__DELIVERY"):
            value = data.get(key)
            if isinstance(value, str) and value.lower() in renamed:
                data[key] = renamed[value.lower()]
        return data


class AuthoritySettings(SettingsGroup):
    """Who may authorise a write to the cluster.

    Two defaults differ from the prototype, both deliberately. The mode defaults to
    ``prompt`` rather than ``auto``: the prototype shipped automatic approval on a
    service holding cluster write permissions, and an artifact other people deploy
    should not. And a timeout now has a real default, so a pending request cannot wedge
    a session indefinitely; on expiry the write is denied, never approved.

    An evaluation harness that wants the prototype's behaviour sets the mode
    explicitly, which every existing run script already does.
    """

    model_config = group_config("authority")

    mode: Literal["auto", "prompt", "deny"] = setting(
        "prompt",
        "KIRA_AUTHORITY__MODE",
        "HITL_MODE",
        description="Approve writes automatically, ask a human, or refuse them all.",
    )
    timeout_s: int | None = setting(
        1800,
        "KIRA_AUTHORITY__TIMEOUT_S",
        "HITL_TIMEOUT_SECONDS",
        description="How long a request waits for a human. None waits forever.",
    )
    on_timeout: Literal["deny", "approve"] = setting(
        "deny",
        "KIRA_AUTHORITY__ON_TIMEOUT",
        description="What an expired request resolves to. Denying is the safe default.",
    )
    require_approval_stages: NameSet = setting(
        frozenset({"execution"}),
        "KIRA_AUTHORITY__REQUIRE_APPROVAL_STAGES",
        description="Stages whose write-capable tool calls pass through the gate.",
    )
    auto_approve_tools: NameSet = setting(
        frozenset(),
        "KIRA_AUTHORITY__AUTO_APPROVE_TOOLS",
        description="Tools exempted from prompting even in prompt mode.",
    )


class KafkaSettings(SettingsGroup):
    """Optional alert consumer. Off by default; needs the ``kafka`` extra."""

    model_config = group_config("intake__kafka")

    enabled: bool = setting(
        False,
        "KIRA_INTAKE__KAFKA__ENABLED",
        "KAFKA_ENABLED",
        description="Consume alerts from a topic in addition to the HTTP endpoint.",
    )
    bootstrap_servers: str | None = setting(
        None,
        "KIRA_INTAKE__KAFKA__BOOTSTRAP_SERVERS",
        "KAFKA_BOOTSTRAP_SERVERS",
        description="Broker list.",
    )
    topic: str = setting(
        "alerts", "KIRA_INTAKE__KAFKA__TOPIC", "KAFKA_TOPIC", description="Topic to consume."
    )
    group_id: str = setting(
        "kira", "KIRA_INTAKE__KAFKA__GROUP_ID", "KAFKA_GROUP_ID", description="Consumer group."
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


class IntakeSettings(SettingsGroup):
    """How incidents reach the system."""

    model_config = group_config("intake")

    http_enabled: bool = setting(
        True,
        "KIRA_INTAKE__HTTP_ENABLED",
        "HTTP_INTAKE_ENABLED",
        description="Accept incidents over the HTTP endpoint.",
    )
    annotate_stale_namespaces: bool = setting(
        False,
        "KIRA_INTAKE__ANNOTATE_STALE_NAMESPACES",
        "STALE_ALERT_ANNOTATION_ENABLED",
        description="Mark alerts that may have carried over from a previous incident.",
    )
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)


class ToolSettings(SettingsGroup):
    """Tool-layer limits and kill switches."""

    model_config = group_config("tools")

    disabled: NameSet = setting(
        frozenset(),
        "KIRA_TOOLS__DISABLED",
        description="Tool names withheld from every role. Submission tools are exempt.",
    )
    code_exec_timeout_s: int = setting(
        120,
        "KIRA_TOOLS__CODE_EXEC_TIMEOUT_S",
        "CODE_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a sandboxed code execution.",
    )
    shell_exec_timeout_s: int = setting(
        120,
        "KIRA_TOOLS__SHELL_EXEC_TIMEOUT_S",
        "SHELL_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a sandboxed shell command.",
    )
    kubectl_timeout_s: int = setting(
        120,
        "KIRA_TOOLS__KUBECTL_TIMEOUT_S",
        "KUBECTL_EXEC_TIMEOUT_SECONDS",
        description="Ceiling on a cluster command.",
    )
    fs_sandbox: bool = setting(
        True,
        "KIRA_TOOLS__FS_SANDBOX",
        "TOOL_FS_SANDBOX_ENABLED",
        description="Confine code and shell execution to the incident directory.",
    )
    blind_namespaces: NameSet = setting(
        frozenset(),
        "KIRA_TOOLS__BLIND_NAMESPACES",
        "BLIND_NAMESPACES",
        description="Namespaces removed from telemetry results before a role sees them.",
    )
    host_recovery_nodes: NameSet = setting(
        frozenset(),
        "KIRA_TOOLS__HOST_RECOVERY_NODES",
        "HOST_RECOVERY_ALLOWED_NODES",
        description="Nodes reachable for host-level recovery. Empty disables it.",
    )


class ObservabilitySettings(SettingsGroup):
    """Tracing, logging, and the audit event log."""

    model_config = group_config("observability")

    log_level: str = setting(
        "INFO", "KIRA_OBSERVABILITY__LOG_LEVEL", "LOG_LEVEL", description="Root log level."
    )
    otlp_endpoint: str | None = setting(
        None,
        "KIRA_OBSERVABILITY__OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "COLLECTOR_SERVICE_ADDR",
        description="Collector address. Unset disables export without disabling instrumentation.",
    )
    otlp_insecure: bool = setting(
        True,
        "KIRA_OBSERVABILITY__OTLP_INSECURE",
        "OTEL_EXPORTER_OTLP_INSECURE",
        description="Whether the collector connection skips TLS.",
    )
    metric_interval_ms: int = setting(
        30_000,
        "KIRA_OBSERVABILITY__METRIC_INTERVAL_MS",
        "OTEL_METRIC_EXPORT_INTERVAL_MS",
        description="Metric export period.",
    )
    deployment_id: str | None = setting(
        None,
        "KIRA_OBSERVABILITY__DEPLOYMENT_ID",
        "AGENT_DEPLOYMENT_ID",
        description="Identifies this deployment in traces and in the config snapshot.",
    )
    event_log_enabled: bool = setting(
        True,
        "KIRA_OBSERVABILITY__EVENT_LOG_ENABLED",
        "JSONL_EVENT_LOG_ENABLED",
        description="Write the ordered per-incident event log.",
    )
    event_log_path: Path | None = setting(
        None,
        "KIRA_OBSERVABILITY__EVENT_LOG_PATH",
        "JSONL_EVENT_LOG_PATH",
        description="Where the event log is written. Defaults under the workspace.",
    )


class ServerSettings(SettingsGroup):
    """Process and filesystem placement."""

    model_config = group_config("server")

    host: str = setting("0.0.0.0", "KIRA_SERVER__HOST", "API_HOSTNAME", description="Bind address.")
    port: int = setting(8080, "KIRA_SERVER__PORT", "API_PORT", description="Bind port.")
    workdir: Path = setting(
        Path(".local-workspace"),
        "KIRA_SERVER__WORKDIR",
        "WORKDIR",
        description="Root for incident directories and embedded stores.",
    )
    environment: Literal["local", "production"] = setting(
        "local",
        "KIRA_SERVER__ENVIRONMENT",
        "ENVIRONMENT",
        description="Selects log formatting and access-log filtering.",
    )
    max_concurrent_incidents: int = setting(
        4,
        "KIRA_SERVER__MAX_CONCURRENT_INCIDENTS",
        description=(
            "Incidents processed at once. A private model endpoint serves a small "
            "number of concurrent sequences, so unbounded intake degrades every run "
            "in flight rather than queueing."
        ),
    )
