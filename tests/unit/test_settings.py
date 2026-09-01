"""The settings model: aliases, validation, redaction, and immutability."""

from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import ValidationError

from kira.settings import Settings, VerifySettings, legacy_names_in_use, load_settings
from kira.settings.authority import AuthorityRule, AuthoritySettings
from kira.settings.root import AliasedField, _iter_aliases

ALIASED = list(_iter_aliases(Settings))
IDS = [f.modern for f in ALIASED]


def test_defaults_load() -> None:
    s = load_settings()
    assert s.pipeline.mode == "multi"
    assert s.memory.backend == "embedded"


class TestAliases:
    @pytest.mark.parametrize("field", ALIASED, ids=IDS)
    def test_the_modern_name_still_resolves(
        self, field: AliasedField, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declaring a validation_alias stops env_prefix applying to that field.

        If the prefixed name is not spelled out in the alias list, it silently stops
        working while the legacy name keeps working. That is invisible without this test.
        """
        assert field.modern.startswith("KIRA_"), f"{field.modern} is not a prefixed name"
        probe = _probe_value(field)
        _apply_companions(field, monkeypatch)
        monkeypatch.setenv(field.modern, probe)
        _assert_reached(field, probe)

    @pytest.mark.parametrize("field", ALIASED, ids=IDS)
    def test_every_legacy_name_still_resolves(
        self, field: AliasedField, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing evaluation run scripts export the flat names and must keep working."""
        probe = _probe_value(field)
        _apply_companions(field, monkeypatch)
        for name in field.legacy:
            monkeypatch.setenv(name, probe)
            _assert_reached(field, probe, via=name)
            monkeypatch.delenv(name)


# Companions needed so a probe does not trip an unrelated cross-group rule.
_COMPANIONS: dict[str, dict[str, str]] = {
    "KIRA_MEMORY__ENABLED": {"LEARNING_ENABLED": "false"},
    "KIRA_INTAKE__HTTP_ENABLED": {
        "KAFKA_ENABLED": "true",
        "KAFKA_BOOTSTRAP_SERVERS": "broker:9092",
    },
    "KIRA_INTAKE__KAFKA__ENABLED": {"KAFKA_BOOTSTRAP_SERVERS": "broker:9092"},
    "KIRA_MEMORY__BACKEND": {"MEMORY_POSTGRES_DSN": "postgresql://u:p@db/kira"},
}


# Three shapes the annotation-derived probe cannot synthesise: a tuple of models, a URL,
# and a mapping. Explicit here rather than making the deriver understand pydantic types,
# with the substring the value should be recognisable by once loaded.
_EXPLICIT_PROBES: dict[str, tuple[str, str]] = {
    "KIRA_AUTHORITY__RULES": ('[{"name": "probe7", "mode": "deny"}]', "probe7"),
    "KIRA_AUTHORITY__WEBHOOK_URL": ("https://probe7.example.com/", "probe7.example.com"),
    "KIRA_MEMORY__LIFECYCLE_WEIGHTS": ('{"invalidated": 0.7}', "0.7"),
    "KIRA_TRANSPORT__PROMETHEUS_URL": ("http://probe7.example.com:9090", "probe7.example.com"),
    "KIRA_TRANSPORT__LOKI_URL": ("http://probe7.example.com:3100", "probe7.example.com"),
    "KIRA_TRANSPORT__JAEGER_URL": ("http://probe7.example.com:16686", "probe7.example.com"),
    "KIRA_ADAPTERS": ('{"probe7": {"conductor_base_url": "http://c:8000"}}', "probe7"),
}


def _probe_value(field: AliasedField) -> str:
    """Choose a value valid for this field's type and different from its default.

    Derived from the annotation rather than hand-listed, so a new field is covered the
    moment it is declared.
    """
    if field.modern in _EXPLICIT_PROBES:
        return _EXPLICIT_PROBES[field.modern][0]

    annotation = field.info.annotation
    default = field.info.default
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        annotation = args[0]
        origin = get_origin(annotation)

    if origin in (frozenset, set, list):
        return '["probe7"]'
    if origin is Literal:
        options = [o for o in get_args(annotation) if o != default]
        return str(options[0])
    if annotation is bool:
        return "false" if default is True else "true"
    if annotation is int:
        # Increment rather than a constant: several integer groups carry an ordering
        # rule, and a fixed value would violate it for at least one member.
        return str(default + 1) if isinstance(default, int) else "7"
    if annotation is float:
        return "0.7"
    return "probe7"


def _assert_reached(field: AliasedField, probe: str, via: str | None = None) -> None:
    """Assert the probe value actually landed on the field it names."""
    node: object = load_settings()
    for attr in field.path:
        node = getattr(node, attr)
    rendered = str(node.get_secret_value()) if hasattr(node, "get_secret_value") else str(node)
    explicit = _EXPLICIT_PROBES.get(field.modern)
    wanted = explicit[1] if explicit else probe.strip('[]"')
    source = via or field.modern
    assert wanted.lower() in rendered.lower().replace("\\", "/"), (
        f"{source} did not reach {'.'.join(field.path)}: got {rendered!r}, wanted {probe!r}"
    )


def _apply_companions(field: AliasedField, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _COMPANIONS.get(field.modern, {}).items():
        monkeypatch.setenv(name, value)


class TestValidation:
    def test_postgres_backend_without_a_dsn_fails_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The prototype raised this at first use, which is one incident too late."""
        monkeypatch.setenv("MEMORY_MODE", "postgres")
        with pytest.raises(ValidationError, match="postgres_dsn"):
            load_settings()

    def test_repeat_limits_must_ascend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warn, then refuse, then stop. Out of order, a stage is unreachable."""
        monkeypatch.setenv("LLM_REPEAT_CALL_HARD_LIMIT", "2")
        with pytest.raises(ValidationError, match="repeat_call_soft"):
            load_settings()

    def test_a_second_pass_that_delivers_precedent_needs_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_SECOND_OPINION_ENABLED", "true")
        monkeypatch.setenv("MEMORY_ENABLED", "false")
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        with pytest.raises(ValidationError, match=r"requires memory\.enabled"):
            load_settings()

    def test_the_resampling_control_may_run_without_memory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It is the configuration that deliberately uses no precedent."""
        monkeypatch.setenv("MEMORY_SECOND_OPINION_ENABLED", "true")
        monkeypatch.setenv("MEMORY_SECOND_OPINION_MODE", "resample")
        monkeypatch.setenv("MEMORY_ENABLED", "false")
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        assert load_settings().secondpass.delivery == "none"

    def test_a_service_with_no_intake_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HTTP_INTAKE_ENABLED", "false")
        with pytest.raises(ValidationError, match="no intake is enabled"):
            load_settings()

    def test_kafka_without_brokers_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAFKA_ENABLED", "true")
        with pytest.raises(ValidationError, match="bootstrap_servers"):
            load_settings()

    def test_an_unknown_prefixed_variable_is_a_typo_not_a_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pydantic only looks up names it knows, so a misspelling is otherwise silent.

        The process would start on defaults and behave differently from what the
        operator asked for, with nothing to show for it.
        """
        monkeypatch.setenv("KIRA_MEMORY__ENABLEDD", "false")
        with pytest.raises(ValueError, match="did you mean KIRA_MEMORY__ENABLED"):
            load_settings()


class TestLegacyVocabulary:
    def test_the_misleading_delivery_names_are_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`memory` and `context` were both about memory; `resample` used none."""
        monkeypatch.setenv("MEMORY_SECOND_OPINION_MODE", "memory")
        assert load_settings().secondpass.delivery == "assignment"

    def test_legacy_names_in_use_is_derived_from_the_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hand-maintained, this map would drift from what the code accepts."""
        monkeypatch.setenv("MEMORY_ENABLED", "false")
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        monkeypatch.setenv("MAX_TURNS", "5")
        found = legacy_names_in_use()
        assert found["MEMORY_ENABLED"] == "KIRA_MEMORY__ENABLED"
        assert found["MAX_TURNS"] == "KIRA_PIPELINE__MAX_TURNS"

    def test_the_snapshot_deploy_typo_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deploy script sets a name the prototype never read, so 'off' meant 'on'."""
        monkeypatch.setenv("RCA_AUTO_SNAPSHOT_ENABLED", "false")
        assert load_settings().pipeline.auto_snapshot is False


class TestSecretsAndImmutability:
    def test_secrets_do_not_appear_in_the_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_MODE", "postgres")
        monkeypatch.setenv("MEMORY_POSTGRES_DSN", "postgresql://u:hunter2@db/kira")
        snapshot = load_settings().snapshot()
        assert "hunter2" not in repr(snapshot)

    def test_settings_are_frozen(self) -> None:
        """Injected, never re-read: a run's behaviour cannot change underneath it."""
        s = load_settings()
        with pytest.raises(ValidationError):
            s.pipeline.max_turns = 99  # type: ignore[misc]

    def test_sampling_overrides_are_reported_for_the_run_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecorded partial sampling change made months of runs incomparable."""
        monkeypatch.setenv("LLM_TEMPERATURE", "1.0")
        s = load_settings()
        assert s.sampling_overrides() == {"temperature": 1.0}
        assert s.snapshot()["sampling_overrides"] == ["temperature"]


class TestRealDeployments:
    def test_comma_separated_lists_still_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prototype spelled these as comma lists and deployments still export them.

        Without this, a legacy name is parsed as JSON, fails, and the process dies on a
        value that used to work.
        """
        monkeypatch.setenv("BLIND_NAMESPACES", "chaos-mesh, load-generator ,monitoring")
        monkeypatch.setenv("HOST_RECOVERY_ALLOWED_NODES", "worker-1,worker-2")
        s = load_settings()
        assert s.tools.blind_namespaces == frozenset({"chaos-mesh", "load-generator", "monitoring"})
        assert s.tools.host_recovery_nodes == frozenset({"worker-1", "worker-2"})

    def test_a_json_array_is_also_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KIRA_TOOLS__BLIND_NAMESPACES", '["a", "b"]')
        assert load_settings().tools.blind_namespaces == frozenset({"a", "b"})

    def test_an_existing_evaluation_environment_loads_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shape an existing run script exports. It must keep working verbatim.

        This is the held contract: the evaluation harness sets flat names and knows
        nothing about the new namespace.
        """
        for name, value in {
            "AGENT_MODE": "multi",
            "MAX_TURNS": "20",
            "MAX_REMEDIATION_RETRIES": "2",
            "HITL_MODE": "auto",
            "MEMORY_ENABLED": "true",
            "LEARNING_ENABLED": "true",
            "MEMORY_MODE": "postgres",
            "MEMORY_POSTGRES_DSN": "postgresql://kira:pw@127.0.0.1:5432/kira",
            "MEMORY_POSTGRES_SCHEMA": "agent_memory",
            "MEMORY_EXPERIMENT_ID": "sweep-a",
            "LLM_MODEL": "openai/qwen3.6-35b-a3b",
            "OPENAI_API_BASE": "http://vllm.internal:8000/v1",
            "LLM_MAX_TOKENS": "8192",
            "LLM_TEMPERATURE": "1.0",
            "LLM_TIMEOUT_SECONDS": "1800",
            "VERIFY_STEP_ENABLED": "true",
            "HTTP_INTAKE_ENABLED": "true",
            # The conductor's port in this file, and not this service's. It must not
            # move the bind port: doing so made the service listen where the conductor
            # already was while the harness posted to 8080, and the run failed as a
            # health-check timeout naming nothing.
            "API_PORT": "8000",
            "WORKDIR": "/workspace",
            "LOG_LEVEL": "INFO",
        }.items():
            monkeypatch.setenv(name, value)

        s = load_settings()
        assert s.pipeline.mode == "multi"
        assert s.authority.mode == "auto"
        assert s.memory.backend == "postgres"
        assert s.memory.corpus_id == "sweep-a"
        assert s.verify.enabled is True
        assert s.server.port == 8080
        assert s.model.endpoint_override == "http://vllm.internal:8000/v1"
        # The sampling override is recorded, not silently absorbed.
        assert s.sampling_overrides() == {
            "model": "openai/qwen3.6-35b-a3b",
            "max_tokens": 8192,
            "temperature": 1.0,
        }

    def test_a_retired_knob_is_reported_rather_than_silently_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PROMPT_VARIANT selected between prompt sets that no longer exist."""
        monkeypatch.setenv("PROMPT_VARIANT", "minimal")
        # It is not a recognised name, so it must not resolve to anything, and because
        # it carries no KIRA_ prefix it must not stop the process either.
        assert "PROMPT_VARIANT" not in legacy_names_in_use()
        assert load_settings().pipeline.mode == "multi"


# The prototype performs cluster writes from these two stages.
WRITE_STAGES = ["execution", "single.mitigation"]


class TestRetiredDeliveryMode:
    def test_critique_is_refused_at_startup_with_a_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A startup error beats a composition failure in the middle of an incident."""
        monkeypatch.setenv("MEMORY_SECOND_OPINION_MODE", "critique")
        with pytest.raises(ValidationError, match="not a shipped mechanism"):
            load_settings()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("context", "context"), ("memory", "assignment"), ("resample", "none")],
    )
    def test_the_shipped_modes_still_resolve(
        self, value: str, expected: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MEMORY_SECOND_OPINION_MODE", value)
        assert load_settings().secondpass.delivery == expected


def test_verification_blindness_is_structural_not_configurable() -> None:
    """A flag would imply a sighted mode is supported. It is not.

    A sighted verification derives its checks from the causal chain it is meant to be
    checking, which lets a wrong diagnosis certify its own recovery.
    """
    assert "blind_to_diagnosis" not in VerifySettings.model_fields


def test_a_legacy_run_script_setting_the_retired_flag_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It carries no KIRA_ prefix, so it is ignored rather than fatal."""
    monkeypatch.setenv("VERIFY_BLIND_TO_DIAGNOSIS", "false")
    assert load_settings().verify.enabled is False


def test_a_group_can_be_built_in_code_by_field_name() -> None:
    """A field with a validation_alias is otherwise populated only by that alias.

    Constructing a group by field name then silently yields defaults — no error, just a
    wrong answer. `AuthoritySettings(mode="auto")` returned mode "prompt", which is the
    difference between an evaluation arm running unattended and every write parking on a
    human who is not there.
    """
    built = AuthoritySettings(
        mode="auto", rules=(AuthorityRule(name="r", mode="deny"),), timeout_s=60
    )
    assert built.mode == "auto"
    assert [r.name for r in built.rules] == ["r"]
    assert built.timeout_s == 60
