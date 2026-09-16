"""The settings model: aliases, validation, redaction, and immutability."""

import logging
import re
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
import yaml
from pydantic import SecretStr, ValidationError

from maver.orchestration.routing import RoutingPolicy
from maver.settings import Settings, VerifySettings, legacy_names_in_use, load_settings
from maver.settings.arms import arm_digest, load_arm
from maver.settings.authority import AuthorityRule, AuthoritySettings
from maver.settings.groups import PipelineSettings
from maver.settings.root import (
    AliasedField,
    _iter_aliases,
    arm_overrides,
    reset_settings,
    unknown_prefixed_names,
)

ALIASED = list(_iter_aliases(Settings))
IDS = [f.modern for f in ALIASED]
SECRETS = [f for f in ALIASED if SecretStr in get_args(f.info.annotation)]
SECRET_IDS = [f.modern for f in SECRETS]


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
        assert field.modern.startswith("MAVER_"), f"{field.modern} is not a prefixed name"
        if reason := _ONE_VALID_VALUE.get(field.modern):
            pytest.skip(reason)
        probe = _probe_value(field)
        _apply_companions(field, monkeypatch)
        monkeypatch.setenv(field.modern, probe)
        _assert_reached(field, probe)

    @pytest.mark.parametrize("field", ALIASED, ids=IDS)
    def test_every_legacy_name_still_resolves(
        self, field: AliasedField, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing evaluation run scripts export the flat names and must keep working."""
        if reason := _ONE_VALID_VALUE.get(field.modern):
            pytest.skip(reason)
        probe = _probe_value(field)
        _apply_companions(field, monkeypatch)
        for name in field.legacy:
            monkeypatch.setenv(name, probe)
            _assert_reached(field, probe, via=name)
            monkeypatch.delenv(name)


# Companions needed so a probe does not trip an unrelated cross-group rule.
_COMPANIONS: dict[str, dict[str, str]] = {
    "MAVER_MEMORY__ENABLED": {"LEARNING_ENABLED": "false"},
    "MAVER_MEMORY__BACKEND": {"MEMORY_POSTGRES_DSN": "postgresql://u:p@db/maver"},
}

# A probe flips a boolean away from its default, which needs the other value to be a
# configuration the settings will accept. These two have only one, and naming them here
# is the honest way to say so: until something consumes from the queue, the queue intake
# cannot be turned on and the HTTP intake cannot be turned off.
_ONE_VALID_VALUE: dict[str, str] = {
    "MAVER_INTAKE__KAFKA__ENABLED": "the queue intake is unbuilt, so enabling it is refused",
    "MAVER_INTAKE__HTTP_ENABLED": "with no queue intake to fall back to, it cannot be off",
}


# Three shapes the annotation-derived probe cannot synthesise: a tuple of models, a URL,
# and a mapping. Explicit here rather than making the deriver understand pydantic types,
# with the substring the value should be recognisable by once loaded.
_EXPLICIT_PROBES: dict[str, tuple[str, str]] = {
    # A real arm, because the name is resolved during the load rather than stored.
    "MAVER_ARM": ("ma-v", "ma-v"),
    "MAVER_AUTHORITY__RULES": ('[{"name": "probe7", "mode": "deny"}]', "probe7"),
    "MAVER_AUTHORITY__WEBHOOK_URL": ("https://probe7.example.com/", "probe7.example.com"),
    "MAVER_MEMORY__LIFECYCLE_WEIGHTS": ('{"invalidated": 0.7}', "0.7"),
    "MAVER_TRANSPORT__PROMETHEUS_URL": ("http://probe7.example.com:9090", "probe7.example.com"),
    "MAVER_TRANSPORT__LOKI_URL": ("http://probe7.example.com:3100", "probe7.example.com"),
    "MAVER_TRANSPORT__JAEGER_URL": ("http://probe7.example.com:16686", "probe7.example.com"),
    "MAVER_ADAPTERS": ('{"probe7": {"conductor_base_url": "http://c:8000"}}', "probe7"),
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


def test_the_turn_budget_defaults_to_what_every_campaign_ran() -> None:
    """Twenty was the prototype's default and its deployment overlay raised it on every
    host, so twenty is a value no published result was ever produced under. Sixteen of
    sixteen archived environments set one hundred, and none set anything else."""
    assert Settings().pipeline.max_turns == 100


def test_the_relevance_floor_and_the_query_filter_default_to_what_was_measured() -> None:
    """No host's environment ever set MEMORY_MIN_TEXT_RANK_PER_TOKEN, so every measured
    retrieval ran without a floor; the boilerplate filter was on for two of the four
    September hosts and off for two, so the prototype's own default stands."""
    assert Settings().memory.min_score_per_token == 0.0
    assert Settings().memory.second_pass.drop_boilerplate is False


def test_the_retrieval_depth_defaults_to_what_the_campaigns_ran() -> None:
    """Five was the prototype's code default and the depth nothing recent measured. The
    archived environment of the campaigns this build reproduces sets
    MEMORY_SEARCH_TOP_K=3 and both frozen memory arms pin three, so a deployment left on
    five reads two records more per retrieval than anything the results describe."""
    assert Settings().memory.top_k == 3


class TestExperienceAdoption:
    """A past incident reaches a present one two ways, and they share one store."""

    def test_both_adoption_paths_live_in_one_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A role calling a retrieval tool and the gated second opinion are the two
        paths, and they were split across two top-level groups — which made a pair that
        draws on one store, and whose invariants are joint, look independent.
        """
        monkeypatch.setenv("MAVER_MEMORY__ROLE_RETRIEVAL", "false")
        monkeypatch.setenv("MAVER_MEMORY__SECOND_PASS__ENABLED", "true")
        monkeypatch.setenv("MAVER_MEMORY__SECOND_PASS__GATE", "nominator")
        memory = load_settings().memory
        assert (memory.role_retrieval, memory.second_pass.enabled) == (False, True)
        assert memory.second_pass.gate == "nominator"
        assert not hasattr(load_settings(), "secondpass")

    def test_the_prototype_names_still_reach_the_nested_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regrouping is ours to do; the run scripts are not."""
        monkeypatch.setenv("MEMORY_SECOND_OPINION_ENABLED", "true")
        monkeypatch.setenv("MEMORY_SECOND_OPINION_QUALITY_MAX_TOKENS", "1200")
        second_pass = load_settings().memory.second_pass
        assert (second_pass.enabled, second_pass.gate_max_tokens) == (True, 1200)

    def test_a_path_with_no_readable_store_behind_it_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checked by the group that holds both, since both draw on the same store."""
        monkeypatch.setenv("MAVER_MEMORY__ENABLED", "false")
        monkeypatch.setenv("MAVER_MEMORY__SECOND_PASS__ENABLED", "true")
        monkeypatch.setenv("MAVER_MEMORY__SECOND_PASS__DELIVERY", "context")
        with pytest.raises(ValidationError, match="has no precedent to deliver"):
            load_settings()


class TestValidation:
    def test_postgres_backend_without_a_dsn_fails_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The prototype raised this at first use, which is one incident too late."""
        monkeypatch.setenv("MEMORY_MODE", "postgres")
        with pytest.raises(ValidationError, match="postgres_dsn"):
            load_settings()

    def test_a_blank_credential_is_unset_rather_than_an_empty_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An env file that declares the name and leaves the value off passed this rule,
        because a set-but-empty variable is not None. What it produced was an empty
        connection string handed to the driver, which fails at the first incident naming
        nothing an operator can act on."""
        monkeypatch.setenv("MEMORY_MODE", "postgres")
        monkeypatch.setenv("MEMORY_POSTGRES_DSN", "")
        with pytest.raises(ValidationError, match="postgres_dsn"):
            load_settings()

    def test_a_credential_that_cannot_be_sent_is_refused_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key pasted with a newline in it reaches the provider verbatim and comes
        back as a 401 minutes into a run, where it reads as the model failing."""
        monkeypatch.setenv("MAVER_MODEL__API_KEY", "sk-first-half\nsk-second-half")
        with pytest.raises(ValidationError, match="control character"):
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
        assert load_settings().memory.second_pass.delivery == "none"

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
        monkeypatch.setenv("MAVER_MEMORY__ENABLEDD", "false")
        with pytest.raises(ValueError, match="did you mean MAVER_MEMORY__ENABLED"):
            load_settings()

    def test_the_built_revision_is_not_read_as_a_typo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The image sets MAVER_REVISION for /healthz to report, and answers to no field.

        Swept up as a misspelling, it stopped the container that set it from starting.
        """
        monkeypatch.setenv("MAVER_REVISION", "9f3c1ab")
        assert unknown_prefixed_names() == []
        assert load_settings().server.port == 8080

    def test_an_unprefixed_group_name_is_not_a_second_control_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep above only looks at prefixed names, so anything the root answered
        to without the prefix was live and invisible at once: it never reached the
        typo check, never reached the run's record, and under an arm it lost silently
        while the operator read the value they had exported. A bare ``MODEL`` is also
        an ordinary name in a container that serves one, and it aborted startup.
        """
        monkeypatch.setenv("MODEL", "some-value")
        monkeypatch.setenv("MEMORY__TOP_K", "9")
        settings = load_settings()
        assert settings.memory.top_k == 3
        assert unknown_prefixed_names() == []


class TestArmFiles:
    """An arm file is only a document until something resolves it into the settings."""

    def test_a_named_arm_reaches_the_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every key in configs/arms/ was inert: nothing read the files at runtime.

        The result was a published configuration that could not be selected, while two
        shipped artefacts told a reader it could.
        """
        monkeypatch.setenv("MAVER_ARM", "ma-s")
        s = load_settings()
        assert (s.arm, s.pipeline.mode) == ("ma-s", "multi")
        assert (s.memory.second_pass.enabled, s.memory.second_pass.gate) == (True, "quality")
        assert (s.memory.top_k, s.memory.role_retrieval) == (3, False)

    def test_the_environment_outranks_the_arm_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ConfigMap is the deployment control surface and an arm file is not.

        Merging an arm as constructor arguments inverts this, because init outranks
        every environment source. It also inverts it one level down, where each group
        reads its own environment and a mapping from the parent arrives as that group's
        own init arguments.
        """
        monkeypatch.setenv("MAVER_ARM", "ma-s")
        monkeypatch.setenv("MAVER_MEMORY__TOP_K", "9")
        s = load_settings()
        assert s.memory.top_k == 9
        assert s.memory.second_pass.gate == "quality", "the rest of the arm still applies"

    def test_a_legacy_name_outranks_the_arm_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Required, not incidental: the evaluation run scripts export the flat names.

        An arm file that could silently override one would change what a campaign
        measured while the operator read the value they exported.
        """
        monkeypatch.setenv("MAVER_ARM", "ma-s")
        monkeypatch.setenv("MEMORY_SEARCH_TOP_K", "9")
        assert load_settings().memory.top_k == 9

    def test_an_override_of_a_held_constant_setting_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Held constant is what licenses reading a two-arm gap as single-variable."""
        monkeypatch.setenv("MAVER_ARM", "ma")
        monkeypatch.setenv("MAX_TURNS", "8")
        with pytest.raises(ValueError, match="holds constant"):
            load_settings()

    def test_a_held_constant_path_the_arm_does_not_write_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every shipped arm holds the model profile and ``runtime.*`` constant and
        declares neither, so a claim enforced only where the file happens to write is a
        claim enforced nowhere: the arm's own digest went into the run record while the
        campaign ran on another model."""
        monkeypatch.setenv("MAVER_ARM", "ma-s")
        monkeypatch.setenv("MAVER_MODEL__PROFILE", "qwen3.8-27b")
        with pytest.raises(ValueError, match=r"model\.profile"):
            load_settings()
        monkeypatch.delenv("MAVER_MODEL__PROFILE")
        monkeypatch.setenv("LLM_REPEAT_CALL_SOFT_LIMIT", "4")
        with pytest.raises(ValueError, match=r"runtime\.repeat_call_soft"):
            load_settings()

    def test_an_override_outside_the_held_constant_set_is_reported_not_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Overriding one key without editing a frozen file is the deployment contract
        working, so it belongs in the report rather than in a refusal."""
        monkeypatch.setenv("MAVER_ARM", "ma")
        monkeypatch.setenv("MAVER_MEMORY__TOP_K", "9")
        assert arm_overrides(load_settings()).allowed["memory.top_k"] == (3, 9)

    def test_the_arm_reaches_the_run_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A frozen arm carries a published number, so a later edit has to be detectable."""
        monkeypatch.setenv("MAVER_ARM", "ma-v")
        snapshot = load_settings().snapshot()
        assert snapshot["settings"]["arm"] == "ma-v"
        assert snapshot["arm_digest"] == arm_digest("ma-v", Path("configs/arms"))

    def test_a_misnamed_arm_is_a_startup_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise the run is attributed to an arm whose settings never applied."""
        monkeypatch.setenv("MAVER_ARM", "ma-x")
        with pytest.raises(FileNotFoundError, match="available"):
            load_settings()

    def test_an_arm_key_nothing_reads_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The root ignores extra keys, so a mistyped path would apply nothing, quietly."""
        arm = yaml.safe_load(Path("configs/arms/ma.yaml").read_text())
        arm["settings"]["memory"]["enabldd"] = False
        (tmp_path / "ma.yaml").write_text(yaml.safe_dump(arm))
        monkeypatch.setenv("MAVER_ARM", "ma")
        monkeypatch.setenv("MAVER_ARM_DIR", str(tmp_path))
        with pytest.raises(ValueError, match=r"memory\.enabldd, which is not a setting"):
            load_settings()


class TestLegacyVocabulary:
    def test_the_misleading_delivery_names_are_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`memory` and `context` were both about memory; `resample` used none."""
        monkeypatch.setenv("MEMORY_SECOND_OPINION_MODE", "memory")
        assert load_settings().memory.second_pass.delivery == "assignment"

    def test_legacy_names_in_use_is_derived_from_the_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hand-maintained, this map would drift from what the code accepts."""
        monkeypatch.setenv("MEMORY_ENABLED", "false")
        monkeypatch.setenv("LEARNING_ENABLED", "false")
        monkeypatch.setenv("MAX_TURNS", "5")
        found = legacy_names_in_use()
        assert found["MEMORY_ENABLED"] == "MAVER_MEMORY__ENABLED"
        assert found["MAX_TURNS"] == "MAVER_PIPELINE__MAX_TURNS"

    def test_the_snapshot_deploy_typo_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deploy script sets a name the prototype never read, so 'off' meant 'on'."""
        monkeypatch.setenv("RCA_AUTO_SNAPSHOT_ENABLED", "false")
        assert load_settings().pipeline.auto_snapshot is False


class TestSecretsAndImmutability:
    def test_secrets_do_not_appear_in_the_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_MODE", "postgres")
        monkeypatch.setenv("MEMORY_POSTGRES_DSN", "postgresql://u:hunter2@db/maver")
        snapshot = load_settings().snapshot()
        assert "hunter2" not in repr(snapshot)

    def test_an_adapters_credential_is_masked_like_every_other(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The slice is untyped so nothing in it is a SecretStr, and this record goes
        into every run and is echoed by a command documented as safe to paste."""
        monkeypatch.setenv("MAVER_ADAPTERS__SREGYM__API_TOKEN", "tok-hunter2")
        monkeypatch.setenv("MAVER_ADAPTERS__SREGYM__CONDUCTOR_BASE_URL", "http://c:8000")
        adapters = load_settings().snapshot()["settings"]["adapters"]
        assert adapters["sregym"]["api_token"] == "**********"
        assert adapters["sregym"]["conductor_base_url"] == "http://c:8000"

    @pytest.mark.parametrize("field", SECRETS, ids=SECRET_IDS)
    def test_a_quoted_secret_is_read_without_its_quotes(
        self, field: AliasedField, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `.env` line written KEY="value" is carried into the process with the quotes
        in the value, and both of these are then used as written: one authenticates
        against the provider, the other is handed to the database driver. Cleaning at the
        declaration rather than at the two call sites is what covers the next secret."""
        _apply_companions(field, monkeypatch)
        monkeypatch.setenv(field.modern, '"probe7"')

        node: object = load_settings()
        for attr in field.path:
            node = getattr(node, attr)

        assert node is not None and node.get_secret_value() == "probe7"

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
        monkeypatch.setenv("MAVER_TOOLS__BLIND_NAMESPACES", '["a", "b"]')
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
            "MEMORY_POSTGRES_DSN": "postgresql://maver:pw@127.0.0.1:5432/maver",
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
        # it carries no MAVER_ prefix it must not stop the process either.
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
        assert load_settings().memory.second_pass.delivery == expected


def test_verification_blindness_is_structural_not_configurable() -> None:
    """A flag would imply a sighted mode is supported. It is not.

    A sighted verification derives its checks from the causal chain it is meant to be
    checking, which lets a wrong diagnosis certify its own recovery.
    """
    assert "blind_to_diagnosis" not in VerifySettings.model_fields


def test_a_legacy_run_script_setting_the_retired_flag_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It carries no MAVER_ prefix, so it is ignored rather than fatal."""
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


def test_an_approval_deadline_that_has_already_passed_is_refused() -> None:
    """A pause exists so a person can answer it. Zero and negative values were accepted,
    and a request raised under one expires before the notification carrying it lands:
    every write is denied by timeout, in a deployment configured to ask a human. Waiting
    forever is spelled null, which is why zero cannot mean it."""
    with pytest.raises(ValidationError, match="at least 1 second"):
        AuthoritySettings(timeout_s=0)

    assert AuthoritySettings(timeout_s=None).timeout_s is None


def test_a_deployment_that_can_reach_no_node_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The allowlist is empty by default, the host tool is offered to the remediating
    role regardless, and every call it makes is refused. Nothing said so, and a whole
    campaign was measured with node-level faults unfixable and graded as the agent
    failing to fix them."""
    with caplog.at_level(logging.WARNING, logger="maver.settings.root"):
        settings = load_settings()

    assert settings.tools.host_recovery_nodes == frozenset()
    assert any("MAVER_TOOLS__HOST_RECOVERY_NODES" in r.getMessage() for r in caplog.records)
    assert settings.snapshot()["settings"]["tools"]["host_recovery_nodes"] == []


def test_a_configured_node_leaves_the_warning_unsaid(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A warning on every start is one an operator learns to scroll past."""
    monkeypatch.setenv("HOST_RECOVERY_ALLOWED_NODES", "worker-1")
    with caplog.at_level(logging.WARNING, logger="maver.settings.root"):
        load_settings()

    assert not any("HOST_RECOVERY" in r.getMessage() for r in caplog.records)


def test_an_arm_may_not_decide_who_authorises_a_cluster_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every shipped arm asked for unattended approval, because a measured campaign
    cannot park on a person. That was inert while nothing read arm files, so wiring them
    up made one line in a research configuration disable the approval gate on a process
    holding cluster write credentials — and `MAVER_ARM=<anything>` was enough to do it.

    No archived campaign supplied the value that way; all twenty-three set it in the
    environment. So the key is refused where it does not belong rather than warned
    about where it does not work."""
    arm = tmp_path / "unattended.yaml"
    arm.write_text(
        "schema_version: 1\n"
        "name: unattended\n"
        "label: Unattended\n"
        "description: an arm that turns the gate off\n"
        "model_profile: qwen3.6-35b-a3b\n"
        "registered: {date: '2026-09-03', campaign: none, hypothesis: none}\n"
        "settings:\n"
        "  authority: {mode: auto}\n"
    )
    with pytest.raises(ValidationError, match="set MAVER_AUTHORITY__MODE in the environment"):
        load_arm("unattended", tmp_path)

    monkeypatch.setenv("MAVER_ARM", "ma-s")
    reset_settings()
    assert load_settings().authority.mode == "prompt"


def test_no_shipped_arm_carries_an_authority_stanza() -> None:
    """The refusal above is only worth having if the files it governs comply."""
    for path in sorted(Path("configs/arms").glob("*.yaml")):
        assert "authority" not in yaml.safe_load(path.read_text()).get("settings", {}), path.name


def test_production_may_not_run_every_cluster_write_unattended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production deployment holding write credentials must not approve itself.

    Nothing enforced this, so a production process with an unattended mode started
    clean with the whole approval mechanism inert underneath it. The escape hatch is a
    rule rather than a second flag: a deployment can say that routine scaling proceeds
    unattended, and cannot say that everything does.
    """
    monkeypatch.setenv("MAVER_SERVER__ENVIRONMENT", "production")
    monkeypatch.setenv("MAVER_AUTHORITY__MODE", "auto")
    with pytest.raises(ValidationError, match="would let every cluster write proceed unreviewed"):
        Settings()

    monkeypatch.setenv("MAVER_AUTHORITY__MODE", "prompt")
    monkeypatch.setenv(
        "MAVER_AUTHORITY__RULES",
        '[{"name": "routine-scaling", "selectors": ["verb:scale"], "mode": "auto"}]',
    )
    settings = Settings()
    assert [rule.mode for rule in settings.authority.rules] == ["auto"]


def test_an_unattended_benchmark_is_still_allowed_outside_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal above is scoped to the posture that makes it dangerous.

    Every measured campaign ran unattended, and a check that stopped one from starting
    would make the artifact unable to reproduce its own results.
    """
    monkeypatch.setenv("MAVER_AUTHORITY__MODE", "auto")
    assert Settings().authority.mode == "auto"


def test_a_queue_intake_that_consumes_nothing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check that refuses a service with no intake was satisfiable by a flag that
    builds no consumer.

    Nothing under `src/` reads from the topic, so a deployment with the HTTP intake off
    and a broker configured validated, started clean, and could never receive an
    incident — which is, word for word, the outcome that check exists to prevent. A flag
    that answers a liveness question without supplying the liveness is worse than one
    that is absent.
    """
    monkeypatch.setenv("MAVER_INTAKE__HTTP_ENABLED", "false")
    monkeypatch.setenv("MAVER_INTAKE__KAFKA__ENABLED", "true")
    monkeypatch.setenv("MAVER_INTAKE__KAFKA__BOOTSTRAP_SERVERS", "broker:9092")

    with pytest.raises(ValidationError, match="queue intake is not implemented"):
        Settings()


def test_the_default_deployment_can_still_receive_an_incident() -> None:
    """The control: refusing the queue must not refuse the configuration everything uses."""
    assert Settings().intake.http_enabled is True


class TestAFrozenArmAndTheSamplingKnobs:
    """A sampling override displaces the profile without touching any path an arm
    declares, so the contradiction check could not see it: an arm holding the model
    profile constant would run at a different temperature and report nothing wrong.

    That is the exact shape of a knob that moved once in this project's history and went
    unnoticed for twelve days, leaving the campaigns either side of it not comparable —
    and the arm mechanism exists to make exactly that impossible.
    """

    def test_an_override_on_a_frozen_arm_is_a_startup_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MAVER_ARM", "ma-s")
        monkeypatch.setenv("LLM_TEMPERATURE", "0")
        reset_settings()

        with pytest.raises(ValueError, match=re.escape("model.overrides.temperature")):
            load_settings()

    def test_the_same_override_without_an_arm_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control. Overrides exist so a run script exporting a flat name keeps
        working; what an arm adds is that it may not do so silently."""
        monkeypatch.setenv("LLM_TEMPERATURE", "0")
        reset_settings()

        assert load_settings().sampling_overrides() == {"temperature": 0.0}


def test_verify_handback_setting_inherits_or_tightens_the_shared_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(pipeline=PipelineSettings(remediation_retries=4))
    assert settings.verify.handback_retries is None
    assert RoutingPolicy.from_settings(settings).verify_handback_limit == 4
    monkeypatch.setenv("MAVER_VERIFY__HANDBACK_RETRIES", "1")
    configured = Settings(pipeline=PipelineSettings(remediation_retries=4))
    assert RoutingPolicy.from_settings(configured).verify_handback_limit == 1
    raised = configured.model_copy(update={"verify": VerifySettings(handback_retries=9)})
    assert RoutingPolicy.from_settings(raised).verify_handback_limit == 4
    with pytest.raises(ValidationError):
        VerifySettings(handback_retries=-1)
