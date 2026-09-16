"""The same local Python and YAML composition is used by startup and validation."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from maver import custom
from maver.agents import AgentConfig, AgentRegistry, RCAAgent
from maver.configuration import behavior_digest, load_configuration
from maver.pipelines import DIAGNOSE, PipelineRegistry
from maver.settings.groups import ModelSettings, PipelineSettings, PromptSettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings, known_env_names


def configured(tmp_path: Path, text: str) -> Settings:
    path = tmp_path / "deployment.yaml"
    path.write_text(text)
    return Settings(customisation=path)


def test_default_composition_preserves_legacy_mode_and_has_all_shipped_choices() -> None:
    result = load_configuration(Settings(pipeline=PipelineSettings(mode="single")))
    assert result.pipelines.default == "single"
    assert set(result.pipelines.pipelines) == {"multi", "single", "diagnose"}
    assert set(result.agents) == {"rca", "planning", "execution", "verify", "learning", "single"}
    assert result.pipelines.get("multi").agents["rca"].spec.name == "rca"
    assert result.selectors == ()


def test_yaml_selects_local_python_extensions_and_inherited_profile_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LocalConfig(AgentConfig):
        model_profile: str | None = "qwen3.6-35b-a3b"
        database: str = "orders"

    class LocalAgent(RCAAgent):
        config_type = LocalConfig

    def agents(registry: AgentRegistry) -> None:
        registry.register("local_rca", LocalAgent)

    def workflows(registry: PipelineRegistry) -> None:
        registry.register(replace(DIAGNOSE, name="local_workflow", version="2"))

    monkeypatch.setattr(custom, "register_agents", agents)
    monkeypatch.setattr(custom, "register_pipelines", workflows)
    settings = configured(
        tmp_path,
        """
agents:
  database:
    implementation: local_rca
    config:
      database: catalog
pipelines:
  database_check:
    implementation: local_workflow
    agents:
      rca: database
default_pipeline: database_check
""",
    )
    result = load_configuration(settings)
    chosen = result.pipelines.get("database_check")
    assert result.pipelines.default == "database_check"
    assert chosen.agents["rca"].config.database == "catalog"
    assert chosen.agents["rca"].profile.name == "qwen3.6-35b-a3b"
    assert chosen.workflow.version == "2"
    shown = result.manifest()["pipelines"]["database_check"]
    assert shown["workflow"] == chosen.manifest()["workflow"]
    assert shown["agents"]["rca"]["config"]["database"] == "catalog"
    assert (
        shown["agents"]["rca"]["profile"]["sampling"]["max_tokens"]
        == chosen.agents["rca"].profile.sampling.max_tokens
    )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("pipline: {}", "Extra inputs"),
        ("default_pipeline: missing", "not configured"),
        ("default_pipeline: multi\ndefault_pipeline: single", "duplicate YAML key"),
        ("agents:\n  rca:\n    implementation: os.system", "unknown agent implementation"),
        ("pipelines:\n  multi:\n    implementation: multi\n    edges: []", "Extra inputs"),
        (
            "agents:\n  rca:\n    implementation: rca\n    config:\n      invented: true",
            "Extra inputs",
        ),
        (
            "agents:\n  rca:\n    implementation: rca\n    config:\n      limits:\n        runtime:\n          repeat_call_soft: 100",
            "repeat_call_soft",
        ),
    ],
)
def test_invalid_compositions_fail_during_load(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises((ValueError, ValidationError), match=match):
        load_configuration(configured(tmp_path, text))


def test_named_file_is_not_silently_ignored_and_customisation_env_is_recognised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "missing.yaml"
    monkeypatch.setenv("MAVER_CUSTOMISATION", str(path))
    assert "MAVER_CUSTOMISATION" in known_env_names()
    settings = Settings()
    assert settings.customisation == path
    with pytest.raises(FileNotFoundError):
        load_configuration(settings)


def test_code_built_deployment_can_supply_its_profile_without_rereading_a_file(
    tmp_path: Path,
) -> None:
    profile = load_profile("qwen3.6-35b-a3b", Path("configs/models"))
    settings = Settings(model=ModelSettings(profile_dir=tmp_path / "not-installed"))
    result = load_configuration(settings, default_profile=profile)
    assert result.pipelines.default == "multi"


def test_invalid_selector_registration_fails_before_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(custom, "pipeline_selectors", lambda: (object(),))
    with pytest.raises(TypeError, match="implement select"):
        load_configuration(Settings())


def test_human_readable_manifest_masks_custom_secrets_but_keeps_token_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PrivateConfig(AgentConfig):
        api_token: str

    class PrivateAgent(RCAAgent):
        config_type = PrivateConfig

    monkeypatch.setattr(
        custom, "register_agents", lambda registry: registry.register("private", PrivateAgent)
    )
    settings = configured(
        tmp_path,
        "agents:\n  private:\n    implementation: private\n    config:\n      api_token: synthetic-example-value\n      model_profile: qwen3.6-35b-a3b\n",
    )
    shown = load_configuration(settings).manifest()["agents"]["private"]
    assert shown["config"]["api_token"] == "**********"
    assert shown["profile"]["sampling"]["thinking_token_budget"] is None
    assert isinstance(shown["profile"]["sampling"]["max_tokens"], int)


def test_resume_identity_tracks_prompt_and_behavior_but_not_default_selection(
    tmp_path: Path,
) -> None:
    roles = tmp_path / "roles"
    roles.mkdir()
    prompt = roles / "rca.txt"
    prompt.write_text("First prompt.")
    settings = Settings(prompts=PromptSettings(override_dir=tmp_path))
    first = load_configuration(settings).pipelines.get("multi")
    changed_default = load_configuration(
        settings.model_copy(
            update={"pipeline": settings.pipeline.model_copy(update={"mode": "single"})}
        )
    ).pipelines.get("multi")
    assert first.digest == changed_default.digest
    changed_budget = load_configuration(
        settings.model_copy(
            update={"pipeline": settings.pipeline.model_copy(update={"max_turns": 3})}
        )
    ).pipelines.get("multi")
    assert first.digest != changed_budget.digest
    prompt.write_text("Second prompt.")
    changed_prompt = load_configuration(settings).pipelines.get("multi")
    assert first.digest != changed_prompt.digest


def test_behavior_digest_tracks_effective_endpoint_but_not_configuration_location(
    tmp_path: Path,
) -> None:
    profile = load_profile("qwen3.6-35b-a3b", Path("configs/models"))
    settings = Settings()
    first = behavior_digest(settings, profile)
    relocated = settings.model_copy(
        update={
            "customisation": tmp_path / "elsewhere.yaml",
            "server": settings.server.model_copy(update={"workdir": tmp_path}),
        }
    )
    assert behavior_digest(relocated, profile) == first
    moved_endpoint = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={"endpoint_override": "http://new-local-model:8000/v1"}
            )
        }
    )
    assert behavior_digest(moved_endpoint, profile) != first
