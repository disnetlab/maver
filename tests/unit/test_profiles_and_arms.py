"""Model profiles and experiment arms: the files a published result is attributed to."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kira.settings.arms import compare_constants, deep_merge, load_arm
from kira.settings.profiles import ModelProfile, load_profile

MODELS = Path("configs/models")
ARMS = Path("configs/arms")
SHIPPED_PROFILES = ["qwen3.6-35b-a3b", "qwen3.8-27b", "claude-sonnet-4-6"]
SHIPPED_ARMS = ["ma", "ma-v", "ma-s", "ma-r", "resample", "single"]


def _profile_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1,
        "name": "p",
        "model": "openai/m",
        "provider": "vllm",
        "endpoint": {"base_url": "http://x/v1", "api_key_env": None, "api_key_literal": "EMPTY"},
        "sampling": {
            "max_tokens": 8192,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": None,
            "min_p": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
            "seed": 42,
            "reasoning_effort": None,
            "thinking_token_budget": None,
        },
        "limits": {"request_timeout_s": 1800, "num_retries": 3},
        "quirks": {"tool_choice": "auto", "keep_reasoning": False},
        "serving": {"runtime": "vLLM"},
    }
    base.update(overrides)
    return base


class TestShippedProfiles:
    @pytest.mark.parametrize("name", SHIPPED_PROFILES)
    def test_each_profile_loads(self, name: str) -> None:
        assert load_profile(name, MODELS).name == name

    @pytest.mark.parametrize("name", SHIPPED_PROFILES)
    def test_the_sampling_set_is_complete(self, name: str) -> None:
        """Null means omit; it must never mean the knob was forgotten."""
        declared = set(load_profile(name, MODELS).sampling.model_dump())
        assert declared == set(ModelProfile.model_fields["sampling"].annotation.model_fields)

    def test_native_knobs_are_routed_away_from_a_hosted_provider(self) -> None:
        top, extra = (p := load_profile("claude-sonnet-4-6", MODELS)).sampling.request_fields(
            p.quirks.extra_body_fields
        )
        assert extra == {}
        assert "top_k" not in top

    def test_self_hosted_native_knobs_travel_in_the_extension(self) -> None:
        p = load_profile("qwen3.6-35b-a3b", MODELS)
        _, extra = p.sampling.request_fields(p.quirks.extra_body_fields)
        assert set(extra) == {"top_k", "min_p", "repetition_penalty"}

    def test_a_missing_profile_names_what_is_available(self) -> None:
        with pytest.raises(FileNotFoundError, match="available"):
            load_profile("nope", MODELS)


class TestProfileValidation:
    def test_a_forgotten_sampling_knob_fails_to_load(self) -> None:
        """No field has a default, so an omission cannot pass as an intentional null."""
        data = _profile_dict()
        del data["sampling"]["temperature"]  # type: ignore[index]
        with pytest.raises(ValidationError, match="temperature"):
            ModelProfile.model_validate(data)

    def test_native_fields_are_refused_on_a_hosted_route(self) -> None:
        """A hosted route rejects the whole request when it sees one."""
        with pytest.raises(ValidationError, match="no native extension fields"):
            ModelProfile.model_validate(
                _profile_dict(
                    provider="anthropic",
                    quirks={
                        "tool_choice": "auto",
                        "keep_reasoning": False,
                        "extra_body_fields": ["top_k"],
                    },
                )
            )

    def test_a_native_knob_not_declared_for_the_extension_is_refused(self) -> None:
        """It has no field in the schema, so it would be dropped or rejected silently."""
        sampling = dict(_profile_dict()["sampling"])  # type: ignore[arg-type]
        sampling["top_k"] = 20
        with pytest.raises(ValidationError, match="extra_body_fields"):
            ModelProfile.model_validate(_profile_dict(sampling=sampling))

    def test_a_reasoning_budget_must_leave_room_for_an_answer(self) -> None:
        sampling = dict(_profile_dict()["sampling"])  # type: ignore[arg-type]
        sampling["thinking_token_budget"] = 8192
        with pytest.raises(ValidationError, match="below"):
            ModelProfile.model_validate(_profile_dict(sampling=sampling))

    def test_a_real_credential_cannot_be_written_into_a_profile(self) -> None:
        with pytest.raises(ValidationError, match="api_key_literal"):
            ModelProfile.model_validate(
                _profile_dict(
                    endpoint={
                        "base_url": "http://x/v1",
                        "api_key_env": None,
                        "api_key_literal": "sk-live-abc123",
                    }
                )
            )


class TestShippedArms:
    @pytest.mark.parametrize("name", SHIPPED_ARMS)
    def test_each_arm_loads_and_pins_a_real_profile(self, name: str) -> None:
        arm = load_arm(name, ARMS)
        assert arm.name == name
        assert load_profile(arm.model_profile, MODELS)

    @pytest.mark.parametrize("name", SHIPPED_ARMS)
    def test_each_arm_carries_its_provenance(self, name: str) -> None:
        """An arm a result is attributed to must say when and why it was registered."""
        arm = load_arm(name, ARMS)
        assert arm.registered.hypothesis
        assert arm.label

    def test_the_published_arms_are_frozen(self) -> None:
        """A change to a frozen arm changes what its published number means."""
        for name in ("ma", "ma-v", "ma-s", "ma-r", "single"):
            assert load_arm(name, ARMS).frozen, f"{name} carries published results"

    def test_the_treatment_between_ma_and_ma_v_is_verification_alone(self) -> None:
        base = {
            "pipeline": {"max_turns": 20, "remediation_retries": 2},
            "runtime": {"no_tool_nudges": 2},
            "memory": {},
            "verify": {},
            "secondpass": {},
        }
        ma, mav = load_arm("ma", ARMS), load_arm("ma-v", ARMS)
        assert compare_constants(ma, ma.apply_to(base), mav, mav.apply_to(base)) == {}
        assert mav.apply_to(base)["verify"]["enabled"] is True
        assert ma.apply_to(base)["verify"]["enabled"] is False

    def test_a_varying_held_constant_is_reported(self) -> None:
        """Otherwise a paired contrast silently varies two things at once."""
        base = {"pipeline": {"max_turns": 20, "remediation_retries": 2}}
        ma, single = load_arm("ma", ARMS), load_arm("single", ARMS)
        drifted = single.model_copy(update={"model_profile": "qwen3.8-27b"})
        diff = compare_constants(ma, ma.apply_to(base), drifted, drifted.apply_to(base))
        assert diff["model_profile"] == ("qwen3.6-35b-a3b", "qwen3.8-27b")


def test_deep_merge_recurses_and_copies() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overlay = {"a": {"y": 9}}
    assert deep_merge(base, overlay) == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base["a"]["y"] == 2
