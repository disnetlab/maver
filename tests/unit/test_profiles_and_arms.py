"""Model profiles and experiment arms: the files a published result is attributed to."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kira.llm.sampling import sampling_digest
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

    @pytest.mark.parametrize("name", SHIPPED_ARMS)
    def test_each_arm_pins_the_snapshot_flag(self, name: str) -> None:
        """A working snapshot raises 8.6 to 15.0 flags against one injected fault, so
        snapshot-on is never neutral context: an arm inheriting the default would vary
        what the diagnosing role is shown without its own file recording that it did."""
        assert "auto_snapshot" in load_arm(name, ARMS).settings.get("pipeline", {})

    def test_the_arms_that_retrieve_pin_the_relevance_floor(self) -> None:
        """Kira floors a hit at 0.5 of its lexical score per query token; the prototype
        shipped that floor off and no campaign set it. An arm inheriting the default
        would be served fewer precedents than the cells it reproduces, and the dose is
        what its memory result is divided by."""
        for name in ("ma-s", "ma-r"):
            memory = load_arm(name, ARMS).settings["memory"]
            assert memory["enabled"] and memory["min_score_per_token"] == 0.0

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


def test_a_profile_without_an_auxiliary_set_gets_a_derived_one() -> None:
    """A helper call is a short structured classification, and the profile's own set is
    tuned for open-ended generation. Running a gate at temperature 1.0 with a presence
    penalty gives a different answer on repeated runs of the same cell, which is a
    research-validity failure rather than a performance one. Deriving rather than
    omitting means every such call is made under a complete, recorded set."""
    profile = load_profile("qwen3.6-35b-a3b", MODELS)
    aux = profile.aux_sampling

    assert aux is not None
    assert aux.temperature == 0.0
    assert (aux.top_p, aux.top_k, aux.min_p) == (None, None, None)
    assert (aux.presence_penalty, aux.frequency_penalty) == (None, None)
    assert (aux.repetition_penalty, aux.no_repeat_ngram_size) == (None, None)
    # The seed and the reasoning budget say which model behaviour is in use, not how
    # widely it may wander, so they are carried rather than cleared.
    assert aux.seed == profile.sampling.seed
    assert aux.thinking_token_budget == profile.sampling.thinking_token_budget


SHIPPED_SAMPLING: dict[str, tuple[str, dict[str, object]]] = {
    "qwen3.6-35b-a3b": (
        "15b241cc5221",
        {
            "max_tokens": 8192,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": None,
            "seed": 42,
            "reasoning_effort": None,
            "thinking_token_budget": None,
        },
    ),
    "qwen3.8-27b": (
        "00a5cc9a2360",
        {
            "max_tokens": 8192,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": None,
            "seed": 42,
            "reasoning_effort": None,
            "thinking_token_budget": None,
        },
    ),
    "claude-sonnet-4-6": (
        "dd9b99700028",
        {
            "max_tokens": 8192,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "frequency_penalty": None,
            "repetition_penalty": None,
            "no_repeat_ngram_size": None,
            "seed": None,
            "reasoning_effort": "medium",
            "thinking_token_budget": None,
        },
    ),
}


@pytest.mark.parametrize("name", sorted(SHIPPED_SAMPLING))
def test_a_shipped_profile_samples_exactly_as_it_was_measured(name: str) -> None:
    """The value of every knob, not merely that every knob has one.

    The existing checks assert the set is *complete*, which a one-character edit to any
    value passes. That is the shape of the failure this project has already paid for: a
    temperature moved from 0 to 1.0 and ran for twelve days before anyone noticed, and
    the eras either side of it are not comparable.

    A profile is a measured constant. Changing what a name samples at is not an edit to
    this file — it is a new profile with a new name, and then this table gains a row
    rather than losing one. The digest is pinned beside the values because it is what a
    run record carries: two campaigns that differ in one knob must not share it.
    """
    expected_digest, expected = SHIPPED_SAMPLING[name]
    values = load_profile(name, MODELS).sampling.model_dump()

    assert values == expected
    assert sampling_digest(values) == expected_digest


def test_every_shipped_profile_is_pinned() -> None:
    """A new profile file joins the table above, rather than shipping unpinned."""
    shipped = {path.stem for path in MODELS.glob("*.yaml")}
    assert shipped == set(SHIPPED_SAMPLING)


@pytest.mark.parametrize("name", sorted(SHIPPED_SAMPLING))
def test_the_auxiliary_set_turns_thinking_off(name: str) -> None:
    """Two modules asserted opposite things about the same value.

    `secondpass/select.py` says thinking must be off for the selector, that the reason is
    a measured failure rather than a preference — with thinking on, the model spent the
    budget reasoning and every selector call in one rerun parsed to empty and fell back to
    the first opinion — and that in Kira this is a property of the auxiliary sampling set.
    The function deriving that set kept the reasoning knobs, and justified keeping them.

    Inert on the profiles every campaign used, which set neither knob; live on any profile
    that sets one, where the gate, the nominator and the selector all ran with reasoning on
    against the module that says they must not.
    """
    aux = load_profile(name, MODELS).aux_sampling

    assert aux.reasoning_effort is None
    assert aux.thinking_token_budget is None
    # The seed survives: it says which behaviour is being used, not how far it may wander.
    assert aux.seed == SHIPPED_SAMPLING[name][1]["seed"]
