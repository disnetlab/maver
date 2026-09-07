"""The AIOpsLab adapter's configuration, and the half of the protocol it refuses.

Two things are pinned here. That a misconfigured bridge is a startup failure naming the
file, because the prototype's version of this returned an empty token, counted the bridge
as unconfigured, removed every cluster tool from every role, and let the run burn its
whole turn budget on tools it could not see. And that the lifecycle methods refuse rather
than answer, because the environment is held by another process and a plausible-looking
answer from this one would be a lie about what is deployed.
"""

import pathlib
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from kira.adapters.aiopslab import ADAPTER, ADAPTER_NAME, AiopsLabAdapter, AiopsLabConfig
from kira.adapters.base import LifecycleRunsElsewhere
from kira.adapters.registry import available_adapters, get_adapter
from kira.settings.groups import ServerSettings
from kira.settings.root import Settings

TOKEN = "a-bridge-token-long-enough"


def settings(tmp_path: pathlib.Path, **slice_: Any) -> Settings:
    return Settings(
        server=ServerSettings(workdir=tmp_path),
        adapters={ADAPTER_NAME: slice_} if slice_ else {},
    )


def test_the_entry_point_registers_the_adapter_and_the_factory_builds_it(
    tmp_path: pathlib.Path,
) -> None:
    """Discovery is not selection: it is installed and reported, and enabling it is a
    settings decision. A registered adapter that could not be built would only be found
    out when a deployment asked for it."""
    assert available_adapters()[ADAPTER_NAME] == "ready"
    assert get_adapter(ADAPTER_NAME) is ADAPTER

    built = ADAPTER(settings(tmp_path))

    assert isinstance(built, AiopsLabAdapter)
    assert built.name == ADAPTER_NAME
    # No episode route, which is what says the environment is held elsewhere and this
    # adapter is bound for the whole process instead.
    assert built.routes() is None


def test_a_short_or_unreadable_token_file_fails_at_startup_naming_the_path(
    tmp_path: pathlib.Path,
) -> None:
    """The recorded defect, in both of its forms.

    The prototype swallowed the read error and returned an empty token, which made the
    bridge count as unconfigured; every cluster tool then vanished from every role's
    toolset with nothing said, and the model was handed a tool list it could not
    investigate with. A truncated read did the same thing, more quietly.
    """
    short = tmp_path / "short.token"
    short.write_text("abc\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=str(short)):
        AiopsLabConfig(enabled=True, bridge_token_file=short)

    missing = tmp_path / "absent.token"
    with pytest.raises(ValidationError, match=str(missing)):
        AiopsLabConfig(enabled=True, bridge_token_file=missing)

    with pytest.raises(ValidationError, match="at least 16"):
        AiopsLabConfig(enabled=True)


def test_a_token_file_is_read_and_an_explicit_token_wins(tmp_path: pathlib.Path) -> None:
    """A deployment that names both means the one it typed."""
    path = tmp_path / "bridge.token"
    path.write_text(f"  {TOKEN}\n", encoding="utf-8")

    from_file = AiopsLabConfig(enabled=True, bridge_token_file=path)
    assert from_file.bridge_token is not None
    assert from_file.bridge_token.get_secret_value() == TOKEN

    explicit = AiopsLabConfig(
        enabled=True, bridge_token=SecretStr("an-explicit-bridge-token"), bridge_token_file=path
    )
    assert explicit.bridge_token is not None
    assert explicit.bridge_token.get_secret_value() == "an-explicit-bridge-token"


def test_a_process_not_serving_a_run_is_not_made_to_hold_a_credential(
    tmp_path: pathlib.Path,
) -> None:
    """The refusal is about a run that meant to reach the bridge. A deployment carrying
    the slice without switching it on is not one, and must still start."""
    config = AiopsLabConfig(bridge_token_file=tmp_path / "absent.token")

    assert config.enabled is False
    assert config.bridge_token is None


def test_the_prototypes_own_names_still_decide_these_values(tmp_path: pathlib.Path) -> None:
    """A deployment file written for the prototype keeps working, and says which
    spelling decided each value in the log."""
    config = AiopsLabConfig.resolve(
        settings(tmp_path, enabled=True),
        env={
            "AIOPSLAB_BRIDGE_URL": "http://127.0.0.1:9999",
            "AIOPSLAB_BRIDGE_TOKEN": TOKEN,
            "AIOPSLAB_BRIDGE_TIMEOUT_SECONDS": "45",
        },
    )

    assert config.bridge_url == "http://127.0.0.1:9999"
    assert config.bridge_token is not None
    assert config.bridge_token.get_secret_value() == TOKEN
    assert config.bridge_timeout_s == 45.0


def test_the_slice_refuses_a_key_it_does_not_know(tmp_path: pathlib.Path) -> None:
    """A typo inside the slice is a startup failure rather than a default that quietly
    changes what an arm means."""
    with pytest.raises(ValidationError, match="bridge_tokenn"):
        AiopsLabConfig.resolve(settings(tmp_path, bridge_tokenn=TOKEN))


def test_the_application_facts_are_carried_verbatim() -> None:
    """Deployment metadata, keyed only by namespace: no task, no fault, no diagnosis.

    Verify needs a reliable user-facing Service to probe and must stay blind to the
    incident's causal answer, so these three entries are the whole of what it is told
    and they are the values every archived AIOpsLab run was measured with.
    """
    contracts = AiopsLabConfig().connectivity_contracts

    assert {ns: c["http_entry_point"] for ns, c in contracts.items()} == {
        "test-social-network": {"service": "nginx-thrift", "port": 8080, "path": "/"},
        "test-hotel-reservation": {"service": "frontend", "port": 5000, "path": "/"},
        "astronomy-shop": {"service": "frontend-proxy", "port": 8080, "path": "/"},
    }


async def test_every_lifecycle_method_says_where_the_environment_is_held(
    tmp_path: pathlib.Path,
) -> None:
    """This process runs the pipeline and nothing else.

    An adapter split across two processes has to refuse the half it does not own, or a
    caller gets an answer about a problem that was never taken here and an environment
    that was never released.
    """
    built = AiopsLabAdapter(AiopsLabConfig())
    context: Any = None

    for call in (
        lambda: built.acquire(),
        lambda: built.await_ready("detection"),
        lambda: built.submit("detection", "yes"),
        lambda: built.fetch_feedback(),
        lambda: built.release(state=None),
    ):
        with pytest.raises(LifecycleRunsElsewhere, match="aiopslab driver"):
            await call()

    for sync in (
        lambda: built.bind_session("s1"),
        lambda: built.build_signal(context),
        lambda: built.encode("detection", {}, context),
        lambda: built.holds_environment_for_grading(),
    ):
        with pytest.raises(LifecycleRunsElsewhere, match="aiopslab driver"):
            sync()


def test_a_bound_adapter_with_no_bridge_refuses_at_startup(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both ways the transport can be unavailable are startup failures naming the fix.

    The alternative is the prototype's: a process that starts, hands every role a
    toolset with no cluster access in it, and reports a model that could not
    investigate.
    """
    with pytest.raises(ValueError, match=r"adapters\.aiopslab\.enabled is false"):
        AiopsLabAdapter(AiopsLabConfig()).tool_transport()

    def missing(name: str) -> Any:
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr("kira.adapters.aiopslab.adapter.import_module", missing)
    enabled = AiopsLabAdapter(AiopsLabConfig(enabled=True, bridge_token=SecretStr(TOKEN)))
    with pytest.raises(ValueError, match="bridge transport is not available"):
        enabled.tool_transport()
