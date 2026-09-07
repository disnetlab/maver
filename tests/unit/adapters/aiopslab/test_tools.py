"""The two tools this environment adds, and the two ways each of them is held.

Everything a bridge-side check would also catch is checked here anyway, because the
agent-side half is the one that runs when the bridge is misconfigured or absent, and
because a refusal the model can read is cheaper than a round trip it has to interpret.
"""

import pathlib
from typing import Any

import pytest

from kira.adapters.aiopslab.config import AiopsLabConfig
from kira.adapters.aiopslab.tools import adapter_tools
from kira.audit.journal import Journal
from kira.settings.groups import ToolSettings
from kira.tools.context import ToolContext
from kira.tools.policy import ActionClass, policy_for
from kira.tools.registry import description_for
from kira.tools.workspace import SessionWorkspace

CONFIG = AiopsLabConfig()

TOOLS = {tool.spec.name: tool for tool in adapter_tools(CONFIG)}


class FakeBridge:
    """Records every action, and answers nothing a validation test could mistake."""

    name = "aiopslab"
    capabilities = frozenset({"kubernetes", "metrics", "traces"})
    timeout_seconds = 30

    def __init__(self) -> None:
        self.actions: list[tuple[str, str, dict[str, Any], str | None]] = []

    async def action(
        self,
        server: str,
        tool: str,
        arguments: Any = None,
        *,
        structured: bool = False,
        stage: str | None = None,
    ) -> str:
        self.actions.append((server, tool, dict(arguments or {}), stage))
        return "HTTP 200"


def context(role: str, tmp_path: pathlib.Path, bridge: FakeBridge) -> ToolContext:
    return ToolContext(
        transport=bridge,  # type: ignore[arg-type]
        workspace=SessionWorkspace(tmp_path),
        role=role,
        session_id="s1",
        journal=Journal(session_id="s1", stage=role),
    )


async def probe(role: str, tmp_path: pathlib.Path, bridge: FakeBridge, **kw: Any) -> str:
    return str(await TOOLS["probe_service"].spec.dispatch(context(role, tmp_path, bridge), **kw))


async def restore(role: str, tmp_path: pathlib.Path, bridge: FakeBridge, **kw: Any) -> str:
    return str(
        await TOOLS["restore_mongodb_access"].spec.dispatch(context(role, tmp_path, bridge), **kw)
    )


def test_each_tool_is_offered_to_exactly_one_role() -> None:
    """Verification is read-only end to end and remediation is the sole writer, so
    neither set has a second member and neither name appears in a core allowlist."""
    assert TOOLS["probe_service"].roles == frozenset({"verify"})
    assert TOOLS["restore_mongodb_access"].roles == frozenset({"execution"})


def test_the_gate_knows_the_recovery_writes_and_the_probe_does_not() -> None:
    """The gate reads the policy registry rather than the toolset, so a tool it cannot
    find is treated as a write by whichever stage calls it: that would put the read-only
    probe behind an approval it must never need."""
    probe_policy = policy_for("probe_service")
    recovery = policy_for("restore_mongodb_access")

    assert probe_policy is not None
    assert probe_policy.classify({}, ToolSettings()) == ActionClass.READ
    assert recovery is not None
    assert recovery.classify({}, ToolSettings()) == ActionClass.WRITE
    assert recovery.write_stages == frozenset({"execution"})
    assert recovery.stage_refusal("verify") == (
        "error: restore_mongodb_access is permitted only for the execution agent"
    )


@pytest.mark.parity
def test_the_probe_describes_itself_differently_to_the_role_that_holds_it() -> None:
    """An adapter's tool carries its own descriptions, since the shipped catalogue
    cannot name a tool that does not exist outside one environment."""
    spec = TOOLS["probe_service"].spec
    assert spec.descriptions is not None
    verify = description_for("probe_service", "verify", descriptions=spec.descriptions)
    default = description_for("probe_service", "rca", descriptions=spec.descriptions)

    assert verify.startswith("AIOpsLab Verify-only independent HTTP check.")
    assert default.startswith("AIOpsLab-only read-only HTTP probe.")


@pytest.mark.parity
async def test_the_probe_answers_only_to_verification(tmp_path: pathlib.Path) -> None:
    """The near half of the double check. The gate keys on the stage too, and a gate
    that was misconfigured or absent must not be enough to let another role open a
    request path of its choosing."""
    bridge = FakeBridge()

    refusal = await probe("execution", tmp_path, bridge, service="frontend", path="/")

    assert refusal == "error: probe_service is permitted only for the verify agent"
    assert bridge.actions == []


@pytest.mark.parity
@pytest.mark.parametrize(
    ("arguments", "refusal"),
    [
        ({"service": "Frontend", "path": "/"}, "error: service must be a DNS-1123 Service name"),
        ({"service": "front end", "path": "/"}, "error: service must be a DNS-1123 Service name"),
        (
            {"service": "frontend", "path": "health"},
            "error: path must be a bounded absolute HTTP path",
        ),
        (
            {"service": "frontend", "path": "//evil"},
            "error: path must be a bounded absolute HTTP path",
        ),
        (
            {"service": "frontend", "path": "/x?u=http://elsewhere"},
            "error: path must be a bounded absolute HTTP path",
        ),
        (
            {"service": "frontend", "path": "/x\r\nHost: y"},
            "error: path must be a bounded absolute HTTP path",
        ),
        (
            {"service": "frontend", "path": "/" + "a" * 300},
            "error: path must be a bounded absolute HTTP path",
        ),
        (
            {"service": "frontend", "path": "/", "port": 0},
            "error: port must be an integer in 1..65535",
        ),
        (
            {"service": "frontend", "path": "/", "port": 70000},
            "error: port must be an integer in 1..65535",
        ),
        # `True` is an int in Python, and a bridge asked to forward to port 1 because
        # the model wrote `true` is a destination nobody chose.
        (
            {"service": "frontend", "path": "/", "port": True},
            "error: port must be an integer in 1..65535",
        ),
    ],
)
async def test_the_probe_refuses_an_argument_it_cannot_bound(
    tmp_path: pathlib.Path, arguments: dict[str, Any], refusal: str
) -> None:
    bridge = FakeBridge()

    assert await probe("verify", tmp_path, bridge, **arguments) == refusal
    assert bridge.actions == []


async def test_a_bounded_probe_reaches_the_bridge_as_a_verify_action(
    tmp_path: pathlib.Path,
) -> None:
    """The port is omitted rather than defaulted, because choosing one here would take
    the destination away from the bridge, which is the half that knows what the Service
    declares."""
    bridge = FakeBridge()

    assert await probe("verify", tmp_path, bridge, service="frontend", path="/") == "HTTP 200"
    await probe("verify", tmp_path, bridge, service="frontend", path="/health", port=5000)

    assert bridge.actions == [
        ("aiopslab", "probe_service", {"service": "frontend", "path": "/"}, "verify"),
        (
            "aiopslab",
            "probe_service",
            {"service": "frontend", "path": "/health", "port": 5000},
            "verify",
        ),
    ]


@pytest.mark.parity
async def test_the_recovery_answers_only_to_remediation(tmp_path: pathlib.Path) -> None:
    bridge = FakeBridge()

    refusal = await restore("verify", tmp_path, bridge, service="mongodb-geo", access="user")

    assert refusal == "error: restore_mongodb_access is permitted only for the execution agent"
    assert bridge.actions == []


@pytest.mark.parity
@pytest.mark.parametrize(
    "arguments",
    [
        {"service": "mongodb-rate", "access": "password"},
        {"service": "mongodb-user", "access": "privileges"},
        {"service": "", "access": ""},
    ],
)
async def test_the_recovery_refuses_a_pair_the_bridge_has_no_script_for(
    tmp_path: pathlib.Path, arguments: dict[str, str]
) -> None:
    """Four shipped scripts and no fifth. The alternative to a closed argument set is a
    generic in-container command, which is the thing the bridge exists to refuse."""
    bridge = FakeBridge()

    refusal = await restore("execution", tmp_path, bridge, **arguments)

    assert refusal == "error: unsupported MongoDB recovery selection"
    assert bridge.actions == []


async def test_an_allowed_repair_reaches_the_bridge_as_an_execution_action(
    tmp_path: pathlib.Path,
) -> None:
    bridge = FakeBridge()

    await restore("execution", tmp_path, bridge, service="mongodb-rate", access="privileges")

    assert bridge.actions == [
        (
            "aiopslab",
            "restore_mongodb_access",
            {"service": "mongodb-rate", "access": "privileges"},
            "execution",
        )
    ]


def test_the_repair_schema_offers_the_configured_pairs_and_nothing_else() -> None:
    """Data, not a literal: the enum is what the deployment configured, so a bridge
    that shipped a fifth script is a configuration change rather than a code change."""
    parameters = TOOLS["restore_mongodb_access"].spec.parameters
    properties = parameters["properties"]

    assert properties["service"]["enum"] == ["mongodb-geo", "mongodb-rate"]
    assert properties["access"]["enum"] == ["privileges", "user"]
    assert parameters["required"] == ["service", "access"]

    narrowed = adapter_tools(AiopsLabConfig(mongodb_services=frozenset({"mongodb-geo"})))
    schema = next(tool.spec.parameters for tool in narrowed if tool.spec.name.startswith("restore"))
    assert schema["properties"]["service"]["enum"] == ["mongodb-geo"]
