"""The kubectl route a deployment runs on, and everything that has to agree about it.

Two routes exist. On the cluster-tool route the sandbox holds no credentials, refuses
kubectl, and names the cluster tool in the refusal. On the proxy route the bound
transport offers a credential-free endpoint, a stage-aware ``kubectl`` shim sits first on
the sandbox's PATH, and the model is told to use it.

Three independent places read the route — the sandbox tools, the description catalogue
and the prompt loader — and any two of them disagreeing is silent at incident time: a
model told the sandbox has no credentials while its kubectl works spends turns on
refusals it can disprove, and one told the opposite spends them on kubectl commands that
cannot run. So they are checked together, against one fake transport.
"""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from kira.audit.journal import Journal
from kira.settings.groups import ToolSettings
from kira.tools.context import EvidencePolicy, ToolContext, kubectl_route
from kira.tools.execution import bash_exec, code_exec, subprocess_env
from kira.tools.workspace import SessionWorkspace


class FakeTransport:
    """A transport that may or may not offer the proxy route."""

    def __init__(self, proxy_url: str | None = None) -> None:
        self._proxy_url = proxy_url

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"kubernetes"})

    @property
    def timeout_seconds(self) -> int:
        return 60

    @property
    def kubectl_proxy_url(self) -> str | None:
        return self._proxy_url


PROXY = "http://127.0.0.1:16444"


@pytest.fixture
def shim(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stand in for ``kira.tools.kubectl_proxy``, recording what it was handed.

    The module under test reaches the real one by name at call time, so a stand-in in
    ``sys.modules`` is what a caller sees. It is a test double rather than a second
    implementation: what is asserted here is the arguments the sandbox tools compute —
    the stage, the deployment's approval mode, the configured kubectl and the timeout —
    which is this module's half of the contract.

    Returns:
        The recorded calls, in order.
    """
    calls: list[dict[str, Any]] = []
    module = types.ModuleType("kira.tools.kubectl_proxy")

    def apply_proxy_env(env: dict[str, str], **kwargs: Any) -> dict[str, str]:
        calls.append({"env": dict(env), **kwargs})
        return {**env, "KUBECTL_MODE": "proxy_exec", "AGENT_STAGE": str(kwargs["stage"])}

    module.apply_proxy_env = apply_proxy_env  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kira.tools.kubectl_proxy", module)
    return calls


def context(
    tmp_path: Path,
    *,
    proxy_url: str | None = PROXY,
    role: str = "execution",
    authority_mode: str = "auto",
    kubectl_binary: str = "/usr/local/bin/kubectl",
    evidence: EvidencePolicy | None = None,
) -> ToolContext:
    """A tool context on either route, with the sandbox wrapper off.

    The filesystem sandbox is off because these tests run real subprocesses and the
    wrapper is a Linux-only confinement this has nothing to say about.
    """
    return ToolContext(
        transport=FakeTransport(proxy_url),  # type: ignore[arg-type]
        workspace=SessionWorkspace(tmp_path),
        role=role,
        session_id="s1",
        journal=Journal(session_id="s1", stage=role),
        settings=ToolSettings(fs_sandbox=False, kubectl_timeout_s=137),
        authority_mode=authority_mode,
        kubectl_binary=kubectl_binary,
        **({"evidence": evidence} if evidence is not None else {}),
    )


# --- which route a transport puts a run on -----------------------------------------


def test_a_transport_with_no_proxy_is_on_the_cluster_tool_route() -> None:
    assert kubectl_route(FakeTransport()) == "tool"  # type: ignore[arg-type]
    assert kubectl_route(FakeTransport(PROXY)) == "proxy_exec"  # type: ignore[arg-type]


def test_a_transport_that_has_never_heard_of_the_property_is_on_the_tool_route() -> None:
    """Transports arrive through an entry point, so one built elsewhere carries no such
    property. Reading it as "no proxy" leaves that deployment behaving as it does today,
    rather than raising in the middle of the first shell command."""

    class Foreign:
        name = "foreign"
        capabilities = frozenset({"kubernetes"})
        timeout_seconds = 60

    assert kubectl_route(Foreign()) == "tool"  # type: ignore[arg-type]


# --- the sandbox tools --------------------------------------------------------------


async def test_the_shell_refuses_kubectl_on_the_tool_route(tmp_path: Path) -> None:
    answer = await bash_exec(context(tmp_path, proxy_url=None), cmd="kubectl get pods -n shop")

    assert answer.startswith("error: do not run kubectl through bash_exec")


async def test_the_shell_runs_kubectl_on_the_proxy_route(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """PATH resolves kubectl to the stage-aware shim there, and the model has been told
    so. A refusal it can disprove in one command costs turns and teaches it to distrust
    the harness rather than the instruction."""
    answer = await bash_exec(context(tmp_path), cmd="kubectl get pods -n shop")

    assert "do not run kubectl" not in answer
    assert len(shim) == 1


async def test_a_script_may_shell_out_to_kubectl_on_the_proxy_route(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    code = "import subprocess\nprint(subprocess.run(['kubectl', 'version']))"

    refused = await code_exec(context(tmp_path, proxy_url=None), code=code, language="python")
    allowed = await code_exec(context(tmp_path), code=code, language="python")

    assert refused.startswith("error: do not run kubectl from code_exec")
    assert "do not run kubectl" not in allowed


async def test_the_proxy_wiring_reaches_the_subprocess_itself(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """Not just the computed dictionary: the environment the command actually runs in.

    The prototype applied this inside the one function that built the environment, so
    every sandbox call inherited it. Computing it and not passing it would leave the
    model with a PATH that has no shim on it and a prompt telling it to use one.
    """
    answer = await bash_exec(context(tmp_path), cmd="echo route=$KUBECTL_MODE/$AGENT_STAGE")

    assert "route=proxy_exec/execution" in answer


@pytest.mark.parametrize("role", ["execution", "single.mitigation"])
async def test_the_stage_handed_to_the_shim_is_the_one_the_gate_keys_on(
    tmp_path: Path, shim: list[dict[str, Any]], role: str
) -> None:
    """The write policy names ``execution`` and ``single.mitigation``, and the shim
    applies the same two. Handing it anything else — the role behind a phased stage, say
    — refuses the single agent's whole remediation pass while the gate would have allowed
    every command in it."""
    await bash_exec(context(tmp_path, role=role), cmd="kubectl get pods -n shop")

    assert shim[0]["stage"] == role


async def test_the_deployments_own_settings_reach_the_shim(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """Three values with an honest source each, and no second copy of any of them: the
    approval gate's mode, the configured kubectl, and the cluster-command ceiling."""
    await bash_exec(
        context(tmp_path, authority_mode="prompt", kubectl_binary="/opt/bin/kubectl"),
        cmd="kubectl get pods -n shop",
    )

    assert shim[0]["authority_mode"] == "prompt"
    assert shim[0]["kubectl_binary"] == "/opt/bin/kubectl"
    assert shim[0]["exec_timeout_s"] == 137.0
    assert shim[0]["proxy_url"] == PROXY
    assert shim[0]["session_dir"] == tmp_path


async def test_the_tool_route_builds_the_environment_without_reaching_the_shim(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """A deployment with no proxy must not acquire proxy wiring, or a direct-transport
    run would start refusing writes on an authority mode it never consulted."""
    env = subprocess_env(context(tmp_path, proxy_url=None))

    assert "KUBECTL_MODE" not in env
    assert shim == []


async def test_ssh_is_still_refused_on_the_proxy_route(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """The proxy carries kubectl and nothing else. Host mutations still belong to the
    tool that is stage-scoped and gated, so the two ssh guards are untouched by it."""
    ctx = context(tmp_path)

    assert (await bash_exec(ctx, cmd="ssh worker1 systemctl restart kubelet")).startswith(
        "error: do not run ssh from bash_exec"
    )
    assert (await bash_exec(ctx, cmd="ssh_exec worker1 uptime")).startswith(
        "error: ssh_exec is a function tool"
    )
    assert (
        await code_exec(
            ctx, code="import subprocess\nsubprocess.run(['ssh', 'w1'])", language="python"
        )
    ).startswith("error: do not run ssh from code_exec")


async def test_output_still_goes_through_the_evidence_policy_on_the_proxy_route(
    tmp_path: Path, shim: list[dict[str, Any]]
) -> None:
    """The archived deployment filtered the sandboxes' output only on this route. Here
    the filter is applied on both, so what the route must not do is bypass it: a command
    that now reaches the cluster is exactly the one that can print what the environment
    is hiding."""

    class HideTheInjector:
        def filter_result(self, content: str) -> str:
            return content.replace("chaos-mesh", "[redacted]")

        def filter_query(self, query: str) -> str:
            return query

    answer = await bash_exec(
        context(tmp_path, evidence=HideTheInjector()), cmd="echo kubectl saw chaos-mesh"
    )

    assert "chaos-mesh" not in answer
    assert "[redacted]" in answer
