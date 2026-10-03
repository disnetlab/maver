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

import ast
from pathlib import Path
from typing import Any

import pytest
import yaml

from maver.audit.journal import Journal
from maver.settings.groups import ToolSettings
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.context import EvidencePolicy, ToolContext, kubectl_route
from maver.tools.execution import bash_exec, code_exec, subprocess_env
from maver.tools.registry import ToolRegistryError, description_for, validate_registry
from maver.tools.toolsets import build_toolset
from maver.tools.workspace import SessionWorkspace

ALL_CAPABILITIES = frozenset({"kubernetes", "kubernetes.rollback", "metrics", "logs", "traces"})


class FakeTransport:
    """A transport that may or may not offer the proxy route."""

    def __init__(
        self, proxy_url: str | None = None, capabilities: frozenset[str] | None = None
    ) -> None:
        self._proxy_url = proxy_url
        self._capabilities = capabilities or frozenset({"kubernetes"})

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    @property
    def timeout_seconds(self) -> int:
        return 60

    @property
    def kubectl_proxy_url(self) -> str | None:
        return self._proxy_url


PROXY = "http://127.0.0.1:16444"


def context(
    tmp_path: Path,
    *,
    proxy_url: str | None = PROXY,
    role: str = "execution",
    authority_mode: str = "auto",
    kubectl_binary: str = "/usr/local/bin/kubectl",
    evidence: EvidencePolicy | None = None,
    capabilities: frozenset[str] | None = None,
) -> ToolContext:
    """A tool context on either route, with the sandbox wrapper off.

    The filesystem sandbox is off because these tests run real subprocesses and the
    wrapper is a Linux-only confinement this has nothing to say about.
    """
    return ToolContext(
        transport=FakeTransport(proxy_url, capabilities),  # type: ignore[arg-type]
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
    assert shim[0]["kubectl_exec_auto_approved"] is False
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


# --- the role prompts ---------------------------------------------------------------


ROLES = Path(__file__).resolve().parents[2] / "src" / "maver" / "roles"


def test_every_role_composes_its_prompt_for_the_route_it_is_running_on() -> None:
    """Six roles compose a prompt and each does it in its own module.

    Five getting it right is not most of the way there: it is one role holding the
    proxy's kubectl and reading guidance that says the sandbox has no credentials,
    inside a pipeline where the other five say the opposite. The call sites are checked
    rather than the composition, because a missing keyword takes the parameter default
    and composes a valid prompt for the wrong route — there is nothing to raise.
    """
    unrouted = []
    for path in sorted(ROLES.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "compose_role_prompt":
                continue
            if not any(keyword.arg == "kubectl_route" for keyword in node.keywords):
                unrouted.append(f"{path.name}:{node.lineno}")

    assert not unrouted, f"these compose a role prompt without a route: {unrouted}"


# --- the description catalogue ------------------------------------------------------


ARCHIVED = yaml.safe_load(
    """
bash_exec:
  default: no credentials here
  agents:
    execution: no credentials here either
  proxy_exec:
    default: kubectl resolves to the shim
    agents:
      execution: kubectl resolves to the shim, and you may write
"""
)


def test_a_route_replaces_the_whole_entry_rather_than_blending_with_it() -> None:
    """Blended, the two would describe a tool that both refuses kubectl and resolves it.

    The per-role rule is the same one a level down, so a route's text has to be complete
    on its own — and a role the route says nothing about reads that route's default, not
    the other route's role text.
    """
    assert description_for("bash_exec", "rca", descriptions=ARCHIVED) == "no credentials here"
    assert (
        description_for("bash_exec", "rca", descriptions=ARCHIVED, route="proxy_exec")
        == "kubectl resolves to the shim"
    )
    assert (
        description_for("bash_exec", "execution", descriptions=ARCHIVED, route="proxy_exec")
        == "kubectl resolves to the shim, and you may write"
    )


def test_a_tool_that_says_nothing_about_a_route_reads_the_same_on_all_of_them() -> None:
    entry = {"get_logs": {"default": "read logs"}}

    assert description_for("get_logs", "rca", descriptions=entry, route="proxy_exec") == "read logs"


def test_a_route_entry_without_a_default_is_refused_at_startup() -> None:
    """It would serve the roles it does not name the *other* route's text, which is the
    contradiction the whole route axis exists to avoid — silently, and only for some
    roles."""
    catalogue = {"bash_exec": {"default": "d", "proxy_exec": {"agents": {"execution": "x"}}}}
    with pytest.raises(ToolRegistryError, match="has no default"):
        validate_registry(
            descriptions=catalogue,
            allowlists={"execution": ("bash_exec",)},
            registry={"bash_exec": _stand_in()},
        )


def test_a_route_entry_may_not_describe_a_role_that_cannot_call_the_tool() -> None:
    """The same rule as the entry above it: a description for a role that is not
    allowlisted means the two hand-edited files disagree while each looks right."""
    catalogue = {
        "bash_exec": {"default": "d", "proxy_exec": {"default": "p", "agents": {"rca": "r"}}}
    }
    with pytest.raises(ToolRegistryError, match="not allowed to call it"):
        validate_registry(
            descriptions=catalogue,
            allowlists={"execution": ("bash_exec",)},
            registry={"bash_exec": _stand_in()},
        )


def _stand_in():
    from maver.tools.registry import ToolSpec  # noqa: PLC0415

    async def fn(ctx: object, **_: object) -> str:
        return "ok"

    return ToolSpec(
        name="bash_exec",
        parameters={"type": "object", "properties": {}},
        safety="read_only",
        dispatch=fn,
    )


SANDBOX_ROLES = sorted(role for role, names in ROLE_TOOLSETS.items() if "bash_exec" in names)


@pytest.mark.parametrize("role", SANDBOX_ROLES)
def test_the_toolset_describes_the_sandbox_tools_for_the_route_it_resolved(
    tmp_path: Path, role: str
) -> None:
    """The description and the guidance block are the same claim in two places. Serving
    the default text on the proxy route tells the model kubectl is rejected in the very
    request whose system prompt tells it to run kubectl."""
    proxied = build_toolset(role, context(tmp_path, role=role))
    unproxied = build_toolset(role, context(tmp_path, role=role, proxy_url=None))

    described = {d.name: d.description for d in proxied.definitions}
    plain = {d.name: d.description for d in unproxied.definitions}
    for name in ("bash_exec", "code_exec"):
        assert "KUBECTL_MODE=proxy_exec" in described[name]
        assert "KUBECTL_MODE" not in plain[name]


async def test_a_killed_sandbox_command_is_reported_in_the_words_the_archive_used(
    tmp_path: Path,
) -> None:
    """A rebuild is validated by comparing the `error:` strings the model saw against the
    archive's, where a string only MAVER emits reads as a defect. The prototype called
    these `command` and `script`; the saved-output labels stay `shell` and `code_exec`,
    which is also what it used."""
    ctx = context(tmp_path)
    object.__setattr__(ctx.settings, "shell_exec_timeout_s", 0)
    object.__setattr__(ctx.settings, "code_exec_timeout_s", 0)

    shell = await bash_exec(ctx, cmd="sleep 30")
    code = await code_exec(ctx, code="import time; time.sleep(30)", language="python")

    assert shell.startswith("error: command timed out after")
    assert code.startswith("error: script timed out after")
