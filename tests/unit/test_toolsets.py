import ast
from dataclasses import replace
from pathlib import Path

import pytest
from prototype_tree import prototype_root

from maver.audit.journal import Journal
from maver.settings.groups import ToolSettings
from maver.tools import sandbox, toolsets
from maver.tools.allowlists import ROLE_TOOLSETS
from maver.tools.context import ToolContext
from maver.tools.toolsets import build_toolset
from maver.tools.workspace import SessionWorkspace


class FakeTransport:
    def __init__(self, capabilities: frozenset[str]) -> None:
        self._capabilities = capabilities

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    @property
    def timeout_seconds(self) -> int:
        return 60


ALL = frozenset({"kubernetes", "kubernetes.rollback", "metrics", "logs", "traces"})


def context(tmp_path, capabilities=ALL, memory=None) -> ToolContext:
    return ToolContext(
        transport=FakeTransport(capabilities),
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id="s1",
        journal=Journal(session_id="s1", stage="rca"),
        memory=memory,
    )


def test_a_capability_the_transport_lacks_removes_its_tools_before_the_run(tmp_path) -> None:
    """A tool that errors reads to the model as a fault to work around, and it retries.

    A tool that was never offered costs nothing, and the reason is recorded so a later
    reader can tell "the model did not look at logs" from "there were no logs to look at".
    """
    resolved = build_toolset("rca", context(tmp_path, capabilities=ALL - {"logs"}))
    offered = {d.name for d in resolved.definitions}

    assert "get_logs" not in offered
    assert resolved.dropped["get_logs"] == "transport lacks logs"
    assert "get_metrics" in offered


@pytest.mark.parametrize("role", sorted(ROLE_TOOLSETS))
def test_the_word_kubernetes_alone_buys_kubectl_for_every_role(tmp_path, role: str) -> None:
    """One tool serves cluster reads and cluster writes, so it needs one capability.

    A second word for writing was briefly required alongside the first, which withheld
    kubectl entirely from a transport advertising `kubernetes` alone — including from the
    diagnostic roles, which hold the tool only to read, and which then lost every cluster
    read silently while the drop record said the transport could not be written to. Who
    may write through the tool is the approval gate's decision and the tool's own stage
    check, not a capability's.
    """
    resolved = build_toolset(role, context(tmp_path, capabilities=frozenset({"kubernetes"})))

    assert "exec_kubectl_cmd_safely" in {d.name for d in resolved.definitions}
    assert "exec_kubectl_cmd_safely" not in resolved.dropped


def test_memory_tools_go_when_the_configuration_has_no_store(tmp_path) -> None:
    resolved = build_toolset("rca", context(tmp_path, memory=None))
    assert "memory_search" not in {d.name for d in resolved.definitions}
    assert resolved.dropped["memory_search"] == "no experience store in this configuration"


def test_a_submit_tool_is_never_dropped(tmp_path) -> None:
    """The loop ends only by calling it. Without one a run cannot terminate at all —
    it burns its whole budget and reports a timeout that looks like the model's fault."""
    resolved = build_toolset("rca", context(tmp_path, capabilities=frozenset()))
    assert "submit_root_cause" in {d.name for d in resolved.definitions}
    assert "submit_root_cause" not in resolved.dropped


def test_the_declared_order_is_preserved(tmp_path) -> None:
    """It is the order the model reads the tools in, and it was hand-tuned."""
    resolved = build_toolset("rca", context(tmp_path, memory=object()))
    assert [d.name for d in resolved.definitions] == list(ROLE_TOOLSETS["rca"])


def test_every_offered_tool_carries_a_description_and_a_rationale(tmp_path) -> None:
    resolved = build_toolset("execution", context(tmp_path))
    for definition in resolved.definitions:
        assert definition.description.strip()
        properties = definition.parameters.get("properties") or {}
        if definition.name.startswith("submit_"):
            assert "rationale" not in properties
        else:
            assert next(iter(properties)) == "rationale"


def test_an_unknown_role_is_an_error_not_an_empty_toolset(tmp_path) -> None:
    """An empty toolset would produce a run that silently cannot act."""
    with pytest.raises(Exception, match="no toolset defined"):
        build_toolset("not_a_role", context(tmp_path))


# --- the allowlists against the prototype that measured them -----------------------

PROTOTYPE_REGISTRY = prototype_root() / "agent" / "tools" / "registry.py"

DROPPED: dict[str, frozenset[str]] = {
    "rca": frozenset({"memory_grep"}),
    "planning": frozenset({"memory_grep"}),
    "execution": frozenset(
        {"restore_mongodb_access", "rollback_command", "get_previous_rollbackable_cmd"}
    ),
    "verify": frozenset({"probe_service"}),
    "learning": frozenset(
        {
            "memory_grep",
            "playbook_read",
            "playbook_append",
            "playbook_reconfirm",
            "playbook_remove",
        }
    ),
    "single": frozenset({"rollback_command", "get_previous_rollbackable_cmd"}),
}
"""What each role no longer offers, and why it is a drop rather than a regression.

``memory_grep`` and the four ``playbook_*`` tools are mechanisms this rebuild does not
ship. The rest are the benchmark's own tools, which now reach a role as adapter extras
instead of sitting in a core allowlist — the same tools, contributed by whatever knows
the environment rather than named in code that must not know it.
"""


def _prototype_toolsets() -> dict[str, tuple[str, ...]]:
    """Read ``AGENT_TOOLSETS`` out of the prototype without importing it."""
    tree = ast.parse(PROTOTYPE_REGISTRY.read_text(encoding="utf-8"))
    for node in tree.body:
        target = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and getattr(target, "id", "") == "AGENT_TOOLSETS":
            return dict(ast.literal_eval(node.value))
    raise AssertionError("AGENT_TOOLSETS not found in the prototype registry")


@pytest.mark.parity
@pytest.mark.parametrize("role", sorted(ROLE_TOOLSETS))
def test_a_role_offers_what_the_prototype_offered_in_the_order_it_offered_it(role: str) -> None:
    """The existing check compares the resolved order against the table it came from.

    That is a tautology, and it leaves the number that matters unpinned: which tools a
    role was offered, and in what order, is an input to every published measurement, and
    the allowlist module says plainly that the order was hand-tuned and that every
    measurement used it. Adding or reordering one tool changes what the model saw with no
    error and no visible symptom.

    So the comparison is against the prototype that produced those numbers, with the
    drops named explicitly. A tool that leaves an allowlist has to be added to
    ``DROPPED`` deliberately, which is the point.
    """
    prototype = _prototype_toolsets()[role]
    assert tuple(t for t in prototype if t not in DROPPED[role]) == tuple(ROLE_TOOLSETS[role])


def test_node_recovery_is_offered_with_no_node_to_reach(tmp_path: Path) -> None:
    """The one tool exempt from the drop-before-the-run rule, and deliberately so.

    Its host allowlist is empty by default and was empty across every measured campaign,
    and the remediating role was offered it in all of them: the definition sat in the
    execution toolset and its description sat in the prompt. Withholding it on an empty
    allowlist reads better — a tool that refuses every call reads to a model as a fault
    to work around — and it changes what the model was shown, which is an input to every
    published number. So it is offered either way and the tool refuses the call.
    """
    empty = build_toolset("execution", context(tmp_path))
    assert "ssh_exec" in {tool.name for tool in empty.definitions}
    assert "ssh_exec" not in empty.dropped

    reachable = context(tmp_path)
    reachable = replace(
        reachable, settings=ToolSettings(host_recovery_nodes=frozenset({"worker-1"}))
    )
    offered = build_toolset("execution", reachable)
    assert "ssh_exec" in {tool.name for tool in offered.definitions}
    assert "ssh_exec" not in offered.dropped


def test_the_sandbox_tools_go_when_the_kernel_has_no_sandbox_to_put_them_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place the drop-before-the-run rule was not applied.

    The sandbox is fail-closed on Linux, so on a kernel with no usable Landlock — the
    documented amd64 image under emulation is one, and so is any distribution built
    without the LSM — bash_exec and code_exec returned `error:` on every call for a whole
    incident. In the results that is indistinguishable from a model that could not use
    them, and the escape hatch appeared in no message the operator saw.
    """
    monkeypatch.setattr(toolsets, "_sandboxed", lambda: False)
    withheld = build_toolset("rca", context(tmp_path))

    assert {"bash_exec", "code_exec"}.isdisjoint({tool.name for tool in withheld.definitions})
    assert withheld.dropped["bash_exec"] == "no filesystem sandbox on this kernel"
    assert withheld.dropped["code_exec"] == "no filesystem sandbox on this kernel"

    # Unsandboxed by configuration is a deployment's choice, not a broken kernel.
    unsandboxed = replace(context(tmp_path), settings=ToolSettings(fs_sandbox=False))
    assert "bash_exec" in {tool.name for tool in build_toolset("rca", unsandboxed).definitions}


def test_the_sandbox_launcher_refuses_with_the_code_and_prefix_the_tool_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """126 and the prefix are the contract between the launcher and the tool that reads
    it: `_run` turns exactly that pair into a tool error naming the sandbox rather than
    into the command's own failed exit code, which is what tells the two apart."""

    def no_landlock(session_dir: object) -> bool:
        raise RuntimeError("Linux kernel does not provide usable Landlock")

    monkeypatch.setattr(sandbox, "enter_session_sandbox", no_landlock)

    assert sandbox._main(["sandbox.py", str(tmp_path), "--", "/bin/sh", "-c", "true"]) == 126
    assert capsys.readouterr().out.startswith(sandbox.SANDBOX_ERROR_PREFIX)

    assert sandbox._main(["sandbox.py", str(tmp_path)]) == 126
    assert capsys.readouterr().out.startswith(sandbox.SANDBOX_ERROR_PREFIX)
