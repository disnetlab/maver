import ast
import os
from dataclasses import replace
from pathlib import Path

import pytest

from kira.audit.journal import Journal
from kira.settings.groups import ToolSettings
from kira.tools.allowlists import ROLE_TOOLSETS
from kira.tools.context import ToolContext
from kira.tools.toolsets import build_toolset
from kira.tools.workspace import SessionWorkspace


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


ALL = frozenset(
    {"kubernetes", "kubernetes.write", "kubernetes.rollback", "metrics", "logs", "traces"}
)


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

PROTOTYPE_REGISTRY = Path(
    os.environ.get(
        "PROTOTYPE_REGISTRY",
        str(
            Path(__file__).resolve().parents[2].parent
            / "k8s-agentic-operator"
            / "agent"
            / "tools"
            / "registry.py"
        ),
    )
)

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
@pytest.mark.skipif(not PROTOTYPE_REGISTRY.exists(), reason="the prototype tree is not beside us")
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


def test_node_recovery_is_withheld_when_there_is_no_node_to_reach(tmp_path: Path) -> None:
    """Every other tool that cannot work is dropped before the run, with the reason
    recorded, because a tool that errors reads to a model as a fault to work around.
    Node recovery was the exception: its host allowlist is empty by default and was empty
    across every measured campaign, so the remediating role was offered a tool that
    refused every call, and a fault repairable only on the node was recorded as one the
    agent failed to fix.
    """
    withheld = build_toolset("execution", context(tmp_path))
    assert "ssh_exec" not in {tool.name for tool in withheld.definitions}
    assert withheld.dropped["ssh_exec"] == "no host-recovery node is configured"

    reachable = context(tmp_path)
    reachable = replace(
        reachable, settings=ToolSettings(host_recovery_nodes=frozenset({"worker-1"}))
    )
    offered = build_toolset("execution", reachable)
    assert "ssh_exec" in {tool.name for tool in offered.definitions}
    assert "ssh_exec" not in offered.dropped
