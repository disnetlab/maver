import pytest

from kira.audit.journal import Journal
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
    {"kubernetes", "kubernetes.write", "kubernetes.rollback", "metrics", "logs", "traces", "host"}
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
