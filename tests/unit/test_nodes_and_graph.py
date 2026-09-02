import ast
import pathlib

import pytest

from kira.audit.journal import Journal, ListSink
from kira.contracts.artifacts import RootCause, VerifyResult
from kira.contracts.events import StageEvent
from kira.contracts.state import IncidentState
from kira.orchestration.nodes import IncidentDeps, make_node
from kira.orchestration.routing import RoutingPolicy
from kira.roles.base import RoleOutcome, RoleSpec
from kira.tools.toolsets import ResolvedToolset

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "kira"
POLICY = RoutingPolicy(verify_enabled=True, remediation_retries=2, inconclusive_retries=1)


class Recorder:
    def __init__(self) -> None:
        self.started: list[tuple[str, str | None]] = []
        self.finished: list[dict] = []

    def start(self, *, session_id: str, role: str, after: str | None) -> str:
        self.started.append((role, after))
        return f"run-{len(self.started)}"

    def finish(self, run_id, *, status, output, trace_summary, error, metadata) -> None:
        self.finished.append({"run_id": run_id, "status": status, "error": error})


def deps(recorder: Recorder, sink: ListSink) -> IncidentDeps:
    def build_context(spec, state, handback, run_id):  # type: ignore[no-untyped-def]
        return _Captured(
            spec=spec,
            state=state,
            handback=handback,
            run_id=run_id,
            toolset=ResolvedToolset(definitions=(), dispatch={}),
            run_label="",
        )

    return IncidentDeps(
        policy=POLICY,
        runs=recorder,
        journal=Journal(sink, session_id="s1", stage="pipeline"),
        build_context=build_context,  # type: ignore[arg-type]
    )


class _Captured:
    """Stands in for a RoleContext.

    It carries a toolset and a run label because the wrapper reads both: what a run
    withheld from the model, and which of a mechanism's several runs of one role this
    was, go on the run row and are recorded nowhere else.
    """

    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


def spec_for(name: str, *, outcome: RoleOutcome, **kw) -> RoleSpec:
    async def run(ctx) -> RoleOutcome:  # type: ignore[no-untyped-def]
        run.seen = ctx  # type: ignore[attr-defined]
        return outcome

    return RoleSpec(
        name=name,
        submit_tool=f"submit_{name}",
        submit_model=RootCause,
        state_slot=kw.pop("state_slot", None),
        prompt=name,
        run=run,
        **kw,
    )


def root_cause() -> RootCause:
    return RootCause(
        fault_type="misconfig",
        affected_resource="deploy/web",
        confidence=0.8,
        evidence=["probe fails"],
        summary="the readiness probe points at the wrong port",
    )


def verify_result(**kw) -> VerifyResult:
    kw.setdefault("checks", ["curl http://checkout/health returned 200"])
    kw.setdefault("confidence", 0.7)
    return VerifyResult(**kw)


# --- the LangGraph containment rule, made checkable --------------------------------

FORBIDDEN = ("checkpoint", "MemorySaver", "interrupt", "create_react_agent", "prebuilt")


def test_only_one_module_imports_the_graph_executor() -> None:
    """Persistence, resume, approval and lifecycle are ours.

    They are load-bearing for what the system claims about auditability, so they must not
    depend on a framework's roadmap. The target is that if this dependency disappeared,
    replacing it would take days.
    """
    importers = [
        path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py") if _imports_langgraph(path)
    ]
    assert importers == ["orchestration/graph.py"]


def _imports_langgraph(path: pathlib.Path) -> bool:
    """Say whether a module imports the executor.

    Imports, not text. The engine setting names its two values as strings and a
    docstring may explain the rule it is subject to; neither is a dependency, and a
    substring scan cannot tell any of them apart.
    """
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "langgraph" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "langgraph":
            return True
    return False


def test_the_permitted_surface_is_four_operations() -> None:
    """The import exemption that makes graph.py possible would otherwise also let the
    forbidden surface in there, so the one exempt file is checked by name.

    Names actually used, not text: this file's own prose names the forbidden surface in
    order to explain why it is forbidden, and a substring scan cannot tell the rule from
    a description of it.
    """
    tree = ast.parse((SRC / "orchestration" / "graph.py").read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            used.update(alias.name for alias in node.names)
            used.update((node.module or "").split("."))
        elif isinstance(node, ast.Import):
            used.update(part for alias in node.names for part in alias.name.split("."))
    assert sorted(name for name in FORBIDDEN if name in used) == []


def test_the_longest_legal_path_fits_the_executor_default() -> None:
    """Fourteen steps under a budget of two with one re-check per cycle, against a
    default of twenty-five. Asserted here rather than trusted, because the default is
    the framework's to change."""
    cycle = ["planning", "execution", "verify", "verify"]
    longest = ["rca", *cycle, *cycle, "rca", *cycle]
    assert len(longest) == 14
    assert len(longest) < 25


# --- what the wrapper does so roles do not ----------------------------------------


async def test_only_a_role_that_declares_it_sets_the_incident_error() -> None:
    """An asymmetry the prototype had in code and nowhere in prose.

    A check that could not run must not invalidate a remediation that applied cleanly,
    so verification alone returns no result and lets the incident be decided on the
    remediation's own claim.
    """
    sink, recorder = ListSink(), Recorder()
    failed = RoleOutcome(artifact=None, transcript=[], error="max_turns_exceeded")

    diagnosing = make_node(
        spec_for("rca", outcome=failed, state_slot="root_cause"), deps(recorder, sink)
    )
    delta = await diagnosing({"session_id": "s1"})
    assert "error" in delta

    checking = make_node(
        spec_for("verify", outcome=failed, state_slot="verify_result", failure_sets_error=False),
        deps(recorder, sink),
    )
    assert "error" not in await checking({"session_id": "s1"})


async def test_a_hand_back_is_charged_before_the_role_runs() -> None:
    """A role that then fails still consumes its retry, which is the correct accounting:
    an attempt was made. The prototype could skip the increment by failing first."""
    sink, recorder = ListSink(), Recorder()
    node = make_node(
        spec_for(
            "rca",
            outcome=RoleOutcome(artifact=root_cause(), transcript=[]),
            state_slot="root_cause",
        ),
        deps(recorder, sink),
    )
    state: IncidentState = {
        "session_id": "s1",
        "root_cause": root_cause(),
        "verify_result": verify_result(
            recovered=False,
            verdict="not_recovered",
            summary="still failing",
            needs_reassessment=True,
            reassessment_reason="the named component was not the origin",
        ),
    }
    delta = await node(state)
    assert delta["remediation_retries"] == 1
    assert delta["reassessment_count"] == 1


async def test_verification_is_charged_for_a_second_look_not_a_first() -> None:
    """The prototype incremented on every pass, so after a replan the allowance was
    already spent and the next cycle could not look twice despite never having done so."""
    sink, recorder = ListSink(), Recorder()
    node = make_node(
        spec_for(
            "verify",
            outcome=RoleOutcome(artifact=None, transcript=[]),
            state_slot="verify_result",
            failure_sets_error=False,
        ),
        deps(recorder, sink),
    )
    first = await node({"session_id": "s1"})
    assert "verify_retries" not in first

    unsure = verify_result(
        recovered=False,
        verdict="inconclusive",
        summary="no signal",
        needs_replan=True,
        replan_reason="the plan did not address the failing binding",
    )
    again = await node({"session_id": "s1", "verify_result": unsure})
    assert again["verify_retries"] == 1


async def test_a_failing_role_closes_its_run_row_before_the_error_escapes() -> None:
    """A run row left open is a run nobody can account for afterwards."""
    sink, recorder = ListSink(), Recorder()

    async def boom(ctx):  # type: ignore[no-untyped-def]
        raise RuntimeError("transport gone")

    spec = RoleSpec(
        name="rca",
        submit_tool="submit_root_cause",
        submit_model=RootCause,
        state_slot="root_cause",
        prompt="rca",
        run=boom,
    )
    with pytest.raises(RuntimeError):
        await make_node(spec, deps(recorder, sink))({"session_id": "s1"})

    assert recorder.finished[-1]["status"] == "failed"
    stages = [e.payload for e in sink.events if isinstance(e.payload, StageEvent)]
    assert [s.event for s in stages] == ["entered", "failed"]
