"""The SREGym adapter, and the eval contract an outside harness already depends on.

Four things are pinned here and nothing else. What an agent is told about an episode,
because a leak there invalidates every comparison with a blind baseline. What a judge is
shown, because that text is what every published diagnosis score was computed on. That
one tool call is one connection, because a held session died mid-episode and took the
rest of the run's cluster access with it. And the four HTTP calls the external driver
makes, because they were written against the prototype and cannot move.
"""

import ast
import json
import pathlib
from typing import Any

import httpx
import pytest
import respx

from kira.adapters.sregym import SREGymAdapter, SregymConfig
from kira.adapters.sregym.episode import reduce_feedback
from kira.adapters.sregym.evidence import EvidenceFilter
from kira.adapters.sregym.signal import build_diagnosis_solution, build_signal
from kira.adapters.sregym.transport import McpTransport
from kira.api.app import create_app
from kira.audit.journal import Journal, ListSink
from kira.llm.types import ModelRequest, ModelResponse, ToolCall, Usage
from kira.memory.factory import build_storage
from kira.orchestration.runner import Deployment
from kira.settings.groups import MemorySettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.workspace import SessionWorkspace

PROFILE = load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models"))

APP = {
    "app_name": "hotel-reservation",
    "namespace": "test-hotel-reservation",
    "descriptions": "a hotel booking microservice application",
}
PROBLEM = {"problem_id": "k8s_target_port-misconfig", "namespace": "leaked-namespace"}


def test_the_agent_is_told_the_symptom_and_never_the_answer() -> None:
    """A real alert names a symptom. The problem id is the ground-truth fault label, and
    nothing derived from the problem may reach the signal except the namespace, which is
    a scoping hint rather than a label."""
    leaked = dict.fromkeys(("problem_id", "id", "name", "fault", "description"), "LEAKED")
    signal = build_signal(leaked, APP)

    assert "LEAKED" not in json.dumps(signal)
    assert signal == {
        "alertname": "ServiceHealthDegraded",
        "signal_role": "generic_incident_trigger",
        "signal_instruction": (
            "This is a generic intake placeholder, not a real monitoring alert. "
            "Discover the concrete firing alert(s) and degraded SLI from live telemetry; "
            "do not use disappearance of this placeholder as recovery evidence."
        ),
        "namespace": "test-hotel-reservation",
        "service": "hotel-reservation",
        "app_name": "hotel-reservation",
        "description": "App hotel-reservation: a hotel booking microservice application",
    }


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (None, "No diagnosis produced."),
        ({"solution": "the service targets a dead port"}, "the service targets a dead port"),
        ({"solution": "  "}, "No diagnosis produced."),
        (
            {"fault_type": "misconfig", "affected_resource": "svc/user", "summary": "wrong port"},
            "Fault type: misconfig. Affected resource: svc/user. wrong port",
        ),
        (
            {"fault_type": "misconfig", "evidence": [str(i) for i in range(6)]},
            "Fault type: misconfig. Affected resource: unknown. Evidence: 0; 1; 2; 3; 4",
        ),
    ],
)
def test_the_judge_sees_exactly_what_it_saw_before(artifact: Any, expected: str) -> None:
    """Byte-compared against the prototype's rendering. The unstructured case is the
    single-agent baseline's own free text, passed through as the benchmark's reference
    agents submit it; the sixth evidence item is dropped; and it never returns empty,
    because an empty diagnosis produced no result row at all and the problem vanished
    from the denominator rather than counting as a miss."""
    assert build_diagnosis_solution(artifact) == expected


@pytest.mark.parametrize("mode", ["scores", "details"])
def test_the_grades_reach_the_distilling_pass_under_the_keys_it_reads(mode: str) -> None:
    """The packet is an untyped mapping, so a renamed key degrades the lesson silently
    rather than failing."""
    results = {
        "Diagnosis": {"success": True, "accuracy": 0.75, "reasoning": "named the right port"},
        "Mitigation": {"success": False},
        "ground_truth_root_cause": "targetPort points at nothing",
    }
    packet = reduce_feedback(results, details=mode == "details").to_packet()

    expected = {
        "source": "sregym_oracle",
        "diagnosis": {"success": True, "score": 0.75},
        "mitigation": {"success": False},
    }
    if mode == "details":
        expected["diagnosis"]["reasoning"] = "named the right port"
        expected["ground_truth_root_cause"] = "targetPort points at nothing"
    assert packet == expected


def test_a_judge_that_crashed_is_not_recorded_as_a_wrong_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The harness writes `judgment=Error` beside `success=False` and `accuracy=0.0` when
    its own judge fails to grade, which downstream is indistinguishable from a genuine
    miss. Believed, it becomes an authoritative correction in the corpus — "the cause was
    this and you were wrong" — which outlives the run that recorded it. The cluster oracle
    graded independently and survives."""
    results = {
        "Diagnosis": {
            "judgment": "Error",
            "success": False,
            "accuracy": 0.0,
            "reasoning": "Error: 'answer'",
            "error": "'answer'",
        },
        "Mitigation": {"success": True},
        "ground_truth_root_cause": "targetPort points at nothing",
    }
    with caplog.at_level("ERROR"):
        packet = reduce_feedback(results, details=True).to_packet()

    assert packet["diagnosis"] == {"success": None, "score": None, "judge_error": "'answer'"}
    assert packet["mitigation"] == {"success": True}
    assert "failed to grade" in caplog.text


class TestEvidenceFilter:
    """The harness can leak its own fault injection into the telemetry it serves, which
    turns a diagnosis into a lookup. That leak was measured, and in one problem an agent
    diagnosed the injector's own fault against a disk-sector ground truth."""

    def test_the_injectors_own_resources_are_removed_from_structured_output(self) -> None:
        body = json.dumps(
            {
                "items": [
                    {"metadata": {"name": "user", "namespace": "shop"}},
                    {"metadata": {"name": "netchaos", "namespace": "chaos-mesh"}},
                    {"metadata": {"name": "loadgen", "labels": {"app": "load-generator"}}},
                ]
            }
        )
        filtered = json.loads(EvidenceFilter().filter_result(body))
        assert [item["metadata"]["name"] for item in filtered["items"]] == ["user"]

    def test_a_refusal_stays_visible_as_one_bounded_line(self) -> None:
        """Useful operational evidence that does not confirm anything was found."""
        body = "Error from server (Forbidden): pods is forbidden\n" + "x" * 900
        assert EvidenceFilter().filter_result(body) == (
            "Error from server (Forbidden): pods is forbidden\n"
        )

    def test_output_with_nothing_to_hide_is_returned_unchanged(self) -> None:
        body = "NAME   READY   STATUS\nuser   1/1     Running\n"
        assert EvidenceFilter().filter_result(body) == body

    def test_a_body_whose_shape_it_does_not_know_still_loses_the_injector(self) -> None:
        """Every rule but one read a Kubernetes shape, and being valid JSON is what chose
        this filter over the line-based one — so an unrecognised shape got the filter that
        did nothing instead of the filter that would have worked.

        A Prometheus result is the case that matters: the series carries the injector's
        experiment name and the service it was aimed at, which is the diagnosis.
        """
        body = json.dumps(
            {
                "status": "success",
                "data": {
                    "result": [
                        {"metric": {"__name__": "up", "namespace": "shop", "pod": "user-1"}},
                        {
                            "metric": {
                                "__name__": "chaos_mesh_experiments",
                                "namespace": "chaos-mesh",
                                "experiment": "pod-failure-user-svc",
                            }
                        },
                    ]
                },
            }
        )
        filtered = EvidenceFilter().filter_result(body)

        assert "chaos" not in filtered
        assert "pod-failure-user-svc" not in filtered
        # The real series survives: this is a filter, not a refusal.
        assert json.loads(filtered)["data"]["result"] == [
            {"metric": {"__name__": "up", "namespace": "shop", "pod": "user-1"}}
        ]

    def test_a_result_in_the_encoding_the_transports_emit_keeps_its_real_series(
        self,
    ) -> None:
        """Neither transport emits JSON: both render structured telemetry as the Python
        repr of a list of dictionaries. A repr is one line, so it fell to the line-based
        filter, lost that line, and came back as an empty cluster with no ``error:``
        prefix for the loop or the audit log to notice.
        """
        body = str(
            {
                "resultType": "vector",
                "result": [
                    {"metric": {"__name__": "up", "pod": "user-1"}, "value": [1757000000, "0"]},
                    {"metric": {"__name__": "up", "pod": "locust-master-0"}},
                ],
            }
        )
        filtered = EvidenceFilter().filter_result(body)

        assert "locust" not in filtered
        # Still a repr: the alert partitioning downstream reads one with literal_eval.
        assert ast.literal_eval(filtered)["result"] == [
            {"metric": {"__name__": "up", "pod": "user-1"}, "value": [1757000000, "0"]}
        ]

    def test_a_body_it_removed_entirely_is_not_reported_as_an_empty_cluster(self) -> None:
        """An empty cluster is a claim about the cluster, and a model acts on it."""
        assert EvidenceFilter().filter_result("locust-master-0   1/1   Running\n") == (
            "Result withheld. This is not a finding that nothing was there.\n"
        )

    def test_the_injector_is_lost_even_when_it_is_the_whole_response(self) -> None:
        """A single-resource read is a bare object, so the rule that drops a hidden list
        element never saw it: ``get networkchaos <name> -o json`` handed back the
        injected fault's action, target and magnitude, which is the diagnosis.
        """
        body = json.dumps(
            {
                "apiVersion": "chaos-mesh.org/v1alpha1",
                "kind": "NetworkChaos",
                "metadata": {"name": "delay-geo", "namespace": "test-hotel-reservation"},
                "spec": {
                    "action": "delay",
                    "selector": {"labelSelectors": {"io.kompose.service": "geo"}},
                    "delay": {"latency": "500ms"},
                },
            }
        )
        filtered = EvidenceFilter().filter_result(body)

        assert "geo" not in filtered
        assert "delay" not in filtered

    def test_a_trace_naming_the_load_generator_is_dropped(self) -> None:
        body = json.dumps(
            {
                "data": [
                    {"traceID": "a1", "spans": [{"operationName": "GET /cart"}]},
                    {"traceID": "b2", "spans": [{"process": {"serviceName": "load-generator"}}]},
                ]
            }
        )
        assert [t["traceID"] for t in json.loads(EvidenceFilter().filter_result(body))["data"]] == [
            "a1"
        ]


def test_an_ungraded_episode_is_not_stamped_with_the_oracles_name() -> None:
    """The function promised None when nothing gradable came back, and returned a record
    with every field null and the oracle named as its source.

    That name is what the distilling pass reads as the authority for a verdict, so an
    episode nobody graded became one the oracle had evaluated and found nothing in. A
    partial result is still a result — one stage graded and the other not is real — so
    only the case with neither stage present is nothing.
    """
    assert reduce_feedback({}, details=True) is None
    assert reduce_feedback({"ground_truth_root_cause": "a cause"}, details=True) is None
    assert reduce_feedback({"Mitigation": {"success": True}}, details=True) is not None
    assert reduce_feedback({"Diagnosis": {"success": False}}, details=True) is not None


class _Session:
    """One MCP session, counting how many times its transport was opened."""

    def __init__(self, opened: list[str]) -> None:
        self._opened = opened

    async def initialize(self) -> None:
        return None

    async def call_tool(self, name: str, arguments: Any, **kwargs: Any) -> Any:
        block = type("Block", (), {"text": "ok"})()
        return type("Result", (), {"content": [block], "is_error": False})()

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _Connection:
    """The server-sent-event connection, recording the headers it was opened with."""

    def __init__(self, opened: list[str], headers: dict[str, str]) -> None:
        self._opened = opened
        self._headers = headers

    async def __aenter__(self) -> tuple[None, None]:
        self._opened.append(self._headers.get("sregym_ssid", ""))
        return (None, None)

    async def __aexit__(self, *exc: Any) -> None:
        return None


async def test_each_tool_call_opens_and_closes_its_own_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session held for an episode dies partway through it and every later call fails
    with a message about invalid parameters, contiguously — 59% of calls across cells
    making forty-five or more, 84% in the worst. The correlation id rides on each of
    those connections, because it is what the rollback journal is keyed by."""
    opened: list[str] = []
    monkeypatch.setattr(
        "kira.adapters.sregym.transport.sse_client",
        lambda url, headers, timeout, sse_read_timeout: _Connection(opened, headers),
    )
    monkeypatch.setattr(
        "kira.adapters.sregym.transport.ClientSession", lambda read, write: _Session(opened)
    )

    transport = McpTransport("http://mcp:9954")
    transport.session_id = "session-7"
    await transport.kubectl(["get", "pods"])
    await transport.query_alerts()
    await transport.rollback_last()

    assert opened == ["session-7", "session-7", "session-7"]


async def test_a_transport_failure_is_a_result_the_loop_can_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Far more of the system reads the ``error:`` prefix than reads any exception, so a
    transport that raised would be counted as a successful turn by all of it."""

    def explode(**kwargs: Any) -> Any:
        raise ConnectionRefusedError("nothing listening")

    monkeypatch.setattr("kira.adapters.sregym.transport.sse_client", lambda *a, **k: explode())
    result = await McpTransport("http://mcp:9954").query_logs('{app="user"}')
    assert result.startswith("error: MCP loki.get_logs call failed: ConnectionRefusedError")


class _Conductor:
    """A conductor that advances its stage machine only as answers arrive."""

    def __init__(self) -> None:
        self.stage = "diagnosis"
        self.submissions: list[dict[str, Any]] = []
        self.seen: list[tuple[str, str]] = []

    def route(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((path, request.headers.get("sregym_ssid", "")))
        if path == "/get_problem":
            return httpx.Response(200, json=PROBLEM)
        if path == "/get_app":
            return httpx.Response(200, json=APP)
        if path == "/status":
            return httpx.Response(200, json={"stage": self.stage})
        body = json.loads(request.content)
        self.submissions.append(body)
        self.stage = {"diagnosis": "mitigation", "mitigation": "accept"}.get(body["stage"], "done")
        return httpx.Response(200, json={"ok": True})


class _Model:
    """Answers each role with its submission tool, so no cluster is needed."""

    async def complete(self, request: ModelRequest, *, caller: str) -> ModelResponse:
        arguments = {
            "rca": {
                "fault_type": "misconfiguration",
                "affected_resource": "svc/user",
                "confidence": 0.8,
                "evidence": ["endpoints empty"],
                "summary": "the user Service targets a dead port",
            },
            "planning": {
                "steps": [
                    {"action": "patch", "target": "svc/user", "params": {}, "justification": "fix"}
                ],
                "expected_outcome": "endpoints present",
                "confidence": 0.8,
            },
            "execution": {
                "succeeded": True,
                "steps": [{"step_index": 0, "succeeded": True, "observation": "patched"}],
                "final_state_summary": "endpoints present",
                "confidence": 0.8,
            },
        }[caller]
        name = {
            "rca": "submit_root_cause",
            "planning": "submit_plan",
            "execution": "submit_execution_result",
        }[caller]
        return ModelResponse(
            text=None,
            tool_calls=(
                ToolCall(
                    id="c1", name=name, arguments=arguments, raw_arguments=json.dumps(arguments)
                ),
            ),
            finish_reason="stop",
            reasoning=None,
            usage=Usage(),
            duration_ms=1,
            attempts=1,
        )


class _FakeTransport:
    name = "fake"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    async def kubectl(self, argv: list[str]) -> str:
        return "error: no cluster in this test"


async def test_the_external_driver_gets_the_reply_and_the_submissions_it_expects(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The held eval contract, end to end against a stubbed conductor.

    The acceptance body's four keys and its 202 are what the kickoff driver parses; the
    polled record's three keys are what it waits on; and the conductor sees exactly one
    submission per stage, in order, with the release last. Every request carries the
    correlation id the rollback journal is keyed by.
    """
    monkeypatch.setattr(SREGymAdapter, "tool_transport", lambda self: _FakeTransport())
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(enabled=False, learning_enabled=False),
        verify=VerifySettings(enabled=False),
        adapters={"sregym": {"stage_poll_s": 0.01}},
        intake={"adapters": ["sregym"]},  # type: ignore[arg-type]
    )
    storage = build_storage(settings)
    deployment = Deployment(
        settings=settings,
        storage=storage,
        completion=_Model(),
        profile=PROFILE,
        transport_for=lambda session_id: _FakeTransport(),  # type: ignore[arg-type, return-value]
        sink=ListSink(),
    )
    app = create_app(deployment=deployment, schedule_resume=lambda request_id: None)
    conductor = _Conductor()

    with respx.mock:
        respx.route(host="localhost", port=8000).mock(side_effect=conductor.route)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://kira"
        ) as client:
            accepted = await client.post("/api/v1/sregym/episodes?submit=true")
            assert accepted.status_code == 202
            assert set(accepted.json()) == {"session_id", "status", "problem_id", "started_at"}
            assert accepted.json()["status"] == "accepted"
            assert accepted.json()["problem_id"] == "k8s_target_port-misconfig"

            session_id = accepted.json()["session_id"]
            await _settle(app)
            polled = (await client.get(f"/api/v1/incidents/{session_id}")).json()

    assert {"run_status", "current_stage", "status"} <= set(polled)
    assert polled["run_status"] == "completed"
    assert [s["stage"] for s in conductor.submissions] == ["diagnosis", "mitigation", "accept"]
    assert conductor.submissions[0]["solution"].startswith("Fault type: misconfiguration.")
    assert conductor.submissions[1]["solution"] == ""
    # The first two reads happen before the session exists, which is what creates it.
    # Everything from the binding onwards carries the id the rollback journal is keyed by.
    assert {
        ssid for path, ssid in conductor.seen if path != "/get_problem" and path != "/get_app"
    } == {session_id}


def test_the_rollback_pair_is_offered_only_where_it_was_measured(tmp_path: pathlib.Path) -> None:
    """These are the harness's own session state, not Kubernetes operations, so they are
    absent from the core registry and contributed here. Their place in the order matters
    as much as their presence: the order a role sees its tools declared in was hand-tuned
    and every published measurement was taken with it, and in the configuration these
    came from they sat immediately before the submission tool."""
    from kira.adapters.sregym.tools import adapter_tools  # noqa: PLC0415
    from kira.tools.context import ToolContext  # noqa: PLC0415
    from kira.tools.toolsets import build_toolset  # noqa: PLC0415

    ctx = ToolContext(
        transport=McpTransport("http://mcp:9954"),
        workspace=SessionWorkspace(tmp_path),
        role="execution",
        session_id="s1",
        journal=Journal(session_id="s1", stage="execution"),
    )
    offered = [
        tool.name for tool in build_toolset("execution", ctx, extras=adapter_tools()).definitions
    ]
    diagnosing = build_toolset("rca", ctx, extras=adapter_tools()).definitions

    assert offered[-3:] == [
        "rollback_command",
        "get_previous_rollbackable_cmd",
        "submit_execution_result",
    ]
    assert "rollback_command" not in [tool.name for tool in diagnosing]


async def test_a_deployment_that_never_saw_the_harness_reads_its_defaults() -> None:
    """The conductor's own address is where the two names that used to fight over the
    bind port now land."""
    config = SregymConfig.resolve(Settings(), {"API_HOSTNAME": "conductor", "API_PORT": "8000"})
    assert str(config.conductor_base_url).rstrip("/") == "http://conductor:8000"
    assert config.feedback_mode == "none"


async def _settle(app: Any) -> None:
    """Wait for the episode this process started."""
    import asyncio  # noqa: PLC0415

    while pending := {task for task in app.state.service.tasks if not task.done()}:
        await asyncio.gather(*pending, return_exceptions=True)
