"""What the transport puts on the wire, and what it hands back when the bridge will not.

The bridge is the only cluster access this environment has and the agent holds no
credential of its own, so a request that arrives in a shape the bridge does not accept
is not a degraded run: it is a role that can read nothing for the whole episode. The
host side is written from the same wire description, which is why the request body and
the headers are asserted literally here rather than through a helper that could agree
with a mistake.
"""

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from kira.adapters.aiopslab.transport import AiopsLabTransport
from kira.audit.journal import Journal
from kira.tools.context import ToolContext
from kira.tools.metrics import _parse_alerts
from kira.tools.toolsets import build_toolset
from kira.tools.workspace import SessionWorkspace

BRIDGE = "http://127.0.0.1:18099"
ACTION = f"{BRIDGE}/v1/action"
TOKEN = "bridge-token"
SESSION = "incident-7"
NAMESPACE = "test-social-network"


@pytest.fixture
def bridge() -> Iterator[respx.Router]:
    """A bridge that accepts every action and answers ``ok``."""
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": "ok"})
        )
        yield router


def transport(**kwargs: Any) -> AiopsLabTransport:
    """Build the transport as the adapter does, with the episode's values bound."""
    bound = AiopsLabTransport(BRIDGE, TOKEN, timeout_s=30.0, **kwargs)
    bound.namespace = NAMESPACE
    bound.stage = "rca"
    return bound


def sent(router: respx.Router) -> httpx.Request:
    return router.calls.last.request


async def test_an_action_is_one_post_carrying_the_body_the_bridge_dispatches_on(
    bridge: respx.Router,
) -> None:
    """Every field is load-bearing on the host side.

    The bridge refuses any action whose namespace is not the active problem's, and it
    permits two of its actions only from the role that should be calling them, so a
    request missing either reaches a role as a refusal it cannot act on.
    """
    await transport(session_id=SESSION).query_metrics('up{job="checkout"}')

    request = sent(bridge)
    assert str(request.url) == ACTION
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert request.headers["X-Kira-Session"] == SESSION
    assert json.loads(request.content) == {
        "server": "prometheus",
        "tool": "get_metrics",
        "arguments": {"query": 'up{job="checkout"}'},
        "namespace": NAMESPACE,
        "stage": "rca",
    }


async def test_a_session_the_adapter_has_not_bound_yet_sends_no_correlation_header(
    bridge: respx.Router,
) -> None:
    """An empty header is worse than none: it joins the bridge's log to nothing."""
    await transport().query_alerts()

    assert "X-Kira-Session" not in sent(bridge).headers


async def test_kubectl_arrives_as_the_whole_command_line_it_was_tokenised_from(
    bridge: respx.Router,
) -> None:
    """The bridge re-tokenises what it is given and accepts nothing but ``kubectl``.

    Quoting is the argument's own: a selector or a jsonpath holds spaces and braces, and
    a naive join hands the bridge a different command from the one the gate classified.
    """
    await transport().kubectl(
        ["get", "pods", "-n", NAMESPACE, "-o", "jsonpath={.items[0].metadata.name}"]
    )

    assert json.loads(sent(bridge).content)["arguments"] == {
        "cmd": f"kubectl get pods -n {NAMESPACE} -o 'jsonpath={{.items[0].metadata.name}}'"
    }


async def test_the_two_environment_tools_reach_the_bridge_with_the_role_that_called_them(
    bridge: respx.Router,
) -> None:
    """The bridge permits probing only from verify and MongoDB recovery only from
    execution, and a tool holding a context knows its own role where the transport,
    built once for the run, does not."""
    await transport().action("aiopslab", "probe_service", {"service": "user"}, stage="verify")

    body = json.loads(sent(bridge).content)
    assert (body["server"], body["tool"], body["stage"]) == ("aiopslab", "probe_service", "verify")


async def test_a_refusal_reaches_the_role_in_the_bridges_own_words() -> None:
    """The prototype's client called ``raise_for_status`` first, so every refusal the
    bridge worded reached the model as an exception's repr instead. Which flag was
    refused, or which namespace a command escaped to, is the one thing that lets a role
    repair the command rather than reissue it."""
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(
                400,
                json={"ok": False, "result": "kubectl flag '--kubeconfig' is not permitted"},
            )
        )
        answer = await transport().kubectl(["get", "pods", "--kubeconfig", "/etc/admin.conf"])

    assert answer == "error: kubectl flag '--kubeconfig' is not permitted"


async def test_a_rejected_token_says_so_rather_than_naming_a_status() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(401, json={"ok": False, "result": "unauthorized"})
        )
        assert await transport().query_alerts() == "error: unauthorized"


async def test_a_reply_that_is_not_the_bridges_own_shape_is_named_as_one() -> None:
    """A proxy's error page reaches the same port and decodes as JSON."""
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(return_value=httpx.Response(200, json=["not", "an", "object"]))
        assert await transport().query_alerts() == (
            "error: AIOpsLab bridge returned a non-object response"
        )


async def test_a_bridge_that_is_not_there_names_the_action_that_could_not_run() -> None:
    """Named, because the bridge dies with the benchmark process that owns it and a
    role reading only "call failed" cannot tell that from a query it wrote wrongly."""
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=httpx.ConnectError("no route to host"))
        answer = await transport().query_traces("services")

    assert answer == (
        "error: AIOpsLab bridge jaeger.get_services call failed: ConnectError: no route to host"
    )


async def test_an_answer_with_nothing_in_it_says_that_it_is_empty() -> None:
    """An empty result gives a greedy decoder nothing to react to, so it reissues the
    same call; naming the emptiness costs one line."""
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(return_value=httpx.Response(200, json={"ok": True, "result": ""}))
        assert await transport().kubectl(["get", "pods"]) == "(empty AIOpsLab bridge response)"


async def test_an_alert_payload_is_re_encoded_so_the_partitioning_can_read_it() -> None:
    """The bridge writes JSON and the partitioning parses with ``ast.literal_eval``
    alone, so an alert carrying ``null`` or ``true`` — every silenced or inactive one
    does — returns to the model unparsed, unscoped and unlabelled, with nothing to say
    another namespace's incident is in it."""
    alerts = [
        {"labels": {"alertname": "PodCrashLooping"}, "value": None, "state": "firing"},
        {"labels": {"alertname": "HighLatency"}, "silenced": True, "state": "firing"},
    ]
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": json.dumps(alerts)})
        )
        answer = await transport().query_alerts()

    assert _parse_alerts(answer) == alerts


async def test_a_kubectl_read_is_returned_exactly_as_the_command_wrote_it() -> None:
    """Two consumers parse this text as JSON. Re-encoding it as a Python literal would
    make the namespace snapshot and the trace scoping's inventory unreadable, and both
    report that as an empty namespace rather than as a failure."""
    payload = json.dumps({"items": [{"metadata": {"name": "user", "enabled": True}}]})
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": payload})
        )
        assert await transport().kubectl(["get", "svc", "-o", "json"]) == payload


async def test_an_oversized_answer_is_cut_where_the_measured_transport_cuts_it() -> None:
    """The SREGym servers bound every response at ten thousand characters and append a
    marker. Carrying the same bound here is what keeps a role reading results of one
    size whichever environment it is bound to; a role that could ask this bridge for an
    unbounded read would behave differently under the two."""
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": "x" * 12_000})
        )
        answer = await transport().kubectl(["get", "events"])

    assert answer == "x" * 10_000 + "… [truncated]"


async def test_every_call_opens_and_closes_its_own_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection held across an episode dies part-way through it, and every later
    call then fails contiguously — after the diagnosis has been graded and while the
    remediation is being attempted, so it is invisible while it happens. Reconnecting
    per call is what makes that impossible, and it is exactly the kind of thing a later
    optimisation reintroduces on the reasonable-sounding grounds that it is wasteful."""
    opened = 0

    class _Counting(httpx.AsyncClient):
        async def __aenter__(self) -> "_Counting":
            nonlocal opened
            opened += 1
            await super().__aenter__()
            return self

    monkeypatch.setattr(httpx, "AsyncClient", _Counting)
    with respx.mock(assert_all_called=False) as router:
        router.post(ACTION).mock(
            return_value=httpx.Response(200, json={"ok": True, "result": "ok"})
        )
        bound = transport()
        await bound.kubectl(["get", "pods"])
        await bound.kubectl(["get", "svc"])
        await bound.query_metrics("up")

    assert opened == 3


def test_a_sub_second_timeout_still_bounds_a_blocking_wait() -> None:
    """The kubectl tool subtracts from this to bound a wait just under it, so a value
    truncated to zero gives every call a deadline that has already passed."""
    assert AiopsLabTransport(BRIDGE, TOKEN, timeout_s=0.4).timeout_seconds == 1


async def test_a_trace_query_the_bridge_does_not_serve_is_refused_without_a_call(
    bridge: respx.Router,
) -> None:
    assert await transport().query_traces("spans") == "error: unknown trace operation 'spans'"  # type: ignore[arg-type]
    assert not bridge.calls


def test_no_role_is_offered_a_log_tool_under_this_transport(tmp_path: Any) -> None:
    """A5. AIOpsLab deploys no log backend, and in the prototype that fact lived as set
    membership inside the global function deciding what every role could call. Here it
    is two words missing from a frozenset, and the drop is recorded so a later reader
    can tell "the model did not look at logs" from "there were no logs to look at"."""
    ctx = ToolContext(
        transport=transport(),
        workspace=SessionWorkspace(tmp_path),
        role="rca",
        session_id=SESSION,
        journal=Journal(session_id=SESSION, stage="rca"),
    )

    for role, expected in (
        ("verify", {"get_logs"}),
        ("rca", {"get_logs", "get_labels", "get_label_values"}),
    ):
        resolved = build_toolset(role, ctx)
        offered = {definition.name for definition in resolved.definitions}

        assert offered.isdisjoint({"get_logs", "get_labels", "get_label_values"})
        assert {
            name: reason for name, reason in resolved.dropped.items() if name in expected
        } == dict.fromkeys(expected, "transport lacks logs")
        assert "get_metrics" in offered and "get_services" in offered
