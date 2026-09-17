"""Tests for the namespace health gate.

Three things earn a test here. The fail-open property, because a gate that blocked when
it could not look would turn an infrastructure fault into a false incident and spend the
retry budget on it. The two calibration constants, because each was tuned against a
measured false block and a test is the only thing standing between the numbers and a
well-meaning tidy-up. And the parse, because the gate falls open on anything it cannot
read, so a transport handing back a Python repr instead of JSON silently disables it.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from maver.audit.context import ToolObservation
from maver.audit.journal import Journal
from maver.roles.health_gate import HealthGate, sweep_namespace_health
from maver.tools.context import ToolContext
from maver.tools.workspace import SessionWorkspace


class FakeCluster:
    """Answers the one pod read, with a payload or an exception."""

    name = "fake"
    capabilities = frozenset({"kubernetes"})
    timeout_seconds = 30

    def __init__(self, payload: str | Exception) -> None:
        self.payload = payload

    async def kubectl(self, argv: list[str]) -> str:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def context(tmp_path: Path, payload: str | Exception) -> ToolContext:
    return ToolContext(
        transport=FakeCluster(payload),
        workspace=SessionWorkspace(tmp_path),
        role="verify",
        session_id="s1",
        journal=Journal(session_id="s1", stage="verify"),
    )


def pod(
    name: str,
    *,
    phase: str = "Running",
    ready: bool = True,
    restarts: int = 0,
    waiting: str = "",
    age_seconds: int = 3600,
    deleting: bool = False,
) -> dict[str, Any]:
    started = datetime.now(UTC) - timedelta(seconds=age_seconds)
    container: dict[str, Any] = {"name": f"{name}-app", "ready": ready, "restartCount": restarts}
    if waiting:
        container["state"] = {"waiting": {"reason": waiting}}
    meta: dict[str, Any] = {"name": name}
    if deleting:
        meta["deletionTimestamp"] = started.isoformat()
    return {
        "metadata": meta,
        "status": {
            "phase": phase,
            "startTime": started.isoformat(),
            "containerStatuses": [container],
        },
    }


async def sweep(tmp_path: Path, payload: str | Exception) -> HealthGate:
    return await sweep_namespace_health(context(tmp_path, payload), "shop")


def listing(pods: list[dict[str, Any]]) -> str:
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": pods})


async def test_a_gate_that_could_not_look_never_blocks(tmp_path: Path) -> None:
    raised = await sweep(tmp_path, RuntimeError("transport down"))
    refused = await sweep(tmp_path, "error: read_kubectl only runs read-only kubectl")
    unparseable = await sweep(tmp_path, "<html>gateway timeout</html>")

    # A body it could not read is a sweep that did not happen, and belongs with the two
    # that could not run at all. It used to count as one that ran: that kept the gate off
    # the blocking path, which was right, but it also rendered the healthy block into
    # Verify's opening turn — the model read "every pod Running/Ready" off a response
    # nothing had parsed — and wrote a clean namespace into the audit row.
    for gate in (raised, refused, unparseable):
        assert gate.checked is False
        assert gate.blocks_recovered is False
        assert gate.error
        assert gate.context_text() == ""
    assert "unparseable" in (unparseable.error or "")

    # A healthy namespace and a gate that never looked both produce zero blocks, which is
    # the pair the audit record and the context block exist to tell apart.
    healthy = await sweep(tmp_path, listing([pod("frontend")]))
    assert healthy.checked is True
    assert healthy.failures == ()
    assert healthy.blocks_recovered is False
    assert healthy.context_text() != ""
    assert healthy.audit()["checked"] is not raised.audit()["checked"]


async def test_each_flag_rule_fires(tmp_path: Path) -> None:
    gate = await sweep(
        tmp_path,
        listing(
            [
                pod("pending", phase="Pending", ready=False),
                pod("unready", ready=False),
                pod("flapping", restarts=11),
            ]
        ),
    )

    assert gate.blocks_recovered is True
    assert len(gate.failures) == 3
    listed = "\n".join(gate.failures)
    assert "pod pending is Pending, not Running" in listed
    assert "pod unready is Running but container(s) unready-app are not Ready" in listed
    assert "pod flapping is Ready but has restarted 11 times" in listed
    assert "You cannot submit recovered while any of them stands." in gate.context_text()
    assert "recovered=true is not available" in gate.rejection_text()


async def test_the_calibration_boundaries_do_not_fire(tmp_path: Path) -> None:
    """Exactly at the restart ceiling, and inside the transient grace.

    Both are the false blocks the constants were tuned to stop: a startup-churn pod held a
    correctly-mitigated cell across all three verify passes, and a frontend pod caught
    mid-restart cost a false block on duplicate_pvc_mounts_hotel_reservation.
    """
    gate = await sweep(
        tmp_path,
        listing(
            [
                pod("churny", restarts=10),
                pod(
                    "starting",
                    phase="Pending",
                    ready=False,
                    waiting="ContainerCreating",
                    age_seconds=10,
                ),
            ]
        ),
    )

    assert gate.checked is True
    assert gate.failures == ()
    assert gate.blocks_recovered is False
    assert any("still starting" in note for note in gate.observations)


async def test_deleting_and_succeeded_pods_are_skipped(tmp_path: Path) -> None:
    gate = await sweep(
        tmp_path,
        listing(
            [
                pod("rolling", phase="Pending", ready=False, deleting=True),
                pod("migration-job", phase="Succeeded", ready=False),
            ]
        ),
    )

    assert gate.checked is True
    assert gate.failures == ()


async def test_a_python_dict_payload_still_parses(tmp_path: Path) -> None:
    gate = await sweep(tmp_path, repr({"items": [pod("unready", ready=False)]}))

    assert gate.checked is True
    assert gate.blocks_recovered is True


async def test_a_read_that_ran_and_failed_is_not_a_clean_sweep(tmp_path: Path) -> None:
    """The direct transport reports a failed read as an exit code, not as `error:`.

    Screened on the prefix alone it reached the parse, which returns no failures for a
    body it cannot read — so a gate that had never seen the namespace came back
    `checked=True, failures=()`, which is a positive assertion of health and the one
    thing this gate exists to withhold.
    """
    gate = await sweep(
        tmp_path,
        'exit_code=1\nError from server (NotFound): namespaces "shop" not found',
    )

    assert gate.checked is False
    assert gate.blocks_recovered is False
    assert gate.error
    assert gate.context_text() == ""


@pytest.mark.parametrize(
    "unknown", [None, {"metadata": {"name": "unknown"}, "status": {"phase": "Running"}}]
)
@pytest.mark.parametrize("include_failure", [False, True])
async def test_unknown_pods_cannot_be_called_healthy_and_do_not_hide_observed_failures(
    tmp_path: Path, unknown: object, include_failure: bool
) -> None:
    items = [unknown]
    if include_failure:
        items.append(pod("broken", phase="Pending", ready=False))
    observation = ToolObservation()
    ctx = replace(context(tmp_path, json.dumps({"items": items})), observation=observation)
    gate = await sweep_namespace_health(ctx, "shop")
    assert observation.incomplete
    assert observation.unavailable
    assert gate.unavailable
    assert gate.audit()["complete"] is False
    assert "unknown" in gate.context_text().lower()
    assert "Every pod" not in gate.context_text()
    assert gate.blocks_recovered is include_failure
    if include_failure:
        assert any("broken" in failure for failure in gate.failures)
