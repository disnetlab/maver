"""Reading an incident's trusted context out of its alert."""

from typing import Any

import pytest

from maver.tools.scope import IncidentScope


@pytest.mark.parametrize(
    ("alert", "expected"),
    [
        ({"namespace": "test-hotel-reservation"}, "test-hotel-reservation"),
        ({"labels": {"namespace": "shop"}}, "shop"),
        ({"labels": {"exported_namespace": "shop"}}, "shop"),
        ({"namespace": "  shop  "}, "shop"),
        # Rejected rather than passed through. Everything downstream reads None as
        # "no scope" and therefore withholds; a malformed value that survived would
        # instead be compared against real namespaces and quietly match nothing, or
        # reach a command line.
        ({"namespace": "Shop"}, None),
        ({"namespace": "shop; rm -rf /"}, None),
        ({"namespace": ""}, None),
        ({}, None),
        (None, None),
    ],
)
def test_the_namespace_is_validated_not_taken(alert: Any, expected: str | None) -> None:
    assert IncidentScope.from_alert(alert).namespace == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-08-30T11:00:00Z", "2026-08-30T11:00:00Z"),
        ("2026-08-30T11:00:00+00:00", "2026-08-30T11:00:00+00:00"),
        ("yesterday", None),
        ("", None),
        (1756550400, None),
    ],
)
def test_the_observation_window_is_validated_not_taken(raw: Any, expected: str | None) -> None:
    scope = IncidentScope.from_alert({"observation_window_start": raw})
    assert scope.observation_window_start == expected
