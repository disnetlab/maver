"""The shipped example is the first thing an adopter runs, so it is kept working.

An example that no longer parses is worse than none: it reads as the contract and fails
in a way that looks like the system is broken rather than the documentation.
"""

import json
import pathlib

import pytest

from maver.api.incidents import IncidentRequest
from maver.roles.prompting import visible_alert
from maver.tools.scope import IncidentScope

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples" / "alerts"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.name)
def test_a_shipped_alert_is_accepted_and_scoped(path: pathlib.Path) -> None:
    alert = json.loads(path.read_text(encoding="utf-8"))

    request = IncidentRequest(alert=alert)
    scope = IncidentScope.from_alert(alert)

    assert request.alert == alert
    # An example whose namespace does not parse would demonstrate the unscoped path,
    # which is the one behaviour the telemetry wrappers exist to prevent.
    assert scope.namespace is not None
    assert scope.observation_window_start is not None
    # Nothing an example ships may be hidden from the model: an underscore key is how
    # an evaluation harness passes its own answer key, and an example carrying one
    # would teach an adopter to send it.
    assert visible_alert(alert) == alert
