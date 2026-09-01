"""What the contract itself is worth.

A property that cannot fail is worse than no property: it passes because a stub returned
what the assertion expected, and it goes on passing through exactly the change it was
written to catch. So the suite is run here against a transport that satisfies it, and
then against one-line deviations from that transport, each of which has to be rejected
by the property it breaks. The deviations double as the plainest statement of what each
property is for, and the last test refuses to let a property exist without one.
"""

import json
from collections.abc import Sequence
from inspect import isawaitable, signature
from typing import Any

import pytest
from transport_contract import ALERTS, KUBECTL_JSON, NAMESPACE, SERVICES, ToolTransportContract

from kira.tools.errors import tool_error

UNREACHABLE = tool_error("the backend is not reachable")

CHECKS = [name for name in dir(ToolTransportContract) if name.startswith("test_")]


class _Conforming:
    """A transport that satisfies every property, so one deviation can be made alone.

    It plays both fixtures. ``reachable=False`` is the same implementation with its
    backend gone, which is the only way one object can stand in for both halves of the
    contract.
    """

    name = "conforming"
    capabilities = frozenset({"kubernetes", "kubernetes.write", "metrics", "logs", "traces"})
    timeout_seconds = 30

    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable

    def _answer(self, payload: str) -> str:
        return payload if self.reachable else UNREACHABLE

    async def kubectl(self, argv: Sequence[str]) -> str:
        return self._answer(KUBECTL_JSON)

    async def rollbackable_commands(self) -> str:
        return self._answer("nothing has been applied yet")

    async def rollback_last(self) -> str:
        return self._answer("nothing to undo")

    async def query_metrics(self, query: str, *, window_minutes: int | None = None) -> str:
        return self._answer("[]")

    async def query_alerts(self) -> str:
        return self._answer(repr(ALERTS))

    async def log_labels(self) -> str:
        return self._answer("Available labels:\n  - namespace")

    async def log_label_values(self, label: str) -> str:
        return self._answer(f"Values for label {label!r}:\n  - {NAMESPACE}")

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        return self._answer("No logs found matching the query.")

    async def query_traces(self, operation: str, **kwargs: Any) -> str:
        return self._answer(repr(SERVICES) if operation == "services" else "None")


class _MissingTraceQuery(_Conforming):
    """A transport that never grew one of the protocol's methods.

    Named and set to None rather than left out, which is what a subclass can express in
    one line and what the runtime protocol check treats as absent anyway.
    """

    query_traces = None


class _MisspeltCapability(_Conforming):
    """One character wrong, and the cluster-write tools are gone from every role."""

    capabilities = frozenset({"kubernetes", "kubernetes.writes", "metrics", "logs", "traces"})


class _SubSecondTimeout(_Conforming):
    """A sub-second timeout truncated to a whole number of seconds.

    The kubectl tool subtracts ten from this to bound a blocking wait, so a zero here is
    a negative deadline before a floor catches it.
    """

    timeout_seconds = 0


class _Raising(_Conforming):
    """A backend failure that escapes as an exception instead of a result."""

    async def query_logs(self, query: str, *, last_n_minutes: int = 15) -> str:
        raise ConnectionRefusedError("nothing listening")


class _UnmarkedFailure(_Conforming):
    """The prototype's in-band marker: a failure worded for a human and for nothing else.

    Every consumer that decides whether a turn succeeded reads the prefix, so this
    arrives as a successful call returning a puzzling result.
    """

    def _answer(self, payload: str) -> str:
        return payload if self.reachable else "could not reach the backend"


class _EnvelopedKubectl(_Conforming):
    """The mistake this suite exists for, rendered as one prefix on a successful read."""

    async def kubectl(self, argv: Sequence[str]) -> str:
        return self._answer(f"exit_code=0\n{KUBECTL_JSON}")


class _RawAlertEnvelope(_Conforming):
    """The backend's whole response where the layer above parses the list inside it.

    Nothing raises and nothing is logged. The payload is returned to the model byte for
    byte, unscoped and unlabelled, which is what the partitioning exists to prevent.
    """

    async def query_alerts(self) -> str:
        return self._answer(json.dumps({"status": "success", "data": {"alerts": ALERTS}}))


class _ProseServiceCatalogue(_Conforming):
    """A catalogue written for a reader where the trace tool parses a structured one."""

    async def query_traces(self, operation: str, **kwargs: Any) -> str:
        return self._answer("Services: checkout, frontend")


CASES = [
    (_MissingTraceQuery, "test_it_satisfies_the_protocol"),
    (_MisspeltCapability, "test_declared_capabilities_are_in_the_vocabulary"),
    (_SubSecondTimeout, "test_the_timeout_is_a_positive_whole_number_of_seconds"),
    (_Raising, "test_no_call_raises_when_the_backend_is_gone"),
    (_UnmarkedFailure, "test_a_backend_that_cannot_answer_is_legible_as_a_failure"),
    (_EnvelopedKubectl, "test_a_successful_kubectl_returns_its_own_output_unwrapped"),
    (_RawAlertEnvelope, "test_firing_alerts_reach_the_partitioning_that_scopes_them"),
    (_ProseServiceCatalogue, "test_the_service_catalogue_reaches_the_scoping_that_narrows_it"),
]


async def _run(check: str, transport: Any, unreachable: Any) -> None:
    """Run one contract check outside pytest's fixture machinery.

    Which of the two stand-ins a check wants is read from its signature rather than
    tabulated here, so a check added to the contract cannot quietly escape this file.
    """
    method = getattr(ToolTransportContract(), check)
    wants_unreachable = "unreachable" in signature(method).parameters
    outcome = method(unreachable if wants_unreachable else transport)
    if isawaitable(outcome):
        await outcome


async def test_the_contract_accepts_a_conforming_transport() -> None:
    """Otherwise the rejections below would prove only that it rejects everything."""
    for check in CHECKS:
        await _run(check, _Conforming(), _Conforming(reachable=False))


@pytest.mark.parametrize(("broken", "check"), CASES)
async def test_each_property_rejects_the_transport_that_breaks_it(
    broken: type[_Conforming], check: str
) -> None:
    with pytest.raises(AssertionError):
        await _run(check, broken(), broken(reachable=False))


def test_every_property_has_a_violation_that_breaks_it() -> None:
    """A property with no entry above is a property nobody has shown can fail."""
    assert sorted(check for _, check in CASES) == sorted(CHECKS)
