"""Fixtures shared by the unit suite."""

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def shim(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Stand in for ``kira.tools.kubectl_proxy``, recording what it was handed.

    The sandbox tools reach the real module by name at call time, so a stand-in in
    ``sys.modules`` is what a caller sees. It is a test double rather than a second
    implementation: what these tests assert is the arguments computed for it — the
    stage above all — which is the sandbox tools' half of that contract.

    Yields:
        The recorded calls, in order.
    """
    calls: list[dict[str, Any]] = []
    module = types.ModuleType("kira.tools.kubectl_proxy")

    def apply_proxy_env(env: dict[str, str], **kwargs: Any) -> dict[str, str]:
        calls.append({"env": dict(env), **kwargs})
        return {**env, "KUBECTL_MODE": "proxy_exec", "AGENT_STAGE": str(kwargs["stage"])}

    module.apply_proxy_env = apply_proxy_env  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kira.tools.kubectl_proxy", module)
    yield calls
