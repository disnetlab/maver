"""Shared fixtures.

Settings read the process environment, and a developer's shell or a sourced `.env`
would otherwise leak into assertions. Every test runs against an environment scrubbed
of anything the settings model recognises.
"""

import os
from collections.abc import Generator, Iterator

import pytest

from kira.adapters import registry as adapter_registry_module
from kira.adapters.registry import load_entry_point_adapters
from kira.settings.root import known_env_names, reset_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every variable the settings model would read."""
    known = known_env_names()
    for name in list(os.environ):
        if name.startswith("KIRA_") or name in known:
            monkeypatch.delenv(name, raising=False)
    # A developer's .env at the repo root must not steer the suite either.
    monkeypatch.chdir(os.environ.get("PYTEST_TMP_CWD", os.getcwd()))
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def adapter_registry() -> Iterator[None]:
    """Give a test the adapter registry to itself, and put back what was there.

    The registry is process-wide and entry points register on import, so a test that
    empties it cannot restore it by importing the packages again: they are already in
    ``sys.modules`` and their registration does not run twice. Restoring the snapshot is
    what makes such a test independent of the order it runs in, and of how many adapters
    happen to be installed.
    """
    # Discovered first, so the snapshot holds every installed adapter. Registration
    # runs on import and an import does not run twice, so an adapter that had not been
    # imported when the snapshot was taken could not be restored into it.
    load_entry_point_adapters()
    registered = dict(adapter_registry_module._REGISTRY)
    unavailable = dict(adapter_registry_module._UNAVAILABLE)
    try:
        yield
    finally:
        adapter_registry_module._REGISTRY.clear()
        adapter_registry_module._REGISTRY.update(registered)
        adapter_registry_module._UNAVAILABLE.clear()
        adapter_registry_module._UNAVAILABLE.update(unavailable)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the flag CI uses to refuse a green run that tested nothing."""
    parser.addoption(
        "--no-skips",
        action="store_true",
        help=(
            "Fail on a skipped test instead of passing. For a job that provisions the "
            "very service the test skips without: a skip there reports green while the "
            "thing under test went unexercised."
        ),
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Turn a skip into a failure when the caller said a skip is not acceptable."""
    report = yield
    # xfail arrives as a skip too, and it is a recorded expectation rather than a
    # check that did not run.
    if report.skipped and not hasattr(report, "wasxfail") and item.config.getoption("--no-skips"):
        report.outcome = "failed"
        report.longrepr = f"skipped under --no-skips: {report.longrepr}"
    return report
