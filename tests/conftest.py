"""Shared fixtures.

Settings read the process environment, and a developer's shell or a sourced `.env`
would otherwise leak into assertions. Every test runs against an environment scrubbed
of anything the settings model recognises.
"""

import os
from collections.abc import Iterator

import pytest

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
