"""Opening the configured stores, once, at startup."""

import pathlib

import pytest

from kira.memory.factory import build_storage
from kira.settings.groups import MemorySettings, ServerSettings
from kira.settings.root import Settings


def settings_for(tmp_path: pathlib.Path, **memory: object) -> Settings:
    return Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(**memory),  # type: ignore[arg-type]
    )


def test_the_embedded_database_lives_under_this_deployment_s_workspace(
    tmp_path: pathlib.Path,
) -> None:
    """Two deployments sharing a machine must not share a corpus: the workspace is
    what separates them, so the database has to follow it rather than a fixed path."""
    stack = build_storage(settings_for(tmp_path))
    try:
        session = stack.sessions.create_session({"alertname": "HighErrorRate"})
        assert stack.sessions.get_session(session.session_id) is not None
        assert (tmp_path / "memory" / "default.sqlite").is_file()
    finally:
        stack.close()


def test_an_unavailable_backend_fails_at_startup_and_says_what_to_install(
    tmp_path: pathlib.Path,
) -> None:
    """A deployment that cannot persist has nothing useful to do, so it should not
    discover that on its first incident."""
    with pytest.raises(ValueError, match=r"kira\[postgres\]"):
        build_storage(settings_for(tmp_path, backend="postgres", postgres_dsn="postgresql://x/y"))
