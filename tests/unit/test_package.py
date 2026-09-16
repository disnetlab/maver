"""The package imports cleanly and reports a version consistent with pyproject."""

import importlib
import pathlib
import tomllib

import maver

PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert maver.__version__ == declared


def test_subpackages_import() -> None:
    """Every subpackage imports without pulling in an optional dependency."""
    for name in (
        "adapters",
        "api",
        "audit",
        "contracts",
        "llm",
        "memory",
        "middleware",
        "orchestration",
        "roles",
        "secondpass",
        "settings",
        "tools",
    ):
        importlib.import_module(f"maver.{name}")
