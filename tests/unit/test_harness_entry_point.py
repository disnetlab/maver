"""The path the evaluation harness names, and the reason it is fragile.

`scripts/sregym-driver.py` is what the harness's own agent configuration invokes. It
loads the driver from its *file* rather than importing it by package name, because it
runs in the harness's interpreter where this project is not installed — importing by name
would execute the adapter's `__init__` and fail before the driver ever ran.

That makes it a hard-coded path into the package, and a hard-coded path is exactly the
thing a refactor moves without noticing. Nothing else references it: it is not imported,
not on an entry point, and not exercised by any other test. It breaks silently, and the
place it breaks is a VM part-way through a campaign.

The driver's standard-library-only constraint is the other half of the same contract, and
it is equally invisible: adding one import from the package would pass every test here and
fail only where it matters.
"""

import ast
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
ENTRY = SCRIPTS / "sregym-driver.py"
DRIVER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "kira"
    / "adapters"
    / "sregym"
    / "driver.py"
)


def test_the_harness_entry_point_still_finds_the_driver() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    assert DRIVER.exists(), "the driver moved and the harness entry point still points here"
    # The literal path segments the loader joins, so a rename of any one of them fails.
    for segment in ("src", "kira", "adapters", "sregym", "driver.py"):
        assert f'"{segment}"' in source


def test_the_driver_imports_only_the_standard_library() -> None:
    """It runs where this project is not installed, so a package import is a crash."""
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert "kira" not in imported
    assert imported <= set(__import__("sys").stdlib_module_names)


def test_the_driver_exposes_the_entry_point_the_script_calls() -> None:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body), (
        "the entry point calls main() and the driver no longer defines one"
    )
