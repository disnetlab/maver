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
import os
import pathlib
import subprocess
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
ENTRY = SCRIPTS / "sregym-driver.py"
DRIVER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "maver"
    / "adapters"
    / "sregym"
    / "driver.py"
)


def test_the_harness_entry_point_loads_the_driver_and_runs_it() -> None:
    """Run it, because reading it is what let the one silent failure through.

    This used to recompute the driver's path itself and then look for the segments the
    script joins. Both assertions stayed true through a change to the index the script
    counts back from, and through a move of either file — the two ways the path it
    hard-codes actually breaks. Nothing else references that path, so the place it
    breaks is the evaluation host, part-way through a campaign, recorded as an agent
    failure.

    Reaching the health check's own refusal proves the loader resolved the file and
    called ``main()``, which is the whole of the contract this file exists for.
    """
    result = subprocess.run(
        [sys.executable, str(ENTRY)],
        env={**os.environ, "HEALTH_TIMEOUT": "0", "AGENT_URL": "http://127.0.0.1:1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "agent not reachable" in result.stdout + result.stderr


def test_the_driver_imports_only_the_standard_library() -> None:
    """It runs where this project is not installed, so a package import is a crash."""
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert "maver" not in imported
    assert imported <= set(__import__("sys").stdlib_module_names)
