"""The path an AIOpsLab host names, and the constraint that makes the whole half work.

`scripts/aiopslab-driver.py` is what a run script on the benchmark host invokes. It
loads the driver from its files rather than importing it by package name, because it
runs in AIOpsLab's Poetry interpreter where this project is not installed. That makes it
a hard-coded path into the package, and a hard-coded path is exactly what a refactor
moves without noticing: nothing else references it, so it breaks silently and the place
it breaks is a benchmark host part-way through a sweep.

The standard-library-only rule is the other half of the same contract, and it is just as
invisible: one import of `maver`, or of `httpx`, would pass every test here and fail only
where it matters.
"""

import ast
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
ENTRY = ROOT / "scripts" / "aiopslab-driver.py"
HOST = ROOT / "src" / "maver" / "adapters" / "aiopslab" / "host"

ALLOWED_THIRD_PARTY = {"aiopslab"}
"""The harness's own package, which is by definition importable where this runs, and is
imported inside the one function that drives it."""


def test_the_entry_point_loads_the_driver_and_runs_it(tmp_path: pathlib.Path) -> None:
    """Run it, because reading it cannot show that the path it computes still resolves.

    Started from an unrelated working directory, so a path accidentally rebuilt from the
    process's cwd rather than from the script's own location fails here rather than on
    the host. Reaching the health check's refusal proves the loader resolved four
    modules that import each other by relative name and called `main()`.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "run",
            "--problem",
            "misconfig-detection-1",
            "--agent-url",
            "http://127.0.0.1:1",
            "--bridge-token",
            "0123456789abcdef",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "agent not reachable" in result.stdout + result.stderr


def test_the_doctor_reports_every_precondition_separately(tmp_path: pathlib.Path) -> None:
    """One line per check: the first failure must not hide the other three."""
    result = subprocess.run(
        [sys.executable, str(ENTRY), "doctor", "--agent-url", "http://127.0.0.1:1"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "AIOPSLAB_BRIDGE_TOKEN": "0123456789abcdef"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 1
    assert "[FAIL] agent:" in result.stdout
    assert "[ok] bridge token: present" in result.stdout
    assert "aiopslab:" in result.stdout
    assert "kubectl:" in result.stdout


def test_the_driver_half_imports_only_the_standard_library() -> None:
    """It runs where this project is not installed, so a package import is a crash."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(HOST.glob("*.py")):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        outside = imported - sys.stdlib_module_names - ALLOWED_THIRD_PARTY
        if outside:
            offenders[path.name] = outside

    assert offenders == {}
    # Not vacuous: a moved package would leave nothing to walk and pass silently.
    assert {"mapping.py", "bridge.py", "lifecycle.py", "main.py"} <= {
        path.name for path in HOST.glob("*.py")
    }


def test_the_entry_point_script_imports_only_the_standard_library() -> None:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(ENTRY.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= sys.stdlib_module_names
