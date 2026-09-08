"""The sandbox's read grants.

One test, for the grant that a developer machine never exercises: the interpreter that
runs the sandboxed command is a virtualenv symlink, and resolving it lands on the base
install, not on the environment Python has to read its own configuration from.
"""

import sys
from pathlib import Path

from kira.tools import sandbox


def test_the_sandbox_grants_the_environment_python_starts_in(tmp_path: Path) -> None:
    roots = sandbox._runtime_read_roots(tmp_path)

    assert Path(sys.prefix).resolve() in roots
    assert Path(sys.base_prefix).resolve() in roots
