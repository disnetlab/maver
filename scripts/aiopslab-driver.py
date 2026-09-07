#!/usr/bin/env python3
"""The path an AIOpsLab host runs, kept working.

The driver lives in the package. This loads it straight from its files rather than
importing it as a module, because it runs in AIOpsLab's own interpreter, where this
project is not installed and its dependencies are absent — importing by package name
would execute the adapter's ``__init__`` and fail before the driver ever ran. The driver
half uses only the standard library, so loading its directory is enough.

It is loaded as a package rather than as one file, because the driver is four modules
that import each other by relative name, and a relative import needs a parent package in
``sys.modules`` to resolve against.
"""

import importlib
import importlib.util
import pathlib
import sys
import types

_HOST = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "kira" / "adapters" / "aiopslab" / "host"
)
_PACKAGE = "kira_aiopslab_host"


def _load() -> types.ModuleType:
    """Load the driver package from its directory and return its command line."""
    spec = importlib.util.spec_from_file_location(
        _PACKAGE, _HOST / "__init__.py", submodule_search_locations=[str(_HOST)]
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the AIOpsLab driver from {_HOST}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{_PACKAGE}.main")


if __name__ == "__main__":
    sys.exit(_load().main())
