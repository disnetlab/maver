#!/usr/bin/env python3
"""The path the harness's own agent configuration names, kept working.

The driver itself lives in the package. This loads it straight from its file rather than
importing it as a module, because it runs in the harness's interpreter, where this
project is not installed and its dependencies are absent — importing by package name
would execute the adapter's ``__init__`` and fail before the driver ever ran. The driver
uses only the standard library, so loading the one file is enough.
"""

import importlib.util
import pathlib
import sys

_DRIVER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "maver"
    / "adapters"
    / "sregym"
    / "driver.py"
)


def _load() -> object:
    """Load the driver module from its file."""
    spec = importlib.util.spec_from_file_location("maver_sregym_driver", _DRIVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the episode driver from {_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    sys.exit(_load().main())
