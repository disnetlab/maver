"""MAVER: a memory-augmented multi-agent pipeline for Kubernetes incident recovery.

MAVER accepts an alert, diagnoses the incident, plans a typed remediation, and confines
classified Kubernetes API writes to execution roles. Pod execution requires human
approval by default, including in diagnostic roles. Optional mechanisms add a verification
loop and a gated second diagnostic opinion drawn from an experience store of past incidents.

The public surface is deliberately small. See ``maver.contracts`` for the typed
artifacts that roles exchange, and ``STYLE_GUIDE.md`` for the architectural
invariants this package holds.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("maver")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
