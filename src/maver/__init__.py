"""MAVER: a memory-augmented multi-agent pipeline for Kubernetes incident recovery.

MAVER accepts an alert, diagnoses the incident with a read-only root-cause analysis
role, plans a typed remediation, and executes it through the only role permitted to
write to the cluster. Optional mechanisms add an independent verification loop and a
gated second diagnostic opinion drawn from an experience store of past incidents.

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
