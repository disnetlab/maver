"""Where the prototype text these checks compare against is read from.

The prototype is a private research tree that will not be published beside this
repository, so every file the parity checks read is frozen under
``tests/fixtures/prototype`` with the layout it has there. The frozen copy is the
default on purpose: on a clone with no prototype on disk the comparisons still run,
rather than skipping and reporting a green suite that checked nothing.

Setting ``MAVER_PROTOTYPE_ROOT`` to a prototype checkout compares against that instead.
That is the author's cross-check — it is how the frozen copy is shown to still match
its source, and how a refreeze is verified — not the path a reviewer runs.

The name is scrubbed from the environment by the ``clean_env`` fixture like every other
``MAVER_``-prefixed variable, so it is read here at import time, before any test runs.
"""

import os
from pathlib import Path

ENV_NAME = "MAVER_PROTOTYPE_ROOT"

FROZEN = Path(__file__).resolve().parents[1] / "fixtures" / "prototype"
"""The frozen copy: exactly the files the parity checks read, and nothing besides."""

MANIFEST = FROZEN / "MANIFEST"
"""Which prototype commit the copy was taken from, and how to take it again."""


def prototype_root() -> Path:
    """Resolve the tree the parity checks read.

    Returns:
        The checkout named by ``MAVER_PROTOTYPE_ROOT`` when it exists, and the frozen
        copy otherwise. A name that points nowhere falls back rather than failing: a
        stale export in a shell should not turn the comparisons off.
    """
    named = os.environ.get(ENV_NAME)
    if named and Path(named).is_dir():
        return Path(named)
    return FROZEN
