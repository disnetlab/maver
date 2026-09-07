"""The driver half of the AIOpsLab adapter: everything that runs on the benchmark host.

These modules execute inside AIOpsLab's own Poetry environment, where this project is
not installed, loaded from their files by ``scripts/aiopslab-driver.py``. **They import
only the standard library**, plus ``aiopslab`` itself inside the one function that drives
its orchestrator. An import of ``kira`` here is not a layering opinion, it is a crash on
the evaluation host part-way through a sweep, so ``tests/unit/adapters/aiopslab``
enforces the constraint with an AST walk over every module in this package.

The split is not tidiness either. This half holds the cluster-admin kubeconfig and
serves the loopback bridge; the agent half runs in a container that has neither, and
reaches the cluster only through that bridge. See ``refactor/12-AIOPSLAB.md`` §2.
"""
