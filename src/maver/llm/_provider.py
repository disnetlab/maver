"""The single import site for the provider client library.

The library resolves its cost-map setting when it is imported, so the default has to be
established before the import rather than by a caller afterwards. Keeping both lines in
one module means the process has exactly one import-time environment write, in a file
whose only job is to explain it.

Fetching the remote cost map is an outbound request from inside the operator's trust
boundary, which is the boundary this system exists to respect, so it is opt-in.
"""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

import litellm

__all__ = ["litellm"]
