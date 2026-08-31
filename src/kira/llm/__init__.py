"""Model provider routing and the sampling contract.

The complete sampling set is pinned and asserted at startup, never one knob at a time: a
partial pin is indistinguishable from a correct one until the results diverge.

Only ``client`` reaches the provider library; everything above this package sees the
neutral types here.
"""

from kira.llm.client import LLMClient, extract_reasoning_text, serialise_assistant
from kira.llm.errors import ErrorClass, ModelCallError, classify
from kira.llm.sampling import (
    SamplingResolution,
    assert_sampling_is_pinned,
    is_token_limit_finish,
    resolve_sampling,
    sampling_digest,
)
from kira.llm.types import (
    Completion,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
    Usage,
    serialise_tool_results,
)

__all__ = [
    "Completion",
    "ErrorClass",
    "LLMClient",
    "Message",
    "ModelCallError",
    "ModelRequest",
    "ModelResponse",
    "SamplingResolution",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    "ToolSpec",
    "Usage",
    "assert_sampling_is_pinned",
    "classify",
    "extract_reasoning_text",
    "is_token_limit_finish",
    "resolve_sampling",
    "sampling_digest",
    "serialise_assistant",
    "serialise_tool_results",
]
