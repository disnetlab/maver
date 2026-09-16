"""Cross-cutting concerns as composable hooks around the model/tool loop.

The hook vocabulary mirrors LangChain 1.x middleware on purpose, so the design is
legible to anyone who knows that framework. The pattern is adopted; the API is not,
because this package owns its own graph execution.

See STYLE_GUIDE.md section 5 for the rule on what belongs here and what does not.
"""

from maver.middleware.base import (
    AgentHook,
    Correction,
    MiddlewareStack,
    ModelHandler,
    ToolHandler,
    ToolRequest,
    TurnContext,
    replace_request,
)

__all__ = [
    "AgentHook",
    "Correction",
    "MiddlewareStack",
    "ModelHandler",
    "ToolHandler",
    "ToolRequest",
    "TurnContext",
    "replace_request",
]
