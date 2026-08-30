"""Cross-cutting concerns as composable hooks around the model/tool loop.

The hook vocabulary mirrors LangChain 1.x middleware on purpose, so the design is
legible to anyone who knows that framework. The pattern is adopted; the API is not,
because this package owns its own graph execution.
"""
