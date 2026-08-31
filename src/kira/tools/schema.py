"""The two schema transforms applied between a tool's declaration and the model.

Both exist because of something a provider does. Keeping them here, in one module, is
deliberate: the rationale argument is injected into a schema in one place and stripped
before dispatch in another, and separating those two halves is a silent, total failure.
Every dispatch would be called with an unexpected keyword, the loop's catch-all would
turn each ``TypeError`` into an ``error:`` string, and the run would degrade to a 100%
tool-failure rate with no exception raised anywhere.
"""

from collections.abc import Mapping
from typing import Any

__all__ = [
    "RATIONALE_ARG",
    "RATIONALE_PROPERTY",
    "normalise_schema",
    "strip_rationale",
    "with_rationale",
]

RATIONALE_ARG = "rationale"

# Every non-submit tool gets a required `rationale` argument injected into its schema.
# The model writes one short line of working state ("leading hypothesis / what's
# confirmed / what this call tests") that rides INSIDE the tool call. Unlike a
# visible-content status line — which reasoning models keep in their hidden reasoning
# channel and the loop then discards — tool-call arguments are always echoed back into
# the conversation, so this trace genuinely survives across turns. Submit tools are
# excluded: they validate against a strict model and already carry their own summary.
RATIONALE_PROPERTY: Mapping[str, str] = {
    "type": "string",
    "description": (
        "REQUIRED. One short sentence of your current working state BEFORE this call: "
        "your leading hypothesis, what you have established or ruled out, and what this "
        "call is meant to test. Your hidden reasoning is NOT retained between turns — "
        "this argument is the trace future turns see. One sentence; do not restate full "
        "reasoning or paste tool output."
    ),
}


def with_rationale(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Prepend a required ``rationale`` property to a tool's parameters.

    Ordering matters: the property is placed first because that is the order the model
    reads the schema in, and the argument is meant to be written before the call is
    composed rather than filled in afterwards.

    Args:
        parameters: The tool's declared JSON Schema.

    Returns:
        A new schema. The input is never mutated, since registry specs share it.
    """
    properties = parameters.get("properties") or {}
    if RATIONALE_ARG in properties:
        return dict(parameters)
    return {
        **parameters,
        "type": parameters.get("type", "object"),
        "properties": {RATIONALE_ARG: dict(RATIONALE_PROPERTY), **properties},
        "required": [RATIONALE_ARG, *(parameters.get("required") or [])],
    }


def strip_rationale(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the rationale before dispatch.

    It stays in the recorded call and in the transcript, so the trace survives to the
    next turn; it is simply not a parameter any tool accepts.

    Args:
        arguments: The model's arguments.

    Returns:
        The arguments a dispatch function can be called with.
    """
    return {k: v for k, v in arguments.items() if k != RATIONALE_ARG}


def normalise_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Make a model-derived JSON Schema safe across function-calling providers.

    Gemini's validator returns INTERNAL 500 on schemas carrying ``$ref``/``$defs`` and
    on the ``anyOf`` wrapper Pydantic emits for a nullable optional field. Refs are
    inlined and a nullable ``anyOf`` is collapsed to the bare type. Optional fields stay
    optional because they are already absent from ``required``, so the model can simply
    omit them. OpenAI and Anthropic accept the simpler shape unchanged.

    Args:
        schema: The generated schema.

    Returns:
        A schema every supported provider accepts.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref[len("#/$defs/") :])
            if isinstance(target, dict):
                merged = walk(target)
                for key, value in node.items():
                    if key != "$ref":
                        merged[key] = walk(value)
                return merged

        branches = node.get("anyOf")
        if isinstance(branches, list):
            non_null = [
                b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            # Only collapse a genuine nullable wrapper: exactly one branch left, and a
            # null branch actually removed. A real two-type union is left alone.
            if len(non_null) == 1 and len(non_null) != len(branches):
                merged = walk(non_null[0])
                for key, value in node.items():
                    if key == "anyOf":
                        continue
                    if key == "default" and value is None:
                        continue  # implied by the field being optional
                    merged[key] = value if key == "default" else walk(value)
                return merged

        return {k: walk(v) for k, v in node.items()}

    out = walk(dict(schema))
    if not isinstance(out, dict):  # a mapping in always walks to a mapping out
        raise TypeError(f"schema normalisation produced {type(out).__name__}")
    out.pop("$defs", None)
    result: dict[str, Any] = out
    return result
