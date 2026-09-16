"""The hook protocol and stack composition."""

import pytest

from maver.llm.types import ModelRequest, ModelResponse, ToolResult, Usage
from maver.middleware import (
    Correction,
    MiddlewareStack,
    ModelHandler,
    ToolHandler,
    ToolRequest,
    TurnContext,
    replace_request,
)

CTX = TurnContext(role="rca", session_id="s-1", turn=1, caller="rca")
REQUEST = ModelRequest(system="s", messages=())
RESPONSE = ModelResponse(
    text="hi",
    tool_calls=(),
    finish_reason="stop",
    reasoning=None,
    usage=Usage(),
    duration_ms=1,
    attempts=1,
)
TOOL_REQUEST = ToolRequest(
    call_id="c1", name="get_metrics", arguments={}, role="rca", session_id="s-1"
)


class _Recorder:
    """A hook that records the order it was entered and left."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def before_model(self, ctx: TurnContext, request: ModelRequest) -> ModelRequest:
        self.log.append(f"before:{self.name}")
        return replace_request(request, system=f"{request.system}|{self.name}")

    async def wrap_model_call(
        self, ctx: TurnContext, request: ModelRequest, handler: ModelHandler
    ) -> ModelResponse:
        self.log.append(f"enter:{self.name}")
        response = await handler(request)
        self.log.append(f"exit:{self.name}")
        return response

    async def wrap_tool_call(self, request: ToolRequest, handler: ToolHandler) -> ToolResult:
        self.log.append(f"tool-enter:{self.name}")
        result = await handler(request)
        self.log.append(f"tool-exit:{self.name}")
        return result


async def _model(request: ModelRequest) -> ModelResponse:
    return RESPONSE


async def _tool(request: ToolRequest) -> ToolResult:
    return ToolResult(tool_call_id=request.call_id, name=request.name, content="ok")


class TestComposition:
    async def test_wrapping_hooks_nest_outermost_first(self) -> None:
        log: list[str] = []
        stack = MiddlewareStack([_Recorder("a", log), _Recorder("b", log)])
        await stack.run_model(CTX, REQUEST, _model)
        assert log == ["enter:a", "enter:b", "exit:b", "exit:a"]

    async def test_tool_hooks_nest_the_same_way(self) -> None:
        log: list[str] = []
        stack = MiddlewareStack([_Recorder("a", log), _Recorder("b", log)])
        result = await stack.run_tool(TOOL_REQUEST, _tool)
        assert log == ["tool-enter:a", "tool-enter:b", "tool-exit:b", "tool-exit:a"]
        assert result.content == "ok"

    def test_shaping_hooks_run_in_declared_order(self) -> None:
        log: list[str] = []
        stack = MiddlewareStack([_Recorder("a", log), _Recorder("b", log)])
        shaped = stack.before_model(CTX, REQUEST)
        assert log == ["before:a", "before:b"]
        assert shaped.system == "s|a|b"

    def test_a_request_is_frozen_so_a_hook_cannot_mutate_a_shared_one(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            REQUEST.system = "mutated"  # type: ignore[misc]

    def test_with_hook_does_not_modify_the_original_stack(self) -> None:
        log: list[str] = []
        base = MiddlewareStack([_Recorder("a", log)])
        extended = base.with_hook(_Recorder("b", log))
        assert len(base.hooks) == 1
        assert len(extended.hooks) == 2

    def test_an_empty_stack_is_a_pass_through(self) -> None:
        assert MiddlewareStack().before_model(CTX, REQUEST) is REQUEST


class TestCorrections:
    def test_the_first_correction_wins(self) -> None:
        """Two hooks correcting one turn would inject two messages for one problem."""

        class _Corrects:
            def __init__(self, name: str) -> None:
                self.name = name

            def after_model(self, ctx: TurnContext, response: ModelResponse) -> Correction:
                return Correction(message=None, control=self.name, detail="d")

        stack = MiddlewareStack([_Corrects("first"), _Corrects("second")])
        correction = stack.after_model(CTX, RESPONSE)
        assert correction is not None
        assert correction.control == "first"

    def test_no_correction_when_nothing_objects(self) -> None:
        assert MiddlewareStack([]).after_model(CTX, RESPONSE) is None


class TestErrorContainment:
    def test_a_failing_shaping_hook_is_skipped_not_fatal(self) -> None:
        """A run cut short by a misbehaving accessory is worse than one that lost it."""
        seen: list[tuple[str, str]] = []

        class _Broken:
            name = "broken"

            def before_model(self, ctx: TurnContext, request: ModelRequest) -> ModelRequest:
                raise RuntimeError("boom")

        log: list[str] = []
        stack = MiddlewareStack(
            [_Broken(), _Recorder("good", log)],
            on_error=lambda hook, phase, exc: seen.append((hook, phase)),
        )
        shaped = stack.before_model(CTX, REQUEST)
        assert shaped.system == "s|good"
        assert seen == [("broken", "before_model")]

    def test_a_failing_after_model_hook_is_skipped(self) -> None:
        class _Broken:
            name = "broken"

            def after_model(self, ctx: TurnContext, response: ModelResponse) -> Correction:
                raise RuntimeError("boom")

        class _Good:
            name = "good"

            def after_model(self, ctx: TurnContext, response: ModelResponse) -> Correction:
                return Correction(message=None, control="good", detail="d")

        stack = MiddlewareStack([_Broken(), _Good()], on_error=lambda *_: None)
        correction = stack.after_model(CTX, RESPONSE)
        assert correction is not None
        assert correction.control == "good"

    async def test_a_failing_tool_hook_is_not_swallowed(self) -> None:
        """It has already taken responsibility for the call. Continuing as though it
        had not is how an authority check gets bypassed."""

        class _Broken:
            name = "gate"

            async def wrap_tool_call(
                self, request: ToolRequest, handler: ToolHandler
            ) -> ToolResult:
                raise RuntimeError("gate failed open")

        with pytest.raises(RuntimeError, match="gate failed open"):
            await MiddlewareStack([_Broken()]).run_tool(TOOL_REQUEST, _tool)


class TestPartialHooks:
    async def test_a_hook_implementing_one_shape_is_fine(self) -> None:
        class _OnlyTool:
            name = "only-tool"

            async def wrap_tool_call(
                self, request: ToolRequest, handler: ToolHandler
            ) -> ToolResult:
                result = await handler(request)
                return ToolResult(result.tool_call_id, result.name, result.content.upper())

        stack = MiddlewareStack([_OnlyTool()])
        assert stack.before_model(CTX, REQUEST) is REQUEST
        assert (await stack.run_tool(TOOL_REQUEST, _tool)).content == "OK"

    async def test_a_hook_may_refuse_a_tool_call_without_dispatching_it(self) -> None:
        """Refusal is the authority hook's whole job."""
        dispatched: list[str] = []

        async def _dispatch(request: ToolRequest) -> ToolResult:
            dispatched.append(request.name)
            return ToolResult(request.call_id, request.name, "ran")

        class _Deny:
            name = "deny"

            async def wrap_tool_call(
                self, request: ToolRequest, handler: ToolHandler
            ) -> ToolResult:
                return ToolResult(request.call_id, request.name, "error: refused by policy")

        result = await MiddlewareStack([_Deny()]).run_tool(TOOL_REQUEST, _dispatch)
        assert dispatched == []
        assert result.content.startswith("error:")


def test_the_context_is_narrow_by_design() -> None:
    """A hook that could read the loop's counters would start making the loop's decisions.

    The turn budget, the submission contract, the repeat tracker and the counter
    interlocks all span turns; a hook seeing one turn cannot implement them correctly.
    """
    fields = set(TurnContext.__slots__)
    assert fields == {"role", "session_id", "turn", "caller", "scratch", "run_state", "agent_state"}
