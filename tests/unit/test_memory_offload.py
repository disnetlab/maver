"""The memory tools must not run their store on the event loop.

The ports are synchronous and the deployment behind them is a database on another host,
so a call made inline holds the loop for as long as that database takes. Nothing above
can cut it short: a timeout is a coroutine, and a coroutine cannot run while the loop is
inside a synchronous call. Reproduced against a live PostgreSQL with the record table
locked — the search took the loop down, an eight-second ``asyncio.wait_for`` around it
never fired, and the process had to be killed.

Every call the store below makes asks the loop to run something and waits for the answer,
which is a thing only another thread can do. On the loop's own thread the answer can
never come, so the call fails on its own clock instead of hanging the suite.
"""

import asyncio
from typing import Any

from maver.audit.journal import Journal, ListSink
from maver.memory.ports import SessionRecord
from maver.tools.context import ToolContext
from maver.tools.memory import memory_search, open_memory, session_trace_summary

STALLED = 2.0
"""Seconds to wait for the loop before calling it stalled. Paid only when it is."""


async def _anything() -> None:
    """Work for the loop to do, whose completion is the proof that it is free."""


class _NeedsTheLoop:
    """A store whose every call needs the event loop to run something for it."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _turn(self) -> None:
        pending = asyncio.run_coroutine_threadsafe(_anything(), self._loop)
        try:
            pending.result(timeout=STALLED)
        except TimeoutError:
            raise AssertionError(
                "the store was called on the event loop, which then stalled"
            ) from None

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        self._turn()
        return []

    def open(self, session_id: str) -> None:
        self._turn()

    def get_session(self, session_id: str) -> SessionRecord:
        self._turn()
        return SessionRecord(session_id=session_id, started_at="2026-01-01T00:00:00+00:00")

    def runs_for(self, session_id: str) -> list[Any]:
        self._turn()
        return []


async def test_no_memory_tool_holds_the_loop_while_its_store_works() -> None:
    store = _NeedsTheLoop(asyncio.get_running_loop())
    ctx = ToolContext(
        transport=None,  # type: ignore[arg-type]
        workspace=None,  # type: ignore[arg-type]
        role="rca",
        session_id="sess-now",
        journal=Journal(ListSink(), session_id="sess-now", stage="rca"),
        memory=store,  # type: ignore[arg-type]
        sessions=store,  # type: ignore[arg-type]
        runs=store,  # type: ignore[arg-type]
    )

    await memory_search(ctx, query="readiness probe failing on payment-service")
    await open_memory(ctx, session_id="memory-1")
    # Two store calls, the session and then its runs, so both hops are exercised.
    await session_trace_summary(ctx, session_id="memory-1")
