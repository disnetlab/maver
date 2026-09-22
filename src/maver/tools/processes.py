"""Bounded cleanup for local tool subprocesses started in their own session."""

import asyncio
import contextlib
import os
import signal


async def stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Kill a tool and its local children, then give the OS a bounded time to reap it.

    The session leader may already have exited while a child still holds its pipes.
    Its PID remains the process group ID; querying its current group would lose that
    information once the leader exits. This cannot undo an already dispatched remote
    operation, including an SSH command or Kubernetes write.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), 5.0)
