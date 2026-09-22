"""Cancelled tools must not leave a local command running behind the incident."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from maver.settings.groups import TransportSettings
from maver.tools import execution, host, kubectl_proxy
from maver.tools.processes import stop_process_group
from maver.tools.transport.direct import DirectTransport


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["direct", "proxy", "shell", "ssh"])
async def test_cancellation_reaps_the_tool_process(
    route: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = asyncio.create_subprocess_exec
    processes = []
    ready = asyncio.Event()

    async def local_process(*args, **kwargs):
        # No SSH or cluster access: keep each real call site's subprocess options.
        process = await spawn(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        processes.append(process)
        ready.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", local_process)
    if route == "direct":
        work = DirectTransport(TransportSettings(kubectl_binary=sys.executable)).kubectl([])
    elif route == "proxy":
        work = kubectl_proxy.run_via_proxy(
            [],
            proxy_url="http://127.0.0.1:1",
            kubeconfig_dir=tmp_path,
            binary=sys.executable,
            timeout_s=60,
        )
    elif route == "shell":
        ctx = SimpleNamespace(
            workspace=SimpleNamespace(path=tmp_path), settings=SimpleNamespace(fs_sandbox=False)
        )
        monkeypatch.setattr(execution, "subprocess_env", lambda ctx: dict(os.environ))
        work = execution._run(ctx, ["bash"], timeout=60, label="shell", killed="command")
    else:
        ctx = SimpleNamespace(
            role="execution", settings=SimpleNamespace(host_recovery_nodes=frozenset({"local"}))
        )
        work = host.ssh_exec(ctx, node="local", command="df")
    task = asyncio.create_task(work)
    try:
        await asyncio.wait_for(ready.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None
    finally:
        # A regression must not leave its intentionally long-lived test child behind.
        for process in processes:
            if process.returncode is None:
                process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_cleanup_reaches_children_after_the_group_leader_exits() -> None:
    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 30 & exit 0",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(3):
            while process.returncode is None:
                await asyncio.sleep(0.01)
            await stop_process_group(process)
            await process.communicate()
    finally:
        await stop_process_group(process)
