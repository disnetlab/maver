"""The dependency probe must not block the liveness endpoint's event loop."""

import asyncio
import threading
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from maver.api.deps import service as service_dependency
from maver.api.health import router


async def test_slow_store_readiness_does_not_block_liveness() -> None:
    entered = threading.Event()
    release = threading.Event()

    class SlowIndex:
        def count(self) -> int:
            entered.set()
            assert release.wait(timeout=5)
            return 0

    service = SimpleNamespace(
        deployment=SimpleNamespace(
            settings=SimpleNamespace(
                authority=SimpleNamespace(mode="auto"),
                intake=SimpleNamespace(kafka=SimpleNamespace(enabled=False)),
            ),
            profile=SimpleNamespace(name="test"),
            storage=SimpleNamespace(index=SlowIndex()),
        ),
        runs=SimpleNamespace(draining=False),
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[service_dependency] = lambda: service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://maver"
    ) as http:
        readiness = asyncio.create_task(http.get("/readyz"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            liveness = await asyncio.wait_for(http.get("/healthz"), timeout=1)
            assert liveness.status_code == 200
            assert liveness.json()["status"] == "ok"
        finally:
            release.set()
        assert (await readiness).status_code == 200
