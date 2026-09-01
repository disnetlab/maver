"""A withheld record is unreachable by every path a role has, not merely unsearchable.

The exclusion set is what keeps a generalisation arm honest: it withholds the evaluated
incident's own record so a run cannot retrieve the answer it is being scored on. That
makes it a measurement instrument, and an instrument that is off by default reports a
number nobody can use. It reached the gated second pass and nothing else, and both of
the tools that take an identifier accept a raw one, so there were three ways through it.

The corpus is a dozen near neighbours rather than a handful, because the retriever ranks
only what the text prefilter hands it: on three documents every term is common, no hit
clears the relevance floor, and an exclusion test would pass without excluding anything.
"""

import json
import pathlib
from collections.abc import AsyncIterator

from kira.audit.journal import Journal, ListSink
from kira.memory.factory import build_storage
from kira.memory.ports import SessionMemory
from kira.orchestration.runner import Deployment, incident_context, tool_context_for
from kira.settings.groups import MemorySettings, ServerSettings, VerifySettings
from kira.settings.profiles import load_profile
from kira.settings.root import Settings
from kira.tools.context import ToolContext
from kira.tools.memory import memory_search, open_memory, session_trace_summary

WITHHELD = "sess-withheld"
QUERY = "probe payment-service 8080 listens 9090"
NEIGHBOURS = (
    "readiness probe failing on cart-service, container never became ready",
    "liveness probe restarts on search-service after a slow start",
    "readiness probe on user-service flapping during the rollout",
    "probe failing on inventory-service, the container crashed on boot",
    "readiness probe timeout on shipping-service under load",
    "liveness probe on email-service killed the container repeatedly",
    "probe misconfigured on auth-service, wrong scheme",
    "readiness probe on catalog-service never passed after the upgrade",
    "probe failing on session-service, dependency unavailable",
    "readiness probe on rating-service failed during the deploy",
    "probe on recommendation-service returned a redirect",
)
ANSWER = (
    "readiness probe on payment-service returns connection refused on port 8080 "
    "because the container listens on 9090"
)


def _deployment(tmp_path: pathlib.Path, *, exclude: frozenset[str]) -> Deployment:
    settings = Settings(
        server=ServerSettings(workdir=tmp_path),
        memory=MemorySettings(enabled=True, learning_enabled=False, exclude_record_ids=exclude),
        verify=VerifySettings(enabled=False),
    )
    storage = build_storage(settings)
    corpus = {f"sess-other-{i}": text for i, text in enumerate(NEIGHBOURS)}
    corpus[WITHHELD] = ANSWER
    for session_id, text in corpus.items():
        storage.index.upsert_summary(
            SessionMemory(
                session_id=session_id,
                evidence_text=text,
                lesson_text=text,
                incident_summary=text,
                lifecycle_state="human_verified",
                fault_category="misconfiguration",
                alert_name="ProbeFailure",
                namespace="shop",
                affected_resource="shop/Deployment/payment",
            )
        )
    return Deployment(
        settings=settings,
        storage=storage,
        completion=None,
        profile=load_profile("qwen3.6-35b-a3b", pathlib.Path("configs/models")),
        transport_for=lambda session_id: None,
        sink=ListSink(),
    )


async def _contexts(deployment: Deployment) -> AsyncIterator[ToolContext]:
    async with incident_context("sess-now", {"namespace": "shop"}, deployment) as resources:
        yield tool_context_for(
            deployment,
            resources,
            "rca",
            "sess-now",
            Journal(deployment.sink, session_id="sess-now", stage="rca"),
        )


async def test_the_record_is_reachable_when_nothing_is_withheld(tmp_path: pathlib.Path) -> None:
    """The control. Without it the exclusion test below passes on an unsearchable corpus."""
    async for ctx in _contexts(_deployment(tmp_path, exclude=frozenset())):
        await memory_search(ctx, query=QUERY, scope="evidence")
        assert WITHHELD in ctx.memory_aliases.values()
        assert "not found" not in await open_memory(ctx, session_id=WITHHELD, scope="lesson")


async def test_a_withheld_record_is_unreachable_by_search_or_by_identifier(
    tmp_path: pathlib.Path,
) -> None:
    async for ctx in _contexts(_deployment(tmp_path, exclude=frozenset({WITHHELD}))):
        assert ctx.memory_exclude == (WITHHELD,)

        await memory_search(ctx, query=QUERY, scope="evidence")
        assert WITHHELD not in ctx.memory_aliases.values()

        # Both take a raw identifier, so filtering search alone leaves the record one
        # well-formed call away from any role that has the id from anywhere else.
        assert json.loads(await open_memory(ctx, session_id=WITHHELD))["error"] == "not found"
        assert (
            json.loads(await session_trace_summary(ctx, session_id=WITHHELD))["error"]
            == "not found"
        )
