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

from maver.audit.journal import Journal, ListSink
from maver.memory.factory import build_storage
from maver.memory.ports import SessionMemory
from maver.orchestration.runner import Deployment, incident_context, tool_context_for
from maver.settings.groups import MemorySettings, ServerSettings, VerifySettings
from maver.settings.profiles import load_profile
from maver.settings.root import Settings
from maver.tools.context import ToolContext
from maver.tools.memory import memory_search, open_memory, session_trace_summary

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


def _deployment(
    tmp_path: pathlib.Path,
    *,
    exclude: frozenset[str] = frozenset(),
    labels: dict[str, str] | None = None,
) -> Deployment:
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
                labels=dict(labels or {}) if session_id == WITHHELD else {},
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


async def _contexts(
    deployment: Deployment, withheld: tuple[str, ...] = ()
) -> AsyncIterator[ToolContext]:
    async with incident_context("sess-now", {"namespace": "shop"}, deployment) as resources:
        yield tool_context_for(
            deployment,
            resources,
            "rca",
            "sess-now",
            Journal(deployment.sink, session_id="sess-now", stage="rca"),
            withheld=withheld,
        )


async def test_the_record_is_reachable_when_nothing_is_withheld(tmp_path: pathlib.Path) -> None:
    """The control. Without it the exclusion test below passes on an unsearchable corpus."""
    async for ctx in _contexts(_deployment(tmp_path, exclude=frozenset())):
        await memory_search(ctx, query=QUERY, scope="evidence")
        assert WITHHELD in ctx.memory_aliases.values()
        served = await open_memory(ctx, session_id=WITHHELD, scope="lesson")
        assert "not found" not in str(served)


async def test_a_withheld_record_is_unreachable_by_search_or_by_identifier(
    tmp_path: pathlib.Path,
) -> None:
    async for ctx in _contexts(_deployment(tmp_path, exclude=frozenset({WITHHELD}))):
        assert ctx.memory_exclude == (WITHHELD,)

        await memory_search(ctx, query=QUERY, scope="evidence")
        assert WITHHELD not in ctx.memory_aliases.values()

        # Both take a raw identifier, so filtering search alone leaves the record one
        # well-formed call away from any role that has the id from anywhere else.
        refused = await open_memory(ctx, session_id=WITHHELD)
        assert json.loads(str(refused))["error"] == "not found"
        assert json.loads(str(await session_trace_summary(ctx, session_id=WITHHELD)))["error"] == (
            "not found"
        )


async def test_a_record_s_labels_are_not_rendered_when_a_role_opens_it(
    tmp_path: pathlib.Path,
) -> None:
    """Opening a record is the only place a whole one is rendered for a model.

    A label is the environment's own key. Under an evaluation harness it is the
    ground-truth fault label, so it must not appear in the JSON a role reads back, even
    for a record the role is perfectly entitled to open.
    """
    deployment = _deployment(tmp_path, labels={"external_key": "leak-the-answer"})
    async for ctx in _contexts(deployment):
        opened = json.loads(str(await open_memory(ctx, session_id=WITHHELD, scope="lesson")))

        assert opened.get("error") is None, "the record is readable; only its labels are not"
        assert "labels" not in opened
        assert "leak-the-answer" not in json.dumps(opened)


async def test_the_incident_s_own_set_is_unioned_with_the_process_floor(
    tmp_path: pathlib.Path,
) -> None:
    """Two sources, and neither is the other's to override.

    The floor is a deployment's own promise about the corpus; the incident's set is what
    an environment adapter withheld for this episode. A run that took one and dropped the
    other would fail open in the direction that flatters the result.
    """
    deployment = _deployment(tmp_path, exclude=frozenset({"sess-other-0"}))
    async for ctx in _contexts(deployment, withheld=(WITHHELD,)):
        assert ctx.memory_exclude == ("sess-other-0", WITHHELD)

        await memory_search(ctx, query=QUERY, scope="evidence")
        assert WITHHELD not in ctx.memory_aliases.values()
        assert "sess-other-0" not in ctx.memory_aliases.values()
        assert json.loads(str(await open_memory(ctx, session_id=WITHHELD)))["error"] == "not found"


async def test_the_records_a_role_was_served_reach_the_log_and_not_its_context(
    tmp_path: pathlib.Path,
) -> None:
    """Which precedents a role read is the question every analysis of memory starts from.

    The model is shown `memory-N` handles on purpose, so the real identifiers existed
    only in the run's alias map and went with the process. The tools return them beside
    their text instead, for the audit log alone — including on a refusal, which is
    reported to the model as a plain not-found and is otherwise indistinguishable in the
    record from a record that was never asked for.
    """
    async for ctx in _contexts(_deployment(tmp_path / "served", exclude=frozenset())):
        served = await memory_search(ctx, query=QUERY, scope="evidence")
        assert served.retrieval
        for record in served.retrieval:
            assert record["handle"] in str(served)
            assert record["record_id"] not in str(served)
            assert record["score"] is not None

    async for ctx in _contexts(_deployment(tmp_path / "withheld", exclude=frozenset({WITHHELD}))):
        refused = await open_memory(ctx, session_id=WITHHELD)
        assert refused.retrieval == ({"record_id": WITHHELD, "handle": None, "score": None},)
