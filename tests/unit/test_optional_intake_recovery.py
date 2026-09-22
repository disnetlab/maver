"""The two opt-in boundaries: queue acknowledgement and safe restart replay."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from maver.api import incidents as incidents_module
from maver.api import queue as queue_module
from maver.api.queue import QueueConsumer
from maver.contracts.runtime import RunState
from maver.memory.ports import SessionRecord
from maver.orchestration import recovery as recovery_module
from maver.orchestration.runner import _checkpointed_node


class FakeConsumer:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.commits = 0
        self.seeks: list[tuple[Any, int]] = []

    def __aiter__(self):
        async def records():
            for message in self.messages:
                yield message

        return records()

    async def commit(self, offsets: dict[Any, int]) -> None:
        assert offsets == {("alerts", 0): 8}
        self.commits += 1

    def seek(self, partition: Any, offset: int) -> None:
        self.seeks.append((partition, offset))


def message(value: bytes = b'{"alert":{"alertname":"A"}}') -> Any:
    return SimpleNamespace(value=value, topic="alerts", partition=0, offset=7)


@pytest.mark.asyncio
async def test_queue_commits_after_durable_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    consumer = FakeConsumer([message()])
    service = SimpleNamespace(
        runs=SimpleNamespace(draining=False),
        deployment=SimpleNamespace(
            settings=SimpleNamespace(intake=SimpleNamespace(kafka=SimpleNamespace(group_id="test")))
        ),
    )
    queue = QueueConsumer(service)
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)

    async def admit(body, service, *, intake_id):
        assert intake_id == ("kafka:test:alerts:0", "7")
        assert consumer.commits == 0
        return {"session_id": "s1"}

    monkeypatch.setattr(queue_module, "accept_incident", admit)
    await queue._run()
    assert queue.error is None
    assert consumer.commits == 1


@pytest.mark.asyncio
async def test_queue_retries_capacity_without_acknowledging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = FakeConsumer([message(), message()])
    queue = QueueConsumer(
        SimpleNamespace(
            runs=SimpleNamespace(draining=False),
            deployment=SimpleNamespace(
                settings=SimpleNamespace(
                    intake=SimpleNamespace(kafka=SimpleNamespace(group_id="test"))
                )
            ),
        )
    )
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)
    attempts = 0

    async def admit(body, service, *, intake_id):
        assert intake_id == ("kafka:test:alerts:0", "7")
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPException(status_code=429)
        return {"session_id": "s1"}

    async def no_delay(seconds):
        pass

    monkeypatch.setattr(queue_module, "accept_incident", admit)
    monkeypatch.setattr(queue_module.asyncio, "sleep", no_delay)
    await queue._run()
    assert consumer.seeks == [(("alerts", 0), 7)]
    assert consumer.commits == 1


@pytest.mark.asyncio
async def test_queue_retries_pending_reconciliation_without_acknowledging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = FakeConsumer([message(), message()])
    queue = QueueConsumer(service())
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)
    attempts = 0

    async def admit(body, service, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPException(status_code=409, headers={"Retry-After": "30"})
        return {"session_id": "s1"}

    async def no_delay(seconds):
        pass

    monkeypatch.setattr(queue_module, "accept_incident", admit)
    monkeypatch.setattr(queue_module.asyncio, "sleep", no_delay)
    await queue._run()
    assert queue.error is None
    assert consumer.seeks == [(("alerts", 0), 7)]
    assert consumer.commits == 1


@pytest.mark.asyncio
async def test_queue_quarantines_poison_record_before_acknowledging() -> None:
    consumer = FakeConsumer([message(b"not JSON")])
    saved = []

    def quarantine(source, key, raw_base64, reason):
        assert consumer.commits == 0
        saved.append((source, key, raw_base64, reason))
        return SimpleNamespace(record_id="q1")

    queue = QueueConsumer(
        SimpleNamespace(
            runs=SimpleNamespace(draining=False),
            deployment=SimpleNamespace(
                settings=SimpleNamespace(
                    intake=SimpleNamespace(kafka=SimpleNamespace(group_id="test"))
                ),
                storage=SimpleNamespace(sessions=SimpleNamespace(quarantine=quarantine)),
            ),
        )
    )
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)
    await queue._run()
    assert queue.error is None
    assert consumer.commits == 1
    assert saved == [("kafka:test:alerts:0", "7", "bm90IEpTT04=", "invalid incident JSON")]


@pytest.mark.asyncio
async def test_queue_does_not_acknowledge_when_quarantine_write_fails() -> None:
    consumer = FakeConsumer([message(b"not JSON")])

    def fail(*args):
        raise OSError("storage unavailable")

    queue = QueueConsumer(
        SimpleNamespace(
            runs=SimpleNamespace(draining=False),
            deployment=SimpleNamespace(
                settings=SimpleNamespace(
                    intake=SimpleNamespace(kafka=SimpleNamespace(group_id="test"))
                ),
                storage=SimpleNamespace(sessions=SimpleNamespace(quarantine=fail)),
            ),
        )
    )
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)
    await queue._run()
    assert isinstance(queue.error, OSError)
    assert consumer.commits == 0


@pytest.mark.asyncio
async def test_queue_quarantines_permanent_rejection_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = FakeConsumer([message(), message()])
    saved = []
    attempts = 0

    def quarantine(source, key, raw_base64, reason):
        saved.append(reason)
        return SimpleNamespace(record_id="q1")

    async def admit(body, service, *, intake_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPException(status_code=422, detail="invalid pipeline")
        return {"session_id": "s1"}

    queue = QueueConsumer(
        SimpleNamespace(
            runs=SimpleNamespace(draining=False),
            deployment=SimpleNamespace(
                settings=SimpleNamespace(
                    intake=SimpleNamespace(kafka=SimpleNamespace(group_id="test"))
                ),
                storage=SimpleNamespace(sessions=SimpleNamespace(quarantine=quarantine)),
            ),
        )
    )
    queue.consumer = consumer
    queue.topic_partition = lambda topic, partition: (topic, partition)
    monkeypatch.setattr(queue_module, "accept_incident", admit)
    await queue._run()
    assert queue.error is None
    assert saved == ["admission rejected: HTTP 422"]
    assert consumer.commits == 2
    assert attempts == 2


def test_recovery_refuses_write_stage_and_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    roles = {"rca": "rca", "execution": "execution"}
    pipeline = SimpleNamespace(
        workflow=SimpleNamespace(stage=lambda name: SimpleNamespace(role=roles[name])),
        agents={
            "rca": SimpleNamespace(tool_names=("get_metrics", "bash_exec", "submit_root_cause")),
            "execution": SimpleNamespace(tool_names=("exec_kubectl_cmd_safely",)),
        },
    )
    monkeypatch.setattr(
        recovery_module,
        "restore_selection",
        lambda catalog, manifest: SimpleNamespace(pipeline=pipeline),
    )
    deployment = SimpleNamespace(
        runtime_digest="unchanged",
        adapter=None,
        tool_extras=(),
        pipelines=object(),
        settings=SimpleNamespace(authority=SimpleNamespace(additional_write_stages=frozenset())),
    )

    def record(stage: str, adapter: str = "") -> SessionRecord:
        return SessionRecord(
            session_id="s1",
            started_at="now",
            checkpoint={
                "kind": "active_stage",
                "version": 1,
                "stage": stage,
                "adapter": adapter,
                "state": {
                    "session_id": "s1",
                    "runtime": {"deployment_digest": "unchanged", "selection": {"name": "default"}},
                },
            },
        )

    assert recovery_module.safe_stage(record("rca"), deployment) == "rca"
    assert recovery_module.safe_stage(record("execution"), deployment) is None
    assert recovery_module.safe_stage(record("rca", "sregym"), deployment) is None
    deployment.settings.authority.additional_write_stages = frozenset({"rca"})
    assert recovery_module.safe_stage(record("rca"), deployment) is None
    deployment.settings.authority.additional_write_stages = frozenset()
    pipeline.agents["rca"].tool_names += ("custom_command",)
    assert recovery_module.safe_stage(record("rca"), deployment) is None


@pytest.mark.asyncio
async def test_stage_checkpoint_is_written_before_role_runs() -> None:
    saved: list[tuple] = []
    sessions = SimpleNamespace(set_run_status=lambda *args, **kwargs: saved.append((args, kwargs)))
    deployment = SimpleNamespace(storage=SimpleNamespace(sessions=sessions), adapter=None)

    async def node(state):
        assert saved and saved[0][1]["checkpoint"]["stage"] == "rca"
        return {"root_cause": None}

    result = await _checkpointed_node("rca", node, deployment)(
        {"session_id": "s1", "runtime": RunState(), "alert": {"alertname": "A"}}
    )
    assert result == {"root_cause": None}


@pytest.mark.asyncio
async def test_draining_preserves_an_opt_in_recovery_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled: list[str] = []

    async def running(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(incidents_module, "run_incident", running)
    monkeypatch.setattr(
        incidents_module,
        "record_cancellation",
        lambda deployment, session: cancelled.append(session),
    )
    service = SimpleNamespace(
        deployment=SimpleNamespace(
            settings=SimpleNamespace(server=SimpleNamespace(resume_interrupted_incidents=True))
        ),
        runs=SimpleNamespace(draining=True, unregister=lambda session: None),
    )
    task = asyncio.create_task(
        incidents_module.run_incident_task(
            session_id="s1", alert={}, target_stage="diagnose", service=service
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == []
