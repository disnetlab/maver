"""Optional Kafka alert intake, sharing HTTP's admission and pipeline selection."""

import asyncio
import base64
import logging
from contextlib import suppress
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from maver.api.deps import Service
from maver.api.incidents import IncidentRequest, accept_incident

try:
    from aiokafka import AIOKafkaConsumer, TopicPartition  # type: ignore[import-untyped]
except ImportError:  # optional maver[kafka] extra
    AIOKafkaConsumer = None
    TopicPartition = None

_log = logging.getLogger(__name__)


class QueueConsumer:
    """Commit an offset only after the incident has been durably admitted.

    Delivery is at least once. A durable receipt maps a redelivered record to its
    initialized session, including after a process restart.
    """

    def __init__(self, service: Service) -> None:
        """Bind the process's common admission service."""
        self.service = service
        self.consumer: Any = None
        self.task: asyncio.Task[None] | None = None
        self.error: BaseException | None = None

    async def start(self) -> None:
        """Connect to Kafka before announcing queue readiness."""
        if AIOKafkaConsumer is None or TopicPartition is None:
            raise RuntimeError("Kafka intake requires the maver[kafka] extra")

        settings = self.service.deployment.settings.intake.kafka
        self.consumer = AIOKafkaConsumer(
            settings.topic,
            bootstrap_servers=settings.bootstrap_servers,
            group_id=settings.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        )
        # A queue-only process must not announce readiness without a broker connection.
        await self.consumer.start()
        self.topic_partition = TopicPartition
        self.task = asyncio.create_task(self._run(), name="maver-kafka-intake")

    async def stop(self) -> None:
        """Stop polling before the process drains its admitted incidents."""
        if self.task is not None:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        if self.consumer is not None:
            await self.consumer.stop()

    async def _run(self) -> None:
        try:
            async for message in self.consumer:
                if self.service.runs.draining:
                    break
                partition = self.topic_partition(message.topic, message.partition)
                source = (
                    f"kafka:{self.service.deployment.settings.intake.kafka.group_id}:"
                    f"{message.topic}:{message.partition}"
                )
                key = str(message.offset)
                try:
                    body = IncidentRequest.model_validate_json(message.value)
                except ValidationError:
                    await self._quarantine(source, key, message.value, "invalid incident JSON")
                    await self.consumer.commit({partition: message.offset + 1})
                    continue
                try:
                    receipt = await accept_incident(body, self.service, intake_id=(source, key))
                except HTTPException as exc:
                    if exc.status_code not in (429, 503):
                        await self._quarantine(
                            source,
                            key,
                            message.value,
                            f"admission rejected: HTTP {exc.status_code}",
                        )
                        await self.consumer.commit({partition: message.offset + 1})
                        continue
                    # Keep the offset uncommitted and retry this record when capacity or
                    # storage returns. Advancing the iterator would skip it until restart.
                    self.consumer.seek(
                        self.topic_partition(message.topic, message.partition), message.offset
                    )
                    await asyncio.sleep(1)
                    continue
                await self.consumer.commit({partition: message.offset + 1})
                _log.info(
                    "admitted Kafka record %s:%d:%d as incident %s",
                    message.topic,
                    message.partition,
                    message.offset,
                    receipt["session_id"],
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            _log.exception("Kafka intake stopped; /readyz will report it unhealthy")

    async def _quarantine(self, source: str, key: str, raw: bytes, reason: str) -> None:
        """Write the rejected bytes durably before their offset can advance."""
        record = await asyncio.to_thread(
            self.service.deployment.storage.sessions.quarantine,
            source,
            key,
            base64.b64encode(raw).decode("ascii"),
            reason,
        )
        _log.warning(
            "quarantined Kafka record %s:%s as %s: %s", source, key, record.record_id, reason
        )
