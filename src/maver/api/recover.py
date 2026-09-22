"""Startup recovery for active incidents left by a dead process."""

import asyncio
import logging
from typing import Any, cast

from maver.api.deps import Service
from maver.api.incidents import _record_failure, fingerprint, run_incident_task
from maver.contracts.lifecycle import RunStatus
from maver.contracts.state import TargetStage
from maver.orchestration.jobs import record_cancellation
from maver.orchestration.recovery import safe_stage
from maver.pipelines import SelectedPipeline, restore_selection

_log = logging.getLogger(__name__)


async def recover_active(service: Service) -> None:
    """Schedule safe continuations; park ambiguous runs for explicit reconciliation."""
    deployment = service.deployment
    records = await asyncio.to_thread(deployment.storage.sessions.running_sessions)
    serial = asyncio.Semaphore(1)
    for record in records:
        stage = safe_stage(record, deployment)
        if stage is None:
            _log.error(
                "incident %s needs reconciliation; its interrupted stage is unsafe to replay",
                record.session_id,
            )
            await asyncio.to_thread(
                deployment.storage.sessions.set_run_status,
                record.session_id,
                RunStatus.NEEDS_RECONCILIATION,
                current_stage=record.current_stage,
                checkpoint=record.checkpoint,
            )
            continue
        checkpoint = record.checkpoint
        assert checkpoint is not None and deployment.pipelines is not None
        state = checkpoint["state"]
        runtime = state["runtime"]
        selection = restore_selection(deployment.pipelines, runtime["selection"])
        target = cast(TargetStage, state["target_stage"])
        service.runs.register(record.session_id, fingerprint=fingerprint(record.alert))

        async def continue_one(
            session_id: str = record.session_id,
            alert: dict[str, Any] = record.alert,
            target_stage: TargetStage = target,
            selected: SelectedPipeline = selection,
            saved: dict[str, Any] = checkpoint,
        ) -> None:
            async with serial:
                await run_incident_task(
                    session_id=session_id,
                    alert=alert,
                    target_stage=target_stage,
                    service=service,
                    selection=selected,
                    recovery=saved,
                )

        task = asyncio.create_task(continue_one(), name=f"maver-recover-{record.session_id}")
        service.runs.attach(record.session_id, task)

        def settle_recovery(
            finished: asyncio.Task[None], session_id: str = record.session_id
        ) -> None:
            # A task cancelled before its first instruction, or while waiting for the
            # serial slot, never enters run_incident_task's own finally block.
            if session_id not in service.runs.live():
                return
            try:
                if finished.cancelled():
                    if not (
                        service.runs.draining
                        and deployment.settings.server.resume_interrupted_incidents
                    ):
                        record_cancellation(deployment, session_id)
                elif error := finished.exception():
                    _record_failure(deployment, session_id, error)
            finally:
                service.runs.unregister(session_id)

        task.add_done_callback(settle_recovery)
