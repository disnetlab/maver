"""Startup recovery for active incidents left by a dead process."""

import asyncio
import logging
from typing import Any, cast

from maver.api.deps import Service
from maver.api.incidents import _record_failure, fingerprint, run_incident_task
from maver.contracts.state import TargetStage
from maver.orchestration.recovery import safe_stage
from maver.pipelines import SelectedPipeline, restore_selection

_log = logging.getLogger(__name__)


async def recover_active(service: Service) -> None:
    """Schedule safe continuations serially; close ambiguous runs with a clear reason."""
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
            _record_failure(
                deployment,
                record.session_id,
                RuntimeError("interrupted stage needs reconciliation; automatic replay is unsafe"),
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
