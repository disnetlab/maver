"""The decision surface: what is waiting on a person, and what they said.

Three routes, carried from the prototype with their behaviour intact, including
``status=all`` and ``status=`` meaning every status, and a decide route that answers
**404** when the id is unknown *or* already decided. A late second decision is a
duplicate, not a new one, and the store's compare-and-set is what makes saying so cheap.

The payload is wider than the prototype's. It gained ``action``, ``selectors``,
``targets``, ``rule`` and ``command``, which are what make a request reviewable: the
prototype summarised a write as its first six argument tokens, so a patch showed the
resource and never the patch body.

Every store call here goes to a thread. There is one event loop in this process: an
incident runs as a task created inside a request handler, so a store call made inline by
a route does not hold "the API" apart from the work — it holds the incidents too, and
the readiness probe with them. The store is a database on another host and the port is
synchronous, so that wait is the database's to end and no timeout written here could cut
it short.
"""

import asyncio
import logging
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from maver.api.deps import ServiceDep
from maver.memory.ports import ApprovalRecord, ApprovalStatus

__all__ = ["DecideRequest", "request_payload", "router"]

_log = logging.getLogger(__name__)

router = APIRouter()

StatusQuery = Literal[
    "pending", "approved", "denied", "answered", "timed_out", "cancelled", "all", ""
]
"""Every status, plus the two spellings of "do not filter" the prototype accepted."""


class DecideRequest(BaseModel):
    """One operator's answer."""

    approve: bool
    reason: str = ""
    decided_by: str = "generic human"


def request_payload(record: ApprovalRecord) -> dict[str, Any]:
    """Render one request for a reader.

    Shared with the command line so that ``maver approvals show`` and this route cannot
    describe the same request differently.

    Args:
        record: The stored request.

    Returns:
        Every field the prototype exposed, plus the ones that make the call reviewable.
    """
    payload = dict(record.payload or {})
    return {
        "request_id": record.request_id,
        "session_id": record.session_id,
        "kind": record.kind,
        "stage": record.stage,
        "tool_call_id": record.tool_call_id,
        "tool_name": record.tool_name,
        "status": record.status,
        "summary": record.summary,
        "prompt": record.prompt,
        "payload": payload,
        "response": record.response,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "decided_at": record.decided_at,
        "decided_by": record.decided_by,
        "reason": record.reason,
        "action": payload.get("action", "write"),
        "selectors": list(payload.get("selectors") or []),
        "targets": list(payload.get("targets") or []),
        "rule": payload.get("rule"),
        "command": record.summary,
    }


@router.get("/hitl")
async def list_approval_requests(
    service: ServiceDep, status: StatusQuery = "pending", limit: int = 100
) -> dict[str, Any]:
    """List requests put to a human, newest first.

    Args:
        service: The process's resources.
        status: One status, or ``all`` (equivalently, empty) for every status.
        limit: How many to return.

    Returns:
        The matching requests.
    """
    selected = None if status in ("all", "") else cast(ApprovalStatus, status)
    records = await asyncio.to_thread(
        service.deployment.storage.approvals.list_requests, status=selected
    )
    return {"requests": [request_payload(record) for record in records[:limit]]}


@router.get("/hitl/{request_id}")
async def read_approval_request(request_id: str, service: ServiceDep) -> dict[str, Any]:
    """Read one request in full.

    Args:
        request_id: Which request.
        service: The process's resources.

    Returns:
        The request.

    Raises:
        HTTPException: 404 when there is no such request.
    """
    record = await asyncio.to_thread(service.deployment.storage.approvals.get_request, request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="request_id not found")
    return request_payload(record)


@router.post("/hitl/{request_id}/decide")
async def decide_approval_request(
    request_id: str, body: DecideRequest, service: ServiceDep
) -> dict[str, Any]:
    """Record a decision and continue the incident it was holding.

    ``decided_by`` is a **claimed** identity, not a verified one. This route has no
    authentication: the audit record says who the caller said they were, and nothing
    checks it. Authentication is a tracked deferral; until it lands, the deployment
    guidance is a NetworkPolicy restricting the port and no ingress. A reader who does
    not know that will assume otherwise, which is why it is written here.

    Args:
        request_id: Which request.
        body: Approve or refuse, why, and who says so.
        service: The process's resources.

    Returns:
        Confirmation carrying the decision.

    Raises:
        HTTPException: 404 when the id is unknown or the request was already decided.
            The store decides by compare-and-set, so a duplicate cannot dispatch a
            second time and this is the same answer either way.
    """
    record = await asyncio.to_thread(
        service.deployment.storage.approvals.decide,
        request_id,
        status="approved" if body.approve else "denied",
        decided_by=body.decided_by,
        reason=body.reason,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="request_id not found or already decided")
    _log.info(
        "approval request %s %s by %s (claimed)",
        record.request_id,
        record.status,
        body.decided_by,
    )
    service.schedule_resume(record.request_id)
    return {"decided": True, "approved": body.approve, "request_id": record.request_id}
