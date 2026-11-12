"""Direct owner-scoped SQL; no cache, provider, or queue I/O in API transactions."""

import hashlib
import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.config import AISettings
from app.ai.contracts import TERMINAL_STATES, JobResponse, RewriteRequest, RewriteSuggestion
from app.models import AIAdmission, AIJob, AIOutbox, Item
from app.telemetry import Telemetry


def job_response(job: AIJob, now: datetime) -> JobResponse:
    if job.content_expires_at is not None and job.content_expires_at <= now:
        raise HTTPException(410, "Job content expired", headers={"Cache-Control": "no-store"})
    suggestion = None
    if job.status == "succeeded":
        suggestion = RewriteSuggestion(title=job.result_title, description=job.result_description)
    return JobResponse(
        id=job.id,
        item_id=job.item_id,
        status=job.status,
        source_item_updated_at=job.source_item_updated_at,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
        content_expires_at=job.content_expires_at,
        suggestion=suggestion,
        error_code=job.error_code,
        status_url=f"/api/v1/ai/jobs/{job.id}",
    )


def _fingerprint(item_id: UUID, payload: RewriteRequest) -> str:
    canonical = json.dumps(
        ["item-rewrite-v1", str(item_id), payload.instruction],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def submit_job(
    *,
    session: AsyncSession,
    settings: AISettings,
    owner_id: UUID,
    item_id: UUID,
    payload: RewriteRequest,
    idempotency_key: str,
    trace_context: dict[str, str],
    telemetry: Telemetry | None = None,
) -> tuple[JobResponse, int]:
    key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    fingerprint = _fingerprint(item_id, payload)

    async def existing() -> tuple[JobResponse, int] | None:
        row = (
            await session.execute(
                select(AIJob, func.clock_timestamp()).where(
                    AIJob.owner_id == owner_id, AIJob.idempotency_key_hash == key_hash
                )
            )
        ).first()
        if row is None:
            return None
        job, now = row
        if job.request_fingerprint != fingerprint:
            await session.rollback()
            raise HTTPException(409, "Idempotency key conflict")
        response = job_response(job, now)
        status_code = 200 if job.status in TERMINAL_STATES else 202
        await session.rollback()
        return response, status_code

    replay = await existing()
    if replay is not None:
        return replay
    # All submissions take this lock before quota checks/insertion. Terminal
    # transitions only release capacity, so they need not take the admission lock.
    (
        await session.execute(select(AIAdmission.id).where(AIAdmission.id == 1).with_for_update())
    ).scalar_one()
    replay = await existing()
    if replay is not None:
        return replay
    unsettled = AIJob.admission_released_at.is_(None)
    global_count = await session.scalar(select(func.count()).select_from(AIJob).where(unsettled))
    if global_count >= settings.ai_global_limit:
        await session.rollback()
        raise HTTPException(503, "Rewrite capacity unavailable", headers={"Retry-After": "5"})
    owner_count = await session.scalar(
        select(func.count()).select_from(AIJob).where(unsettled, AIJob.owner_id == owner_id)
    )
    if owner_count >= settings.ai_owner_limit:
        await session.rollback()
        raise HTTPException(429, "Rewrite owner limit reached", headers={"Retry-After": "5"})
    row = (
        await session.execute(
            select(Item, func.clock_timestamp()).where(
                Item.id == item_id, Item.owner_id == owner_id
            )
        )
    ).first()
    if row is None:
        await session.rollback()
        raise HTTPException(404, "Item not found")
    item, now = row
    job_id = uuid4()
    job = AIJob(
        id=job_id,
        owner_id=owner_id,
        item_id=item.id,
        source_item_updated_at=item.updated_at,
        source_title=item.title,
        source_description=item.description,
        instruction=payload.instruction,
        idempotency_key_hash=key_hash,
        request_fingerprint=fingerprint,
        status="pending",
        attempt=1,
        next_attempt_at=now,
        deadline_at=now + timedelta(seconds=settings.ai_queue_timeout_seconds),
        model=settings.ai_model,
        prompt_version=settings.ai_prompt_version,
        schema_version=settings.ai_schema_version,
        trace_context=trace_context,
    )
    session.add(job)
    await session.flush()
    session.add(AIOutbox(job_id=job_id, event_kind="requested", attempt=1, available_at=now))
    await session.flush()
    response = job_response(job, now)
    await session.commit()
    if telemetry is not None:
        telemetry.increment("ai.jobs", status="pending")
    return response, 202


async def poll_job(*, session: AsyncSession, owner_id: UUID, job_id: UUID) -> JobResponse:
    row = (
        await session.execute(
            select(AIJob, func.clock_timestamp()).where(
                AIJob.id == job_id, AIJob.owner_id == owner_id
            )
        )
    ).first()
    if row is None:
        await session.rollback()
        raise HTTPException(404, "Job not found")
    response = job_response(*row)
    await session.rollback()
    return response
