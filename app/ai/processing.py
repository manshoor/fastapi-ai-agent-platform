"""Fenced PostgreSQL job transitions; no transaction crosses a network boundary."""

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import delete, exists, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.config import AISettings
from app.ai.contracts import TERMINAL_STATES, RewriteEvent
from app.ai.provider import Generation, ProviderFailure
from app.models import AIJob, AIOutbox
from app.telemetry import Telemetry

SessionFactory = async_sessionmaker[AsyncSession]


class Rewriter(Protocol):
    async def rewrite(self, title: str, description: str, instruction: str) -> Generation: ...

    async def aclose(self) -> None: ...


ProviderFactory = Callable[[AISettings], Rewriter]


class HandleResult(StrEnum):
    HANDLED = "handled"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class JobLease:
    job_id: UUID
    attempt: int
    token: UUID
    generation: int
    title: str
    description: str
    instruction: str
    model: str
    prompt_version: str
    schema_version: str
    trace_context: dict[str, str]
    created_at: datetime
    claimed_at: datetime


def _fence(lease: JobLease, state: str):
    return (
        AIJob.id == lease.job_id,
        AIJob.status == state,
        AIJob.attempt == lease.attempt,
        AIJob.lease_token == lease.token,
        AIJob.lease_generation == lease.generation,
        AIJob.lease_expires_at > func.clock_timestamp(),
    )


def _terminal(job: AIJob, now: datetime, settings: AISettings, status: str, code: str | None):
    job.status = status
    job.error_code = code
    job.finished_at = now
    job.updated_at = now
    job.content_expires_at = now + timedelta(seconds=settings.ai_content_retention_seconds)
    job.lease_token = None
    job.lease_expires_at = None


def _dead_letter(session: AsyncSession, job: AIJob, now: datetime) -> None:
    session.add(
        AIOutbox(job_id=job.id, event_kind="dead_letter", attempt=job.attempt, available_at=now)
    )


def _record_outcome(
    telemetry: Telemetry | None,
    status: str,
    created_at: datetime,
    code: str | None = None,
) -> None:
    if telemetry is not None:
        attrs = {"status": status}
        if code is not None:
            attrs["error_code"] = code
        telemetry.increment("ai.outcomes", **attrs)
        telemetry.observe(
            "ai.job.duration",
            max(0, (datetime.now(UTC) - created_at).total_seconds()),
            **attrs,
        )


async def claim_job(
    *, session_factory: SessionFactory, settings: AISettings, event: RewriteEvent
) -> JobLease | None:
    """Commit a short-lived lease, accepting only an authentic persisted outbox identity."""
    async with session_factory() as session:
        async with session.begin():
            job = await session.scalar(
                select(AIJob).where(AIJob.id == event.job_id).with_for_update(skip_locked=True)
            )
            if job is None or job.status != "pending" or job.attempt != event.attempt:
                return None
            authentic = await session.scalar(
                select(AIOutbox.id).where(
                    AIOutbox.id == event.event_id,
                    AIOutbox.job_id == job.id,
                    AIOutbox.attempt == event.attempt,
                    AIOutbox.event_kind == "requested",
                )
            )
            if authentic is None:
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            if job.next_attempt_at > now or job.deadline_at <= now:
                return None
            if (
                job.source_title is None
                or job.source_description is None
                or job.instruction is None
            ):
                return None
            job.status = "leased"
            job.lease_token = uuid4()
            job.lease_generation += 1
            job.lease_expires_at = now + timedelta(seconds=settings.ai_lease_seconds)
            job.updated_at = now
            lease = JobLease(
                job.id,
                job.attempt,
                job.lease_token,
                job.lease_generation,
                job.source_title,
                job.source_description,
                job.instruction,
                job.model,
                job.prompt_version,
                job.schema_version,
                dict(job.trace_context),
                job.created_at,
                now,
            )
        return lease


async def start_call(*, session_factory: SessionFactory, lease: JobLease) -> bool:
    """Only a positively acknowledged commit authorizes the provider call."""
    async with session_factory() as session:
        result = await session.execute(
            update(AIJob)
            .where(*_fence(lease, "leased"), AIJob.deadline_at > func.clock_timestamp())
            .values(
                status="calling",
                provider_started_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
            )
        )
        await session.commit()
        return result.rowcount == 1


async def finish_success(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    lease: JobLease,
    generation: Generation,
    telemetry: Telemetry | None = None,
) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(AIJob)
            .where(*_fence(lease, "calling"))
            .values(
                status="succeeded",
                error_code=None,
                finished_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
                content_expires_at=func.clock_timestamp()
                + timedelta(seconds=settings.ai_content_retention_seconds),
                lease_token=None,
                lease_expires_at=None,
                result_title=generation.title,
                result_description=generation.description,
                response_model=generation.model,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                provider_request_id=generation.request_id,
            )
        )
        await session.commit()
        if result.rowcount == 1:
            _record_outcome(telemetry, "succeeded", lease.created_at)
        return result.rowcount == 1


async def finish_failure(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    lease: JobLease,
    failure: ProviderFailure,
    telemetry: Telemetry | None = None,
) -> bool:
    async with session_factory() as session:
        job = await session.scalar(select(AIJob).where(*_fence(lease, "calling")).with_for_update())
        if job is None:
            return False
        now = await session.scalar(select(func.clock_timestamp()))
        delay = timedelta(seconds=min(2 ** (job.attempt - 1), 60))
        retry = (
            failure.retryable
            and not failure.ambiguous
            and job.attempt < min(settings.ai_max_attempts, 3)
            and now + delay < job.deadline_at
        )
        values = {
            "updated_at": func.clock_timestamp(),
            "error_code": failure.code,
            "lease_token": None,
            "lease_expires_at": None,
        }
        if retry:
            values.update(
                status="pending",
                attempt=job.attempt + 1,
                next_attempt_at=func.clock_timestamp() + delay,
                provider_started_at=None,
            )
        else:
            values.update(
                status="needs_reconciliation" if failure.ambiguous else "failed",
                finished_at=func.clock_timestamp(),
                content_expires_at=func.clock_timestamp()
                + timedelta(seconds=settings.ai_content_retention_seconds),
            )
        changed = await session.scalar(
            update(AIJob)
            .where(*_fence(lease, "calling"))
            .values(**values)
            .returning(AIJob.id)
            .execution_options(synchronize_session=False)
        )
        if changed is None:
            return False
        session.add(
            AIOutbox(
                job_id=job.id,
                event_kind="requested" if retry else "dead_letter",
                attempt=lease.attempt + 1 if retry else lease.attempt,
                available_at=now + delay if retry else now,
            )
        )
        await session.commit()
        if retry:
            if telemetry is not None:
                telemetry.increment("ai.retries", error_code=failure.code)
        else:
            _record_outcome(
                telemetry,
                "needs_reconciliation" if failure.ambiguous else "failed",
                lease.created_at,
                failure.code,
            )
        return True


async def _durable_without_claim(
    session_factory: SessionFactory, event: RewriteEvent
) -> HandleResult:
    async with session_factory() as session:
        row = (
            await session.execute(
                select(AIJob.status, AIJob.attempt).where(AIJob.id == event.job_id)
            )
        ).one_or_none()
        authentic = await session.scalar(
            select(AIOutbox.id).where(
                AIOutbox.id == event.event_id,
                AIOutbox.job_id == event.job_id,
                AIOutbox.attempt == event.attempt,
                AIOutbox.event_kind == "requested",
            )
        )
    if row is None or authentic is None or row.status in TERMINAL_STATES:
        return HandleResult.HANDLED
    if row.attempt != event.attempt:
        return HandleResult.HANDLED
    # Another holder, a future retry, or an expired lease must reach durable recovery first.
    return HandleResult.DEFERRED


async def process_event(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    event: RewriteEvent,
    provider_factory: ProviderFactory,
    telemetry: Telemetry | None = None,
    parent_context: dict[str, str] | None = None,
) -> HandleResult:
    claim_context = telemetry.extract_context(parent_context or {}) if telemetry else None
    claim_span = (
        telemetry.span(
            "ai.claim",
            context=claim_context,
            job_id=str(event.job_id),
            event_id=str(event.event_id),
            attempt=event.attempt,
        )
        if telemetry
        else nullcontext()
    )
    with claim_span:
        lease = await claim_job(session_factory=session_factory, settings=settings, event=event)
        if lease is None:
            return await _durable_without_claim(session_factory, event)
        if telemetry is not None:
            telemetry.increment("ai.jobs", status="leased")
            telemetry.observe(
                "ai.queue.wait",
                max(0, (lease.claimed_at - lease.created_at).total_seconds()),
            )
        # An uncertain calling commit outcome escapes BEFORE provider construction.
        if not await start_call(session_factory=session_factory, lease=lease):
            return HandleResult.DEFERRED
    snapshot_settings = settings.model_copy(
        update={
            "ai_model": lease.model,
            "ai_prompt_version": lease.prompt_version,
            "ai_schema_version": lease.schema_version,
        }
    )
    provider = provider_factory(snapshot_settings)
    context = (
        telemetry.extract_context(parent_context or lease.trace_context) if telemetry else None
    )
    span = (
        telemetry.span(
            "ai.worker.process",
            context=context,
            job_id=str(lease.job_id),
            event_id=str(event.event_id),
            attempt=lease.attempt,
            lease_generation=lease.generation,
        )
        if telemetry
        else nullcontext()
    )
    try:
        with span:
            persist_span = (
                telemetry.span(
                    "ai.persist",
                    job_id=str(lease.job_id),
                    attempt=lease.attempt,
                    lease_generation=lease.generation,
                )
                if telemetry
                else nullcontext()
            )
            try:
                generation = await provider.rewrite(
                    lease.title, lease.description, lease.instruction
                )
            except ProviderFailure as failure:
                with persist_span:
                    committed = await finish_failure(
                        session_factory=session_factory,
                        settings=settings,
                        lease=lease,
                        failure=failure,
                        telemetry=telemetry,
                    )
            else:
                with persist_span:
                    committed = await finish_success(
                        session_factory=session_factory,
                        settings=settings,
                        lease=lease,
                        generation=generation,
                        telemetry=telemetry,
                    )
        return HandleResult.HANDLED if committed else HandleResult.DEFERRED
    finally:
        # Cancellation or a failed result commit leaves calling intact, never resubmits a call.
        try:
            async with asyncio.timeout(5):
                await provider.aclose()
        except TimeoutError:
            pass


async def recover_jobs(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    batch_size: int = 100,
    telemetry: Telemetry | None = None,
) -> int:
    """Recover safe pre-call leases; uncertain calls become terminal plus a durable DLQ."""
    async with session_factory() as session:
        async with session.begin():
            jobs = (
                await session.scalars(
                    select(AIJob)
                    .where(
                        or_(
                            (AIJob.status == "pending")
                            & (AIJob.deadline_at <= func.clock_timestamp()),
                            AIJob.status.in_(("leased", "calling"))
                            & (AIJob.lease_expires_at <= func.clock_timestamp()),
                        )
                    )
                    .order_by(AIJob.updated_at, AIJob.id)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            now = await session.scalar(select(func.clock_timestamp()))
            for job in jobs:
                if job.status == "calling":
                    _terminal(job, now, settings, "needs_reconciliation", "worker_call_unknown")
                    _dead_letter(session, job, now)
                elif job.deadline_at <= now:
                    _terminal(job, now, settings, "failed", "queue_expired")
                    _dead_letter(session, job, now)
                else:
                    job.status = "pending"
                    job.lease_token = None
                    job.lease_expires_at = None
                    job.updated_at = now
                    # Keep the original event identity. Its prior broker ACK is not proof of
                    # consumption; reset visibility so this same outbox row is sent again.
                    await session.execute(
                        update(AIOutbox)
                        .where(
                            AIOutbox.job_id == job.id,
                            AIOutbox.event_kind == "requested",
                            AIOutbox.attempt == job.attempt,
                        )
                        .values(published_at=None, available_at=now)
                    )
        for job in jobs:
            if job.status in TERMINAL_STATES:
                _record_outcome(telemetry, job.status, job.created_at, job.error_code)
        return len(jobs)


async def maintain_jobs(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    batch_size: int = 100,
    telemetry: Telemetry | None = None,
) -> int:
    """Bounded publication-settled admission release, content purge, and tombstone expiry."""
    unsettled = exists(
        select(AIOutbox.id).where(
            AIOutbox.job_id == AIJob.id,
            AIOutbox.published_at.is_(None),
        )
    )
    async with session_factory() as session:
        async with session.begin():
            now = await session.scalar(select(func.clock_timestamp()))
            jobs = (
                await session.scalars(
                    select(AIJob)
                    .where(
                        AIJob.status.in_(TERMINAL_STATES),
                        or_(
                            AIJob.admission_released_at.is_(None) & ~unsettled,
                            AIJob.content_purged_at.is_(None) & (AIJob.content_expires_at <= now),
                            (
                                AIJob.finished_at
                                <= now
                                - timedelta(
                                    seconds=max(settings.ai_metadata_retention_seconds, 30 * 86400)
                                )
                            )
                            & ~unsettled,
                        ),
                    )
                    .order_by(AIJob.finished_at, AIJob.id)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            for job in jobs:
                pending_publication = await session.scalar(
                    select(
                        exists().where(
                            AIOutbox.job_id == job.id,
                            AIOutbox.published_at.is_(None),
                        )
                    )
                )
                if not pending_publication and job.admission_released_at is None:
                    job.admission_released_at = now
                if job.content_expires_at is not None and job.content_expires_at <= now:
                    job.source_title = None
                    job.source_description = None
                    job.instruction = None
                    job.result_title = None
                    job.result_description = None
                    job.trace_context = {}
                    job.content_purged_at = now
                cutoff = now - timedelta(
                    seconds=max(settings.ai_metadata_retention_seconds, 30 * 86400)
                )
                if not pending_publication and job.finished_at <= cutoff:
                    await session.execute(delete(AIJob).where(AIJob.id == job.id))
            queue_depth = (
                await session.scalar(
                    select(func.count())
                    .select_from(AIJob)
                    .where(
                        AIJob.status.in_(("pending", "leased", "calling")),
                    )
                )
                if telemetry is not None
                else 0
            )
        if telemetry is not None:
            telemetry.observe("ai.queue.depth", queue_depth)
        return len(jobs)
