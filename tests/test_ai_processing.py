"""Durability regressions at real isolated PostgreSQL commit and lock boundaries."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.config import AISettings
from app.ai.contracts import RewriteEvent, RewriteRequest
from app.ai.jobs import submit_job
from app.ai.processing import (
    HandleResult,
    claim_job,
    finish_failure,
    finish_success,
    maintain_jobs,
    process_event,
    recover_jobs,
    start_call,
)
from app.ai.provider import Generation, ProviderFailure
from app.models import AIJob, AIOutbox, Item, User


class ControlledProvider:
    def __init__(self, *, failure=None, on_call=None):
        self.failure = failure
        self.on_call = on_call
        self.calls = []
        self.closed = 0

    async def rewrite(self, title, description, instruction):
        self.calls.append((title, description, instruction))
        if self.on_call:
            await self.on_call()
        if self.failure:
            raise self.failure
        return Generation("Clear title", "Preserved facts: 4 blue cups.", "snapshot-model", 12, 8)

    async def aclose(self):
        self.closed += 1


@pytest.fixture
async def runtime(app, settings):
    ai = AISettings.model_validate({**settings.model_dump(), "ai_enabled": True})
    factory = app.state.session_factory
    async with factory() as session:
        user = User(email=f"rewrite-{uuid4().hex}@example.com", password_hash="unused-test-hash")
        session.add(user)
        await session.flush()
        item = Item(owner_id=user.id, title="4 cups", description="4 blue cups.")
        session.add(item)
        await session.flush()
        await session.refresh(item)
        owner_id, item_id, updated_at = user.id, item.id, item.updated_at
        await session.commit()

    async def create_job(**overrides):
        async with factory() as session:
            now = await session.scalar(select(func.clock_timestamp()))
            fields = {
                "owner_id": owner_id,
                "item_id": item_id,
                "source_item_updated_at": updated_at,
                "source_title": "4 cups",
                "source_description": "4 blue cups.",
                "instruction": "Make this clearer",
                "idempotency_key_hash": uuid4().hex * 2,
                "request_fingerprint": "f" * 64,
                "deadline_at": now + timedelta(hours=24),
                "model": "snapshot-model",
                "prompt_version": "item-rewrite-v1",
                "schema_version": "1",
            }
            fields.update(overrides)
            job = AIJob(**fields)
            session.add(job)
            await session.flush()
            outbox = AIOutbox(job_id=job.id, event_kind="requested", attempt=job.attempt)
            session.add(outbox)
            await session.flush()
            await session.refresh(outbox)
            event = RewriteEvent(
                version=1,
                event_id=outbox.id,
                job_id=job.id,
                attempt=job.attempt,
                created_at=outbox.created_at,
            )
            await session.commit()
        return event

    async def read_job(job_id):
        async with factory() as session:
            return await session.get(AIJob, job_id)

    async def expire(job_id):
        async with factory() as session:
            await session.execute(
                update(AIJob)
                .where(AIJob.id == job_id)
                .values(
                    lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
                )
            )
            await session.commit()

    return SimpleNamespace(
        settings=ai,
        factory=factory,
        create_job=create_job,
        read_job=read_job,
        expire=expire,
        owner_id=owner_id,
        item_id=item_id,
        app=app,
    )


async def test_success_commits_calling_before_network_and_preserves_snapshot(runtime):
    event = await runtime.create_job()
    seen_settings = []

    async def during_call():
        assert runtime.app.state.engine.pool.checkedout() == 0
        job = await runtime.read_job(event.job_id)
        assert job.status == "calling"
        assert job.provider_started_at is not None
        assert runtime.app.state.engine.pool.checkedout() == 0

    provider = ControlledProvider(on_call=during_call)

    def provider_factory(pinned):
        seen_settings.append(pinned)
        return provider

    current = runtime.settings.model_copy(update={"ai_model": "changed-model"})
    result = await process_event(
        session_factory=runtime.factory,
        settings=current,
        event=event,
        provider_factory=provider_factory,
    )
    assert result == HandleResult.HANDLED
    assert provider.calls == [("4 cups", "4 blue cups.", "Make this clearer")]
    assert provider.closed == 1
    assert seen_settings[0].ai_model == "snapshot-model"
    job = await runtime.read_job(event.job_id)
    assert job.status == "succeeded"
    assert job.result_title == "Clear title"
    assert job.input_tokens == 12 and job.output_tokens == 8
    assert job.finished_at is not None and job.content_expires_at > job.finished_at
    assert job.lease_token is None
    async with runtime.factory() as session:
        item = await session.get(Item, runtime.item_id)
        assert item.title == "4 cups" and item.description == "4 blue cups."
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=current,
            event=event,
            provider_factory=provider_factory,
        )
        == HandleResult.HANDLED
    )
    assert len(provider.calls) == 1


async def test_concurrent_duplicate_claims_never_start_a_second_call(runtime):
    event = await runtime.create_job()
    leases = await asyncio.gather(
        *(
            claim_job(session_factory=runtime.factory, settings=runtime.settings, event=event)
            for _ in range(2)
        )
    )
    assert sum(lease is not None for lease in leases) == 1
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.DEFERRED
    )
    assert provider.calls == []


async def test_expired_pre_call_lease_reuses_event_and_fences_stale_holder(runtime):
    event = await runtime.create_job()
    old = await claim_job(session_factory=runtime.factory, settings=runtime.settings, event=event)
    await runtime.expire(event.job_id)
    assert await recover_jobs(session_factory=runtime.factory, settings=runtime.settings) == 1
    async with runtime.factory() as session:
        outboxes = (
            await session.scalars(
                select(AIOutbox).where(
                    AIOutbox.job_id == event.job_id,
                )
            )
        ).all()
        assert len(outboxes) == 1 and outboxes[0].id == event.event_id
    fresh = await claim_job(session_factory=runtime.factory, settings=runtime.settings, event=event)
    assert fresh.generation == old.generation + 1 and fresh.token != old.token
    assert not await start_call(session_factory=runtime.factory, lease=old)
    assert await start_call(session_factory=runtime.factory, lease=fresh)
    generation = Generation("Clear", "Facts", "snapshot-model")
    assert not await finish_success(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=old,
        generation=generation,
    )
    assert not await finish_failure(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=old,
        failure=ProviderFailure("provider_refusal"),
    )
    assert await finish_success(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=fresh,
        generation=generation,
    )


async def test_expired_call_is_reconciled_atomically_and_never_replayed(runtime):
    event = await runtime.create_job()
    lease = await claim_job(session_factory=runtime.factory, settings=runtime.settings, event=event)
    assert await start_call(session_factory=runtime.factory, lease=lease)
    await runtime.expire(event.job_id)
    assert not await finish_success(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=lease,
        generation=Generation("Too late", "Too late", "snapshot-model"),
    )
    assert await recover_jobs(session_factory=runtime.factory, settings=runtime.settings) == 1
    assert await recover_jobs(session_factory=runtime.factory, settings=runtime.settings) == 0
    job = await runtime.read_job(event.job_id)
    assert job.status == "needs_reconciliation" and job.error_code == "worker_call_unknown"
    async with runtime.factory() as session:
        rows = (
            await session.scalars(
                select(AIOutbox).where(
                    AIOutbox.job_id == job.id,
                    AIOutbox.event_kind == "dead_letter",
                )
            )
        ).all()
        assert len(rows) == 1 and rows[0].attempt == 1
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.HANDLED
    )
    assert provider.calls == []


async def test_expired_queue_never_calls_provider(runtime):
    event = await runtime.create_job()
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == event.job_id)
            .values(
                deadline_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.DEFERRED
    )
    await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
    job = await runtime.read_job(event.job_id)
    assert job.status == "failed" and job.error_code == "queue_expired"
    assert provider.calls == []


async def test_known_rejection_retries_three_durable_attempts_only(runtime):
    event = await runtime.create_job()
    original = event
    provider = ControlledProvider(failure=ProviderFailure("provider_rate_limit", retryable=True))
    for attempt in range(1, 4):
        assert (
            await process_event(
                session_factory=runtime.factory,
                settings=runtime.settings,
                event=event,
                provider_factory=lambda _: provider,
            )
            == HandleResult.HANDLED
        )
        job = await runtime.read_job(event.job_id)
        if attempt < 3:
            assert job.status == "pending" and job.attempt == attempt + 1
            assert job.next_attempt_at > job.updated_at
            # The stale Kafka request is settled, not a fourth provider invocation.
            assert (
                await process_event(
                    session_factory=runtime.factory,
                    settings=runtime.settings,
                    event=original,
                    provider_factory=lambda _: provider,
                )
                == HandleResult.HANDLED
            )
            async with runtime.factory() as session:
                await session.execute(
                    update(AIJob)
                    .where(AIJob.id == job.id)
                    .values(
                        next_attempt_at=func.clock_timestamp() - timedelta(seconds=1),
                    )
                )
                row = await session.scalar(
                    select(AIOutbox).where(
                        AIOutbox.job_id == job.id,
                        AIOutbox.attempt == attempt + 1,
                        AIOutbox.event_kind == "requested",
                    )
                )
                event = RewriteEvent(
                    version=1,
                    event_id=row.id,
                    job_id=job.id,
                    attempt=row.attempt,
                    created_at=row.created_at,
                )
                await session.commit()
        else:
            assert job.status == "failed" and job.error_code == "provider_rate_limit"
    assert len(provider.calls) == 3
    async with runtime.factory() as session:
        rows = (
            await session.scalars(
                select(AIOutbox).where(
                    AIOutbox.job_id == event.job_id,
                )
            )
        ).all()
        assert sorted((row.event_kind, row.attempt) for row in rows) == [
            ("dead_letter", 3),
            ("requested", 1),
            ("requested", 2),
            ("requested", 3),
        ]


@pytest.mark.parametrize(
    "ambiguous,code,state",
    [
        (True, "provider_timeout", "needs_reconciliation"),
        (False, "provider_refusal", "failed"),
        (False, "provider_invalid_output", "failed"),
    ],
)
async def test_terminal_provider_outcomes_never_schedule_another_call(
    runtime, ambiguous, code, state
):
    event = await runtime.create_job()
    provider = ControlledProvider(failure=ProviderFailure(code, ambiguous=ambiguous))
    await process_event(
        session_factory=runtime.factory,
        settings=runtime.settings,
        event=event,
        provider_factory=lambda _: provider,
    )
    job = await runtime.read_job(event.job_id)
    assert job.status == state and job.error_code == code and job.attempt == 1
    async with runtime.factory() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(AIOutbox)
            .where(
                AIOutbox.job_id == job.id,
            )
        )
        assert count == 2


def failing_factory(runtime, *, fail_commit, after_commit=False):
    commits = 0

    class CommitFailure(AsyncSession):
        async def commit(self):
            nonlocal commits
            commits += 1
            if commits == fail_commit:
                if after_commit:
                    await super().commit()
                raise ConnectionError("synthetic lost database acknowledgement")
            await super().commit()

    return async_sessionmaker(
        runtime.app.state.engine,
        class_=CommitFailure,
        expire_on_commit=False,
    )


@pytest.mark.parametrize("after_commit", [False, True])
async def test_unknown_calling_commit_never_authorizes_provider(runtime, after_commit):
    event = await runtime.create_job()
    provider = ControlledProvider()
    with pytest.raises(ConnectionError):
        await process_event(
            session_factory=failing_factory(runtime, fail_commit=1, after_commit=after_commit),
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
    assert provider.calls == []
    assert (await runtime.read_job(event.job_id)).status == (
        "calling" if after_commit else "leased"
    )
    await runtime.expire(event.job_id)
    await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
    assert (await runtime.read_job(event.job_id)).status == (
        "needs_reconciliation" if after_commit else "pending"
    )


@pytest.mark.parametrize("after_commit", [False, True])
async def test_result_commit_failure_never_issues_another_provider_call(runtime, after_commit):
    event = await runtime.create_job()
    provider = ControlledProvider()
    with pytest.raises(ConnectionError):
        await process_event(
            session_factory=failing_factory(runtime, fail_commit=2, after_commit=after_commit),
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
    assert len(provider.calls) == 1 and provider.closed == 1
    assert await process_event(
        session_factory=runtime.factory,
        settings=runtime.settings,
        event=event,
        provider_factory=lambda _: provider,
    ) == (HandleResult.HANDLED if after_commit else HandleResult.DEFERRED)
    assert len(provider.calls) == 1
    if not after_commit:
        await runtime.expire(event.job_id)
        await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
        assert (await runtime.read_job(event.job_id)).status == "needs_reconciliation"


async def test_cancellation_during_provider_call_requires_reconciliation(runtime):
    event = await runtime.create_job()
    started = asyncio.Event()

    async def block():
        started.set()
        await asyncio.Event().wait()

    provider = ControlledProvider(on_call=block)
    task = asyncio.create_task(
        process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await runtime.read_job(event.job_id)).status == "calling"
    await runtime.expire(event.job_id)
    await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
    assert (await runtime.read_job(event.job_id)).status == "needs_reconciliation"
    assert len(provider.calls) == 1 and provider.closed == 1


async def test_terminal_admission_waits_for_all_publication_then_allows_new_work(runtime):
    ai = runtime.settings.model_copy(update={"ai_owner_limit": 1})

    async def submit(key):
        async with runtime.factory() as session:
            return await submit_job(
                session=session,
                settings=ai,
                owner_id=runtime.owner_id,
                item_id=runtime.item_id,
                payload=RewriteRequest(instruction="Make clearer"),
                idempotency_key=key,
                trace_context={},
            )

    result, _ = await submit("quota-lifecycle-0001")
    async with runtime.factory() as session:
        row = await session.scalar(select(AIOutbox).where(AIOutbox.job_id == result.id))
        event = RewriteEvent(
            version=1,
            event_id=row.id,
            job_id=result.id,
            attempt=1,
            created_at=row.created_at,
        )
    provider = ControlledProvider(failure=ProviderFailure("provider_refusal"))
    await process_event(
        session_factory=runtime.factory,
        settings=ai,
        event=event,
        provider_factory=lambda _: provider,
    )
    await maintain_jobs(session_factory=runtime.factory, settings=ai)
    assert (await runtime.read_job(result.id)).admission_released_at is None
    with pytest.raises(HTTPException) as full:
        await submit("quota-lifecycle-0002")
    assert full.value.status_code == 429
    async with runtime.factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(
                AIOutbox.job_id == result.id,
                AIOutbox.event_kind == "requested",
            )
            .values(published_at=func.clock_timestamp(), last_published_at=func.clock_timestamp())
        )
        await session.commit()
    await maintain_jobs(session_factory=runtime.factory, settings=ai)
    assert (await runtime.read_job(result.id)).admission_released_at is None
    async with runtime.factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.job_id == result.id)
            .values(
                published_at=func.clock_timestamp(),
                last_published_at=func.clock_timestamp(),
            )
        )
        await session.commit()
    await maintain_jobs(session_factory=runtime.factory, settings=ai)
    assert (await runtime.read_job(result.id)).admission_released_at is not None
    second, code = await submit("quota-lifecycle-0002")
    assert code == 202 and second.id != result.id


async def test_retention_purges_content_but_keeps_unpublished_tombstones(runtime):
    event = await runtime.create_job()
    provider = ControlledProvider()
    await process_event(
        session_factory=runtime.factory,
        settings=runtime.settings,
        event=event,
        provider_factory=lambda _: provider,
    )
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == event.job_id)
            .values(
                finished_at=func.clock_timestamp() - timedelta(days=31),
                content_expires_at=func.clock_timestamp() - timedelta(days=24),
            )
        )
        await session.commit()
    await maintain_jobs(session_factory=runtime.factory, settings=runtime.settings)
    job = await runtime.read_job(event.job_id)
    assert job is not None and job.content_purged_at is not None
    assert job.source_title is None and job.source_description is None and job.instruction is None
    assert job.result_title is None and job.result_description is None
    assert job.admission_released_at is None
    async with runtime.factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.job_id == event.job_id)
            .values(
                published_at=func.clock_timestamp(),
                last_published_at=func.clock_timestamp(),
            )
        )
        await session.commit()
    await maintain_jobs(session_factory=runtime.factory, settings=runtime.settings)
    assert await runtime.read_job(event.job_id) is None
    async with runtime.factory() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AIOutbox)
                .where(
                    AIOutbox.job_id == event.job_id,
                )
            )
            == 0
        )


async def test_wrong_event_identity_cannot_trigger_a_real_job(runtime):
    event = await runtime.create_job()
    forged = event.model_copy(update={"event_id": uuid4()})
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=forged,
            provider_factory=lambda _: provider,
        )
        == HandleResult.HANDLED
    )
    assert provider.calls == []
    assert (await runtime.read_job(event.job_id)).status == "pending"


@pytest.mark.parametrize("retry", [False, True])
async def test_terminal_or_retry_outbox_failure_rolls_back_state_atomically(runtime, retry):
    from sqlalchemy.exc import IntegrityError

    event = await runtime.create_job()
    # Force a real unique-constraint failure during the outcome transaction's outbox INSERT.
    async with runtime.factory() as session:
        session.add(
            AIOutbox(
                job_id=event.job_id,
                event_kind="requested" if retry else "dead_letter",
                attempt=2 if retry else 1,
            )
        )
        await session.commit()
    provider = ControlledProvider(
        failure=ProviderFailure(
            "provider_rate_limit" if retry else "provider_refusal",
            retryable=retry,
        )
    )
    with pytest.raises(IntegrityError):
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
    job = await runtime.read_job(event.job_id)
    assert job.status == "calling" and job.finished_at is None and job.attempt == 1
    assert job.error_code is None and job.result_title is None
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.DEFERRED
    )
    assert len(provider.calls) == 1


async def test_known_rejection_cannot_schedule_after_latest_start_deadline(runtime):
    event = await runtime.create_job()
    lease = await claim_job(session_factory=runtime.factory, settings=runtime.settings, event=event)
    assert await start_call(session_factory=runtime.factory, lease=lease)
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == event.job_id)
            .values(
                deadline_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    assert await finish_failure(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=lease,
        failure=ProviderFailure("provider_rate_limit", retryable=True),
    )
    job = await runtime.read_job(event.job_id)
    assert job.status == "failed" and job.attempt == 1
    async with runtime.factory() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AIOutbox)
                .where(
                    AIOutbox.job_id == job.id,
                    AIOutbox.event_kind == "requested",
                )
            )
            == 1
        )


async def test_recovery_batches_are_bounded_and_include_expired_leased_deadlines(runtime):
    events = [await runtime.create_job() for _ in range(5)]
    await claim_job(session_factory=runtime.factory, settings=runtime.settings, event=events[0])
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(
                AIJob.id.in_(
                    [event.job_id for event in events],
                )
            )
            .values(
                deadline_at=func.clock_timestamp() - timedelta(seconds=1),
                lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    for expected in (2, 2, 1, 0):
        assert (
            await recover_jobs(
                session_factory=runtime.factory,
                settings=runtime.settings,
                batch_size=2,
            )
            == expected
        )
    for event in events:
        job = await runtime.read_job(event.job_id)
        assert job.status == "failed" and job.error_code == "queue_expired"


async def test_lost_claim_commit_ack_does_not_construct_or_call_provider(runtime):
    from contextlib import asynccontextmanager

    class LostClaimAckSession(AsyncSession):
        def begin(self):
            transaction = super().begin()

            @asynccontextmanager
            async def boundary():
                async with transaction:
                    yield
                raise ConnectionError("synthetic lost lease commit acknowledgement")

            return boundary()

    broken = async_sessionmaker(
        runtime.app.state.engine,
        class_=LostClaimAckSession,
        expire_on_commit=False,
    )
    event = await runtime.create_job()
    constructed = []

    def provider_factory(settings):
        constructed.append(settings)
        return ControlledProvider()

    with pytest.raises(ConnectionError):
        await process_event(
            session_factory=broken,
            settings=runtime.settings,
            event=event,
            provider_factory=provider_factory,
        )
    assert constructed == []
    assert (await runtime.read_job(event.job_id)).status == "leased"
    await runtime.expire(event.job_id)
    await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
    assert (await runtime.read_job(event.job_id)).status == "pending"


async def test_wire_created_at_never_changes_database_latest_start_decision(runtime):
    from datetime import UTC, datetime

    event = await runtime.create_job()
    future_event = event.model_copy(update={"created_at": datetime(2099, 1, 1, tzinfo=UTC)})
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == event.job_id)
            .values(
                deadline_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=future_event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.DEFERRED
    )
    await recover_jobs(session_factory=runtime.factory, settings=runtime.settings)
    assert (await runtime.read_job(event.job_id)).error_code == "queue_expired"
    assert provider.calls == []


async def test_expired_content_keeps_settled_metadata_until_thirty_days(runtime):
    event = await runtime.create_job()
    await process_event(
        session_factory=runtime.factory,
        settings=runtime.settings,
        event=event,
        provider_factory=lambda _: ControlledProvider(),
    )
    async with runtime.factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == event.job_id)
            .values(
                finished_at=func.clock_timestamp() - timedelta(days=8),
                content_expires_at=func.clock_timestamp() - timedelta(days=1),
            )
        )
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.job_id == event.job_id)
            .values(
                published_at=func.clock_timestamp(),
                last_published_at=func.clock_timestamp(),
            )
        )
        await session.commit()
    await maintain_jobs(session_factory=runtime.factory, settings=runtime.settings)
    job = await runtime.read_job(event.job_id)
    assert job is not None and job.status == "succeeded"
    assert job.content_purged_at is not None and job.admission_released_at is not None
    assert job.result_title is None and job.source_title is None


async def test_deleted_source_item_does_not_erase_or_change_accepted_snapshot(runtime):
    event = await runtime.create_job()
    async with runtime.factory() as session:
        item = await session.get(Item, runtime.item_id)
        await session.delete(item)
        await session.commit()
    provider = ControlledProvider()
    assert (
        await process_event(
            session_factory=runtime.factory,
            settings=runtime.settings,
            event=event,
            provider_factory=lambda _: provider,
        )
        == HandleResult.HANDLED
    )
    job = await runtime.read_job(event.job_id)
    assert job.status == "succeeded" and job.item_id == runtime.item_id
    assert provider.calls == [("4 cups", "4 blue cups.", "Make this clearer")]
