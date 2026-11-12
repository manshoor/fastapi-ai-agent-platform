"""Real Kafka delivery/offset regressions, with controlled failures at real DB boundaries."""

import asyncio
import hashlib
import json
import os
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaConnectionError
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from test_ai_processing import ControlledProvider
from test_ai_processing import runtime as runtime

from app.ai import kafka
from app.ai.contracts import RewriteEvent
from app.ai.kafka import (
    acknowledge_outbox,
    claim_outbox,
    close_producer,
    create_producer,
    dispatch_once,
    extract_trace_headers,
    handle_record,
)
from app.ai.processing import HandleResult, claim_job, maintain_jobs, process_event, recover_jobs
from app.ai.provider import ProviderFailure
from app.ai.worker import PartitionProcessor
from app.models import AIJob, AIOutbox


@pytest.fixture
async def broker(runtime):
    if os.environ.get("AI_QA") != "1":
        pytest.skip("Real Kafka integration requires the isolated AI QA topology")
    suffix = uuid4().hex
    settings = runtime.settings.model_copy(
        update={
            "kafka_request_topic": f"test.rewrite.request.{suffix}",
            "kafka_dead_letter_topic": f"test.rewrite.dlq.{suffix}",
            "kafka_group_id": f"test.rewrite.group.{suffix}",
            "ai_redispatch_seconds": 60,
        }
    )
    admin = AIOKafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        request_timeout_ms=5000,
    )
    producer = None
    consumer = None
    try:
        async with asyncio.timeout(15):
            await admin.start()
            await admin.create_topics(
                [
                    NewTopic(
                        name=topic,
                        num_partitions=6,
                        replication_factor=1,
                        topic_configs={
                            "retention.ms": "3600000",
                            "retention.bytes": "10485760",
                            "max.message.bytes": "16384",
                        },
                    )
                    for topic in (settings.kafka_request_topic, settings.kafka_dead_letter_topic)
                ]
            )
        producer = await create_producer(settings)
        consumer = AIOKafkaConsumer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            request_timeout_ms=5000,
            fetch_max_bytes=262144,
            max_partition_fetch_bytes=65536,
        )
        partitions = [
            TopicPartition(topic, p)
            for topic in (settings.kafka_request_topic, settings.kafka_dead_letter_topic)
            for p in range(6)
        ]
        consumer.assign(partitions)
        async with asyncio.timeout(15):
            await consumer.start()

        async def receive(topic=None):
            wanted = [tp for tp in partitions if topic is None or tp.topic == topic]
            async with asyncio.timeout(15):
                return await consumer.getone(*wanted)

        yield SimpleNamespace(
            settings=settings,
            producer=producer,
            consumer=consumer,
            receive=receive,
        )
    finally:
        if consumer is not None:
            async with asyncio.timeout(10):
                await consumer.stop()
        if producer is not None:
            await close_producer(producer)
        try:
            async with asyncio.timeout(10):
                await admin.delete_topics(
                    [
                        settings.kafka_request_topic,
                        settings.kafka_dead_letter_topic,
                    ]
                )
        finally:
            await admin.close()


async def test_outbox_send_and_ack_use_no_checked_out_database_connection(runtime, broker):
    event = await runtime.create_job()

    class BoundaryProducer:
        async def send_and_wait(self, topic, value, *, key):
            assert runtime.app.state.engine.pool.checkedout() == 0
            async with runtime.factory() as session:
                row = await session.get(AIOutbox, event.event_id)
                assert row.lease_token is not None and row.published_at is None
            return await broker.producer.send_and_wait(topic, value=value, key=key)

    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=BoundaryProducer(),
        )
        == 1
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    assert RewriteEvent.model_validate_json(record.value) == event
    assert record.key == str(event.job_id).encode()
    async with runtime.factory() as session:
        row = await session.get(AIOutbox, event.event_id)
        assert row.published_at is not None and row.last_published_at is not None
        assert row.publish_attempts == 1 and row.lease_token is None


async def test_crash_after_broker_ack_republishes_same_event_identity(runtime, broker):
    event = await runtime.create_job()

    class CrashAfterSend:
        async def send_and_wait(self, topic, value, *, key):
            await broker.producer.send_and_wait(topic, value=value, key=key)
            raise ConnectionError("synthetic process death after broker acknowledgement")

    with pytest.raises(ConnectionError):
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=CrashAfterSend(),
        )
    first = await broker.receive(broker.settings.kafka_request_topic)
    async with runtime.factory() as session:
        row = await session.get(AIOutbox, event.event_id)
        assert row.published_at is None and row.lease_token is not None
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.id == event.event_id)
            .values(
                lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=broker.producer,
        )
        == 1
    )
    second = await broker.receive(broker.settings.kafka_request_topic)
    assert first.value == second.value
    assert RewriteEvent.model_validate_json(second.value).event_id == event.event_id
    provider = ControlledProvider()
    for record in (first, second):
        assert (
            await handle_record(
                session_factory=runtime.factory,
                settings=broker.settings,
                record=record,
                producer=broker.producer,
                provider_factory=lambda _: provider,
            )
            == HandleResult.HANDLED
        )
    assert len(provider.calls) == 1


async def test_unknown_db_publication_ack_republishes_without_losing_admission(runtime, broker):
    event = await runtime.create_job()

    class LostAckSession(AsyncSession):
        async def commit(self):
            raise ConnectionError("synthetic publication database failure")

    broken = async_sessionmaker(
        runtime.app.state.engine,
        class_=LostAckSession,
        expire_on_commit=False,
    )
    with pytest.raises(ConnectionError):
        await dispatch_once(
            session_factory=broken,
            settings=broker.settings,
            producer=broker.producer,
        )
    first = await broker.receive(broker.settings.kafka_request_topic)
    async with runtime.factory() as session:
        row = await session.get(AIOutbox, event.event_id)
        job = await session.get(AIJob, event.job_id)
        assert row.published_at is None and job.admission_released_at is None
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.id == event.event_id)
            .values(
                lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    await dispatch_once(
        session_factory=runtime.factory,
        settings=broker.settings,
        producer=broker.producer,
    )
    second = await broker.receive(broker.settings.kafka_request_topic)
    assert first.value == second.value


async def test_outbox_claim_is_exclusive_and_expired_holder_cannot_ack(runtime, broker):
    event = await runtime.create_job()
    claims = await asyncio.gather(
        *(claim_outbox(session_factory=runtime.factory, settings=broker.settings) for _ in range(2))
    )
    assert sum(claim is not None for claim in claims) == 1
    old = next(claim for claim in claims if claim is not None)
    async with runtime.factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.id == event.event_id)
            .values(
                lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()
    fresh = await claim_outbox(session_factory=runtime.factory, settings=broker.settings)
    assert fresh.token != old.token
    assert not await acknowledge_outbox(session_factory=runtime.factory, lease=old)
    assert await acknowledge_outbox(session_factory=runtime.factory, lease=fresh)


async def test_unconsumed_pending_job_redispatches_without_new_outbox_row(runtime, broker):
    event = await runtime.create_job()
    await dispatch_once(
        session_factory=runtime.factory,
        settings=broker.settings,
        producer=broker.producer,
    )
    first = await broker.receive(broker.settings.kafka_request_topic)
    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=broker.producer,
        )
        == 0
    )
    async with runtime.factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(AIOutbox.id == event.event_id)
            .values(
                last_published_at=func.clock_timestamp() - timedelta(seconds=61),
            )
        )
        await session.commit()
    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=broker.producer,
        )
        == 1
    )
    second = await broker.receive(broker.settings.kafka_request_topic)
    assert first.value == second.value
    async with runtime.factory() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AIOutbox)
                .where(
                    AIOutbox.job_id == event.job_id,
                )
            )
            == 1
        )


async def test_broker_failure_keeps_publication_unsettled_and_does_not_spin(runtime):
    event = await runtime.create_job()

    class OfflineProducer:
        calls = 0

        async def send_and_wait(self, topic, value, *, key):
            self.calls += 1
            raise KafkaConnectionError()

    producer = OfflineProducer()
    with pytest.raises(KafkaConnectionError):
        await dispatch_once(
            session_factory=runtime.factory,
            settings=runtime.settings,
            producer=producer,
        )
    assert producer.calls == 1
    async with runtime.factory() as session:
        row = await session.get(AIOutbox, event.event_id)
        assert row.published_at is None and row.lease_token is None
        assert row.available_at > row.created_at


async def test_hung_broker_send_has_deadline_and_shutdown_remains_finite(runtime, monkeypatch):
    event = await runtime.create_job()
    monkeypatch.setattr(kafka, "KAFKA_IO_TIMEOUT", 0.05)

    class HungProducer:
        async def send_and_wait(self, topic, value, *, key):
            await asyncio.Event().wait()

    async with asyncio.timeout(2):
        with pytest.raises(TimeoutError):
            await dispatch_once(
                session_factory=runtime.factory,
                settings=runtime.settings,
                producer=HungProducer(),
            )
    async with runtime.factory() as session:
        assert (await session.get(AIOutbox, event.event_id)).published_at is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"secret":"never copy this"}',
        b"not-json private content",
        b"x" * 4097,
        b"",
    ],
)
async def test_malformed_delivery_dlq_has_only_hash_coordinates_and_fixed_code(
    runtime, broker, raw
):
    metadata = await broker.producer.send_and_wait(
        broker.settings.kafka_request_topic,
        value=raw,
        key=b"malformed",
        partition=0,
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    provider = ControlledProvider()
    assert (
        await handle_record(
            session_factory=runtime.factory,
            settings=broker.settings,
            record=record,
            producer=broker.producer,
            provider_factory=lambda _: provider,
        )
        == HandleResult.HANDLED
    )
    dlq = await broker.receive(broker.settings.kafka_dead_letter_topic)
    assert json.loads(dlq.value) == {
        "version": 1,
        "error_code": "malformed_event",
        "source_topic": broker.settings.kafka_request_topic,
        "source_partition": 0,
        "source_offset": metadata.offset,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert provider.calls == []


async def test_malformed_dlq_failure_never_commits_source_offset(runtime, broker):
    await broker.producer.send_and_wait(
        broker.settings.kafka_request_topic,
        value=b"private invalid json",
        key=b"bad",
        partition=0,
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    tp = TopicPartition(record.topic, record.partition)

    class OfflineProducer:
        async def send_and_wait(self, topic, value, *, key):
            raise KafkaConnectionError()

    processor = PartitionProcessor(
        consumer=broker.consumer,
        producer=OfflineProducer(),
        session_factory=runtime.factory,
        settings=broker.settings,
        provider_factory=lambda _: ControlledProvider(),
        stop=asyncio.Event(),
    )
    with pytest.raises(KafkaConnectionError):
        await processor.process_partition(tp, [record])
    assert await broker.consumer.committed(tp) is None


async def test_one_partition_commits_contiguous_offsets_after_each_durable_job(runtime, broker):
    events = [await runtime.create_job() for _ in range(3)]
    records = []
    for event in events:
        await broker.producer.send_and_wait(
            broker.settings.kafka_request_topic,
            value=event.model_dump_json().encode(),
            key=str(event.job_id).encode(),
            partition=0,
        )
        records.append(await broker.receive(broker.settings.kafka_request_topic))
    tp = TopicPartition(broker.settings.kafka_request_topic, 0)
    seen = []

    class CommitObserver:
        async def commit(self, offsets):
            expected = events[len(seen)]
            assert (await runtime.read_job(expected.job_id)).status == "succeeded"
            assert offsets == {tp: records[len(seen)].offset + 1}
            seen.append(offsets[tp])
            await broker.consumer.commit(offsets)

        def assignment(self):
            return broker.consumer.assignment()

        def resume(self, partition):
            broker.consumer.resume(partition)

    provider = ControlledProvider()
    processor = PartitionProcessor(
        consumer=CommitObserver(),
        producer=broker.producer,
        session_factory=runtime.factory,
        settings=broker.settings,
        provider_factory=lambda _: provider,
        stop=asyncio.Event(),
    )
    await processor.process_partition(tp, records)
    assert seen == [record.offset + 1 for record in records]
    assert await broker.consumer.committed(tp) == records[-1].offset + 1
    assert len(provider.calls) == 3


async def test_partition_cannot_skip_a_job_leased_elsewhere(runtime, broker):
    first, second = await runtime.create_job(), await runtime.create_job()
    await claim_job(session_factory=runtime.factory, settings=broker.settings, event=first)
    records = []
    for event in (first, second):
        await broker.producer.send_and_wait(
            broker.settings.kafka_request_topic,
            value=event.model_dump_json().encode(),
            key=b"ordered",
            partition=0,
        )
        records.append(await broker.receive(broker.settings.kafka_request_topic))
    provider = ControlledProvider()
    stop = asyncio.Event()
    tp = TopicPartition(broker.settings.kafka_request_topic, 0)
    deferred = asyncio.Event()

    class ObserveWaitConsumer:
        async def commit(self, offsets):
            await broker.consumer.commit(offsets)

        def assignment(self):
            return broker.consumer.assignment()

        def resume(self, partition):
            broker.consumer.resume(partition)

    processor = PartitionProcessor(
        consumer=ObserveWaitConsumer(),
        producer=broker.producer,
        session_factory=runtime.factory,
        settings=broker.settings,
        provider_factory=lambda _: provider,
        stop=stop,
    )
    # Synchronize on the real database read which observes the lease, not a sleep.
    original = kafka.process_event

    async def observed_process(**kwargs):
        result = await original(**kwargs)
        if result == HandleResult.DEFERRED:
            deferred.set()
        return result

    from unittest.mock import patch

    with patch.object(kafka, "process_event", observed_process):
        task = asyncio.create_task(processor.process_partition(tp, records))
        await asyncio.wait_for(deferred.wait(), 5)
        assert provider.calls == []
        assert await broker.consumer.committed(tp) is None
        assert (await runtime.read_job(second.job_id)).status == "pending"
        await runtime.expire(first.job_id)
        await recover_jobs(session_factory=runtime.factory, settings=broker.settings)
        await asyncio.wait_for(task, 10)
    assert len(provider.calls) == 2
    assert await broker.consumer.committed(tp) == records[-1].offset + 1


async def test_rebalance_cancels_call_without_committing_or_replaying(runtime, broker, monkeypatch):
    event = await runtime.create_job()
    await broker.producer.send_and_wait(
        broker.settings.kafka_request_topic,
        value=event.model_dump_json().encode(),
        key=b"rebalance",
        partition=0,
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    started = asyncio.Event()

    async def block_call():
        started.set()
        await asyncio.Event().wait()

    provider = ControlledProvider(on_call=block_call)
    tp = TopicPartition(broker.settings.kafka_request_topic, 0)
    processor = PartitionProcessor(
        consumer=broker.consumer,
        producer=broker.producer,
        session_factory=runtime.factory,
        settings=broker.settings,
        provider_factory=lambda _: provider,
        stop=asyncio.Event(),
    )
    from app.ai import worker

    monkeypatch.setattr(worker, "DRAIN_SECONDS", 0.01)
    processor.submit(tp, [record])
    await asyncio.wait_for(started.wait(), 5)
    async with asyncio.timeout(2):
        await processor.on_partitions_revoked([tp])
    assert await broker.consumer.committed(tp) is None
    assert (await runtime.read_job(event.job_id)).status == "calling"
    await runtime.expire(event.job_id)
    await recover_jobs(session_factory=runtime.factory, settings=broker.settings)
    assert (await runtime.read_job(event.job_id)).status == "needs_reconciliation"
    assert len(provider.calls) == 1


async def test_global_generation_limit_applies_across_partitions(runtime, broker):
    active = 0
    highest = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def block_call():
        nonlocal active, highest
        active += 1
        highest = max(highest, active)
        if active == 2:
            two_started.set()
        await release.wait()
        active -= 1

    provider = ControlledProvider(on_call=block_call)
    records = []
    for partition in range(4):
        event = await runtime.create_job()
        await broker.producer.send_and_wait(
            broker.settings.kafka_request_topic,
            value=event.model_dump_json().encode(),
            key=b"bounded",
            partition=partition,
        )
        records.append(await broker.receive(broker.settings.kafka_request_topic))
    processor = PartitionProcessor(
        consumer=broker.consumer,
        producer=broker.producer,
        session_factory=runtime.factory,
        settings=broker.settings.model_copy(update={"ai_concurrency": 2}),
        provider_factory=lambda _: provider,
        stop=asyncio.Event(),
    )
    tasks = [
        asyncio.create_task(
            processor.process_partition(
                TopicPartition(record.topic, record.partition),
                [record],
            )
        )
        for record in records
    ]
    try:
        await asyncio.wait_for(two_started.wait(), 5)
        assert active == 2 and len(provider.calls) == 2
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 10)
        assert highest == 2 and len(provider.calls) == 4
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_terminal_dlq_publication_contains_no_business_content(runtime, broker):
    event = await runtime.create_job()
    provider = ControlledProvider(failure=ProviderFailure("provider_refusal"))
    await process_event(
        session_factory=runtime.factory,
        settings=broker.settings,
        event=event,
        provider_factory=lambda _: provider,
    )
    await maintain_jobs(session_factory=runtime.factory, settings=broker.settings)
    assert (await runtime.read_job(event.job_id)).admission_released_at is None
    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=broker.producer,
            batch_size=1,
        )
        == 1
    )
    await maintain_jobs(session_factory=runtime.factory, settings=broker.settings)
    assert (await runtime.read_job(event.job_id)).admission_released_at is None
    assert (
        await dispatch_once(
            session_factory=runtime.factory,
            settings=broker.settings,
            producer=broker.producer,
            batch_size=1,
        )
        == 1
    )
    await maintain_jobs(session_factory=runtime.factory, settings=broker.settings)
    assert (await runtime.read_job(event.job_id)).admission_released_at is not None
    record = await broker.receive(broker.settings.kafka_dead_letter_topic)
    payload = json.loads(record.value)
    assert payload["job_id"] == str(event.job_id)
    assert payload["status"] == "failed" and payload["error_code"] == "provider_refusal"
    assert set(payload) == {
        "version",
        "event_id",
        "job_id",
        "attempt",
        "created_at",
        "status",
        "error_code",
    }
    assert b"4 blue cups" not in record.value and b"Make this clearer" not in record.value


async def test_bounded_producer_abort_closes_real_sender_and_connections(broker, monkeypatch):
    async def hung_flush():
        await asyncio.Event().wait()

    monkeypatch.setattr(kafka, "KAFKA_CLOSE_TIMEOUT", 0.05)
    monkeypatch.setattr(broker.producer, "flush", hung_flush)
    async with asyncio.timeout(2):
        await close_producer(broker.producer)
    assert broker.producer._closed
    assert broker.producer._sender.sender_task.done()


async def test_malformed_offset_commits_only_after_actual_dlq_ack(runtime, broker):
    await broker.producer.send_and_wait(
        broker.settings.kafka_request_topic,
        value=b"private malformed data",
        key=b"bad",
        partition=0,
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    tp = TopicPartition(record.topic, record.partition)
    acknowledged = asyncio.Event()

    class ObserveProducer:
        async def send_and_wait(self, topic, value, *, key):
            result = await broker.producer.send_and_wait(topic, value=value, key=key)
            acknowledged.set()
            return result

    class ObserveConsumer:
        async def commit(self, offsets):
            assert acknowledged.is_set()
            dlq = await broker.receive(broker.settings.kafka_dead_letter_topic)
            assert json.loads(dlq.value)["source_offset"] == record.offset
            await broker.consumer.commit(offsets)

        def assignment(self):
            return broker.consumer.assignment()

        def resume(self, partition):
            broker.consumer.resume(partition)

    processor = PartitionProcessor(
        consumer=ObserveConsumer(),
        producer=ObserveProducer(),
        session_factory=runtime.factory,
        settings=broker.settings,
        provider_factory=lambda _: ControlledProvider(),
        stop=asyncio.Event(),
    )
    await processor.process_partition(tp, [record])
    assert await broker.consumer.committed(tp) == record.offset + 1


@pytest.mark.parametrize(
    "headers",
    [
        None,
        [],
        [("traceparent", b"private-invalid")],
        [("traceparent", b"00-" + b"0" * 32 + b"-" + b"1" * 16 + b"-01")],
        [("traceparent", b"00-" + b"1" * 32 + b"-" + b"0" * 16 + b"-01")],
        [("traceparent", b"01-" + b"1" * 32 + b"-" + b"2" * 16 + b"-01")],
        [("baggage", b"x")] * 9,
        [("baggage", b"x" * 513)],
        [("traceparent", b"00-" + b"1" * 32 + b"-" + b"2" * 16 + b"-01")] * 2,
    ],
)
def test_invalid_or_excessive_trace_metadata_is_dropped(headers):
    assert extract_trace_headers(headers) == {}


def test_trace_extraction_never_forwards_baggage_or_tracestate():
    traceparent = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    assert extract_trace_headers(
        [
            ("traceparent", traceparent.encode()),
            ("baggage", b"private=secret"),
            ("tracestate", b"vendor=private"),
        ]
    ) == {"traceparent": traceparent}


async def test_real_kafka_headers_chain_submit_publish_consume_and_generation(runtime, broker):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.telemetry import Telemetry

    exporter = InMemorySpanExporter()
    telemetry = Telemetry("kafka-propagation-test")
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with telemetry.span("ai.request"):
            root_context = telemetry.capture_context()
            await runtime.create_job(trace_context=root_context)
        assert (
            await dispatch_once(
                session_factory=runtime.factory,
                settings=broker.settings,
                producer=broker.producer,
                telemetry=telemetry,
            )
            == 1
        )
        record = await broker.receive(broker.settings.kafka_request_topic)
        assert set(dict(record.headers)) == {"traceparent"}
        assert (
            extract_trace_headers(record.headers)["traceparent"].split("-")[1]
            == (root_context["traceparent"].split("-")[1])
        )
        assert (
            await handle_record(
                session_factory=runtime.factory,
                settings=broker.settings,
                record=record,
                producer=broker.producer,
                provider_factory=lambda _: ControlledProvider(),
                telemetry=telemetry,
            )
            == HandleResult.HANDLED
        )
        spans = {span.name: span for span in exporter.get_finished_spans()}
        publish, consume, generation = (
            spans["ai.dispatcher.publish"],
            spans["ai.consume"],
            spans["ai.worker.process"],
        )
        assert publish.parent.span_id == spans["ai.request"].context.span_id
        assert consume.parent.span_id == publish.context.span_id
        assert generation.parent.span_id == consume.context.span_id
        assert {span.context.trace_id for span in spans.values()} == {publish.context.trace_id}
    finally:
        telemetry.shutdown()


@pytest.mark.parametrize("invalid_field", ["missing_version", "wrong_version", "naive_time"])
async def test_invalid_event_contract_goes_to_dlq_without_claim(runtime, broker, invalid_field):
    event = await runtime.create_job()
    payload = event.model_dump(mode="json")
    if invalid_field == "missing_version":
        del payload["version"]
    elif invalid_field == "wrong_version":
        payload["version"] = 2
    else:
        payload["created_at"] = "2026-01-01T00:00:00"
    await broker.producer.send_and_wait(
        broker.settings.kafka_request_topic,
        value=json.dumps(payload).encode(),
        key=b"invalid-wire",
    )
    record = await broker.receive(broker.settings.kafka_request_topic)
    provider = ControlledProvider()
    assert (
        await handle_record(
            session_factory=runtime.factory,
            settings=broker.settings,
            record=record,
            producer=broker.producer,
            provider_factory=lambda _: provider,
        )
        == HandleResult.HANDLED
    )
    dlq = await broker.receive(broker.settings.kafka_dead_letter_topic)
    assert json.loads(dlq.value)["error_code"] == "malformed_event"
    assert (await runtime.read_job(event.job_id)).status == "pending"
    assert provider.calls == []


@pytest.mark.parametrize("cleanup_failure", [False, True])
async def test_broker_down_startup_preserves_failure(runtime, monkeypatch, cleanup_failure):
    from aiokafka.errors import KafkaError

    settings = runtime.settings.model_copy(update={"kafka_bootstrap_servers": "127.0.0.1:1"})
    if cleanup_failure:
        original_close = kafka.close_producer

        async def close_then_fail(producer):
            await original_close(producer)
            raise RuntimeError("synthetic cleanup failure")

        monkeypatch.setattr(kafka, "close_producer", close_then_fail)
    async with asyncio.timeout(20):
        with pytest.raises(KafkaError):
            await create_producer(settings)


async def test_real_offset_lag_and_processing_utilization_measurements(runtime, broker):
    from app.telemetry import Telemetry

    observed = []

    class RecordedTelemetry(Telemetry):
        def observe(self, name, value, **attrs):
            observed.append((name, value, attrs))
            super().observe(name, value, **attrs)

    telemetry = RecordedTelemetry("kafka-measurement-test")
    records = []
    for _ in range(2):
        event = await runtime.create_job()
        await broker.producer.send_and_wait(
            broker.settings.kafka_request_topic,
            value=event.model_dump_json().encode(),
            key=b"measurements",
            partition=0,
        )
        records.append(await broker.receive(broker.settings.kafka_request_topic))
    tp = TopicPartition(broker.settings.kafka_request_topic, 0)
    processor = PartitionProcessor(
        consumer=broker.consumer,
        producer=broker.producer,
        session_factory=runtime.factory,
        settings=broker.settings.model_copy(update={"ai_concurrency": 2}),
        provider_factory=lambda _: ControlledProvider(),
        stop=asyncio.Event(),
        telemetry=telemetry,
    )
    try:
        await processor.process_partition(tp, records)
        lag = [(value, attrs) for name, value, attrs in observed if name == "ai.consumer.lag"]
        assert lag == [
            (1, {"topic": tp.topic, "partition": 0}),
            (0, {"topic": tp.topic, "partition": 0}),
        ]
        utilization = [value for name, value, _ in observed if name == "ai.worker.utilization"]
        assert utilization == [0.5, 0, 0.5, 0]
        assert processor.active == 0
    finally:
        telemetry.shutdown()
