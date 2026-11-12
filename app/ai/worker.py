"""Bounded, partition-ordered Kafka consumer: ``python -m app.ai.worker``."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener, TopicPartition
from aiokafka.errors import KafkaError

from app.ai.config import AISettings
from app.ai.dispatcher import install_stop_signals, wait_for_tick
from app.ai.kafka import close_producer, create_producer, handle_record
from app.ai.processing import HandleResult, ProviderFactory, SessionFactory
from app.ai.provider import OpenAIRewriter
from app.db import create_engine, create_session_factory
from app.telemetry import Telemetry, configure_telemetry

DRAIN_SECONDS = 5
PARTITION_COUNT = 6


class PartitionProcessor(ConsumerRebalanceListener):
    """At most one bounded batch per partition; only contiguous durable offsets commit."""

    def __init__(
        self,
        *,
        consumer: Any,
        producer: Any,
        session_factory: SessionFactory,
        settings: AISettings,
        provider_factory: ProviderFactory,
        stop: asyncio.Event,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.consumer = consumer
        self.producer = producer
        self.session_factory = session_factory
        self.settings = settings
        self.provider_factory = provider_factory
        self.stop = stop
        self.telemetry = telemetry
        self.semaphore = asyncio.Semaphore(settings.ai_concurrency)
        self.commit_lock = asyncio.Lock()
        self.tasks: dict[TopicPartition, asyncio.Task] = {}
        self.revoking: set[TopicPartition] = set()
        self.active = 0

    def observe_utilization(self) -> None:
        if self.telemetry is not None:
            self.telemetry.observe(
                "ai.worker.utilization",
                self.active / self.settings.ai_concurrency,
                topic=self.settings.kafka_request_topic,
            )

    async def observe_lag(self, tp: TopicPartition) -> None:
        if self.telemetry is None:
            return
        try:
            async with asyncio.timeout(5):
                committed = await self.consumer.committed(tp)
        except (KafkaError, TimeoutError):
            return  # An unavailable measurement is not a fabricated zero or a write retry.
        highwater = self.consumer.highwater(tp)
        if highwater is not None and committed is not None:
            self.telemetry.observe(
                "ai.consumer.lag",
                max(0, highwater - committed),
                topic=tp.topic,
                partition=tp.partition,
            )

    async def on_partitions_revoked(self, revoked) -> None:
        self.revoking.update(revoked)
        await self.drain(revoked)

    async def on_partitions_assigned(self, assigned) -> None:
        self.revoking.difference_update(assigned)

    async def drain(self, partitions=None) -> None:
        selected = set(self.tasks) if partitions is None else set(partitions)
        tasks = [self.tasks[tp] for tp in selected if tp in self.tasks]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=DRAIN_SECONDS)
            for task in pending:
                task.cancel()
            # Provider cleanup has its own five-second bound. Calling remains unknown if
            # cancellation interrupted network I/O or result persistence.
            await asyncio.gather(*tasks, return_exceptions=True)
        for tp in selected:
            self.tasks.pop(tp, None)

    async def process_partition(self, tp: TopicPartition, records: Sequence[Any]) -> None:
        try:
            for record in records:
                while not self.stop.is_set() and tp not in self.revoking:
                    async with self.semaphore:
                        if self.stop.is_set() or tp in self.revoking:
                            return
                        self.active += 1
                        self.observe_utilization()
                        try:
                            result = await handle_record(
                                session_factory=self.session_factory,
                                settings=self.settings,
                                record=record,
                                producer=self.producer,
                                provider_factory=self.provider_factory,
                                telemetry=self.telemetry,
                            )
                        finally:
                            self.active -= 1
                            self.observe_utilization()
                    if result == HandleResult.DEFERRED:
                        await wait_for_tick(self.stop, self.settings.ai_poll_seconds)
                        continue
                    if tp in self.revoking:
                        return
                    # Offset + 1 only, never the consumer's already-prefetched position.
                    async with self.commit_lock:
                        async with asyncio.timeout(10):
                            await self.consumer.commit({tp: record.offset + 1})
                    await self.observe_lag(tp)
                    break
                else:
                    return
        finally:
            if tp not in self.revoking and tp in self.consumer.assignment():
                self.consumer.resume(tp)

    def submit(self, tp: TopicPartition, records: Sequence[Any]) -> None:
        if tp in self.tasks:
            previous = self.tasks[tp]
            if not previous.done():
                raise RuntimeError("Concurrent partition processing is forbidden")
            previous.result()
            del self.tasks[tp]
        self.consumer.pause(tp)
        self.tasks[tp] = asyncio.create_task(self.process_partition(tp, records))

    def reap(self) -> None:
        for tp, task in tuple(self.tasks.items()):
            if task.done():
                del self.tasks[tp]
                task.result()  # Fail closed on any unknown DB/offset outcome; no write retry.


async def consume(
    *,
    consumer: Any,
    processor: PartitionProcessor,
    stop: asyncio.Event,
) -> None:
    """Testable consumer loop; Kafka prefetch and per-poll records are explicitly finite."""
    while not stop.is_set():
        processor.reap()
        batches = await consumer.getmany(timeout_ms=250, max_records=PARTITION_COUNT)
        for tp, records in batches.items():
            if records and tp not in processor.revoking:
                processor.submit(tp, records)


async def run_worker(settings: AISettings | None = None) -> None:
    settings = settings or AISettings()
    if not settings.ai_enabled:
        raise RuntimeError("AI worker requires AI_ENABLED")
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise RuntimeError("AI worker requires OPENAI_API_KEY")
    stop = asyncio.Event()
    install_stop_signals(stop)
    telemetry = configure_telemetry("item-rewrite-worker")
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    producer = None
    consumer = None
    processor = None
    shared_provider = None
    try:
        shared_provider = OpenAIRewriter(settings, telemetry)
        producer = await create_producer(settings)
        consumer = AIOKafkaConsumer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_records=PARTITION_COUNT,
            fetch_max_bytes=262144,
            max_partition_fetch_bytes=65536,
            request_timeout_ms=10000,
            retry_backoff_ms=250,
            session_timeout_ms=10000,
            heartbeat_interval_ms=2500,
            rebalance_timeout_ms=30000,
            max_poll_interval_ms=120000,
            metadata_max_age_ms=30000,
        )
        processor = PartitionProcessor(
            consumer=consumer,
            producer=producer,
            session_factory=session_factory,
            settings=settings,
            provider_factory=lambda pinned: OpenAIRewriter(
                pinned,
                telemetry,
                client=shared_provider.client,
            ),
            stop=stop,
            telemetry=telemetry,
        )
        consumer.subscribe([settings.kafka_request_topic], listener=processor)
        async with asyncio.timeout(15):
            await consumer.start()
        await consume(consumer=consumer, processor=processor, stop=stop)
    finally:
        stop.set()
        try:
            if processor is not None:
                await processor.drain()
        finally:
            try:
                if consumer is not None:
                    async with asyncio.timeout(10):
                        await consumer.stop()
            finally:
                try:
                    if producer is not None:
                        await close_producer(producer)
                finally:
                    try:
                        if shared_provider is not None:
                            async with asyncio.timeout(5):
                                await shared_provider.aclose()
                    finally:
                        await engine.dispose()
                        await asyncio.to_thread(telemetry.shutdown)


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit("AI worker stopped after a dependency failure") from None


if __name__ == "__main__":
    main()
