"""Document ingestion consumer: ``python -m app.rag.worker``.

Reuses the rewrite worker's partition-ordered processor with a RAG record handler.
"""

import asyncio
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer

from app.ai.dispatcher import install_stop_signals
from app.ai.kafka import close_producer, create_producer
from app.ai.processing import HandleResult
from app.ai.provider import OpenAIEmbedder
from app.ai.worker import PARTITION_COUNT, PartitionProcessor, consume
from app.db import create_engine, create_session_factory
from app.rag.config import RAGSettings
from app.rag.ingestion import graph_extractor
from app.rag.outbox import handle_record
from app.telemetry import configure_telemetry


def _consumer(settings: RAGSettings) -> AIOKafkaConsumer:
    # Same bounds as the rewrite worker; a lease renews per batch, well inside the poll interval.
    return AIOKafkaConsumer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_rag_group_id,
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
        max_poll_interval_ms=300000,
        metadata_max_age_ms=30000,
    )


async def run_worker(settings: RAGSettings | None = None) -> None:
    settings = settings or RAGSettings()
    if not settings.rag_enabled:
        raise RuntimeError("RAG worker requires RAG_ENABLED")
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise RuntimeError("RAG worker requires OPENAI_API_KEY")
    stop = asyncio.Event()
    install_stop_signals(stop)
    telemetry = configure_telemetry("rag-ingest-worker")
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    producer = None
    consumer = None
    processor = None
    embedder = None
    try:
        embedder = OpenAIEmbedder(settings, telemetry)
        extractor = graph_extractor(settings, telemetry, embedder)
        producer = await create_producer(settings)
        consumer = _consumer(settings)

        async def handler(*, record: Any, **_: Any) -> HandleResult:
            return await handle_record(
                session_factory=session_factory,
                settings=settings,
                record=record,
                producer=producer,
                embedder=embedder,
                extractor=extractor,
                telemetry=telemetry,
            )

        processor = PartitionProcessor(
            consumer=consumer,
            producer=producer,
            session_factory=session_factory,
            settings=settings,
            provider_factory=lambda pinned: embedder,
            stop=stop,
            telemetry=telemetry,
            handler=handler,
            topic=settings.kafka_rag_event_topic,
        )
        consumer.subscribe([settings.kafka_rag_event_topic], listener=processor)
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
                        if embedder is not None:
                            await embedder.aclose()
                    finally:
                        await engine.dispose()
                        await asyncio.to_thread(telemetry.shutdown)


def main() -> None:
    # Dependency exceptions may embed URLs or payloads; never print their messages.
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit("RAG worker stopped after a dependency failure") from None


if __name__ == "__main__":
    main()
