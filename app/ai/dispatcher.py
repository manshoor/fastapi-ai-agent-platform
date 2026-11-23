"""Run with ``python -m app.ai.dispatcher``; no API-process background loops."""

import asyncio
import logging
import signal

from app.ai.config import AISettings
from app.ai.kafka import close_producer, create_producer, dispatch_once
from app.ai.processing import maintain_jobs, recover_jobs
from app.assistant.memory import purge_tick
from app.db import create_engine, create_session_factory
from app.rag import ingestion, outbox
from app.rag.config import RAGSettings
from app.telemetry import configure_telemetry


def install_stop_signals(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)


async def wait_for_tick(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        pass


async def run_dispatcher(settings: AISettings | None = None) -> None:
    settings = settings or AISettings()
    if not settings.ai_enabled:
        raise RuntimeError("AI dispatcher requires AI_ENABLED")
    stop = asyncio.Event()
    install_stop_signals(stop)
    telemetry = configure_telemetry("item-rewrite-dispatcher")
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    producer = None
    # RAG settings add only rag_* fields; the database and Kafka values are shared.
    rag_settings = RAGSettings(**settings.model_dump())
    try:
        producer = await create_producer(settings)
        while not stop.is_set():
            if rag_settings.rag_enabled:
                await ingestion.recover_documents(
                    session_factory=session_factory,
                    settings=rag_settings,
                    telemetry=telemetry,
                )
                await outbox.dispatch_once(
                    session_factory=session_factory,
                    settings=rag_settings,
                    producer=producer,
                    telemetry=telemetry,
                    stop=stop,
                )
            await recover_jobs(
                session_factory=session_factory,
                settings=settings,
                telemetry=telemetry,
            )
            # Expired and moved-out memories are never served; this bounds their storage.
            await purge_tick(session_factory)
            await maintain_jobs(
                session_factory=session_factory,
                settings=settings,
                telemetry=telemetry,
            )
            # Small batches give shutdown and recovery a bounded turn even under saturation.
            await dispatch_once(
                session_factory=session_factory,
                settings=settings,
                producer=producer,
                telemetry=telemetry,
                batch_size=10,
                stop=stop,
            )
            await maintain_jobs(
                session_factory=session_factory,
                settings=settings,
                telemetry=telemetry,
            )
            await wait_for_tick(stop, settings.ai_poll_seconds)
    finally:
        try:
            if producer is not None:
                await close_producer(producer)
        finally:
            await engine.dispose()
            await asyncio.to_thread(telemetry.shutdown)


def main() -> None:
    # Dependency exceptions may embed URLs or payloads; never print their messages/tracebacks.
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(run_dispatcher())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit("AI dispatcher stopped after a dependency failure") from None


if __name__ == "__main__":
    main()
