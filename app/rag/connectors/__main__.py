"""Connector sync service: ``python -m app.rag.connectors``.

Loads RAG_CONNECTORS_FILE once (invalid config stops the service before any run), then
every RAG_CONNECTOR_POLL_SECONDS runs each due connector. Secrets come from the process
environment by the names the file gives.
"""

import asyncio
import logging
import os

from sqlalchemy.exc import SQLAlchemyError

from app.ai.dispatcher import install_stop_signals, wait_for_tick
from app.db import create_engine, create_session_factory
from app.rag.config import RAGSettings
from app.rag.connectors.config import load_connectors
from app.rag.connectors.sync import run_connector

logger = logging.getLogger("app.rag.connectors")


async def run_service(settings: RAGSettings | None = None) -> None:
    settings = settings or RAGSettings()
    if not settings.rag_enabled:
        raise RuntimeError("Connector sync requires RAG_ENABLED")
    if not settings.rag_connectors_file:
        raise RuntimeError("Connector sync requires RAG_CONNECTORS_FILE")
    connectors = load_connectors(settings.rag_connectors_file)
    stop = asyncio.Event()
    install_stop_signals(stop)
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        while not stop.is_set():
            for connector in connectors:
                if stop.is_set():
                    break
                try:
                    await run_connector(
                        engine=engine,
                        session_factory=session_factory,
                        settings=settings,
                        connector=connector,
                        environ=os.environ,
                    )
                except (SQLAlchemyError, OSError) as exc:
                    # The run row records database_error; the next tick tries again.
                    logger.error(
                        "Connector run failed connector=%s exception=%s",
                        connector.name,
                        type(exc).__name__,
                    )
            await wait_for_tick(stop, settings.rag_connector_poll_seconds)
    finally:
        await engine.dispose()


def main() -> None:
    # Dependency exceptions may embed DSNs, URLs or record text; never print them.
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"Connector sync stopped: {type(exc).__name__}") from None


if __name__ == "__main__":
    main()
