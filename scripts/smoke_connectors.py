"""Connector smoke inside the AI stack's network, with the local fake sources only.

Run it in the connector-sync container, which reaches the database, the API and the
fakes (see README, "Database and API connectors"):

    register a manager (demo residence) -> one postgres and one http_json connector
      -> run each once: 3 documents each, doc.created -> ingested by the workers
      -> change one record and remove one in each fake source -> run again
      -> one new version (ready) and one tombstone per connector
      -> the fakes are put back as seeded

It writes only its own uniquely named connectors' rows. Prints counts, never tokens,
DSNs or record text.
"""

import argparse
import asyncio
import os
import time
from datetime import timedelta
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import asyncpg
import httpx
from sqlalchemy import select, update

from app.db import create_engine, create_session_factory
from app.models import RAGConnectorRun, RAGDocument
from app.rag.config import RAGSettings
from app.rag.connectors.config import Connector, parse_connectors
from app.rag.connectors.sync import run_connector

SOURCE_API = "http://source-api:8080"
ADMIN_DSN_ENV = "SOURCE_DB_ADMIN_DSN"
READY_SECONDS = 120
RECORDS = 3


def _connectors(building_id: UUID, owner_id: UUID, suffix: str) -> list[Connector]:
    common = {
        "building_id": str(building_id),
        "owner_user_id": str(owner_id),
        "interval_seconds": 60,
        "max_records": 50,
    }
    return parse_connectors(
        {
            "connectors": [
                {
                    **common,
                    "name": f"smoke-db-{suffix}",
                    "source": {
                        "kind": "postgres",
                        "dsn_env": "SOURCE_DB_DSN",
                        "query": "SELECT id AS external_id, title, body AS content FROM notices",
                    },
                },
                {
                    **common,
                    "name": f"smoke-api-{suffix}",
                    "source": {
                        "kind": "http_json",
                        "url": f"{SOURCE_API}/records",
                        "allowed_hosts": ["source-api"],
                        "pagination": {"mode": "page"},
                    },
                },
            ]
        }
    )


def _manager(client: httpx.Client) -> tuple[UUID, UUID]:
    """A new account with a demo residence: it manages that building."""
    email = f"connector-smoke-{uuid4().hex}@example.com"
    password = f"synthetic-{uuid4().hex}!"
    registered = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert registered.status_code == 201, f"register: HTTP {registered.status_code}"
    login = client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert login.status_code == 200, f"login: HTTP {login.status_code}"
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    demo = client.post("/api/v1/residence/demo", headers=headers)
    assert demo.status_code == 201, f"demo residence: HTTP {demo.status_code}"
    assert demo.json()["manager"] is True
    return UUID(demo.json()["building"]["id"]), UUID(registered.json()["id"])


async def _run(engine, factory, settings, connector: Connector) -> RAGConnectorRun:
    # The smoke's own connector only: make it due again.
    async with factory() as session:
        await session.execute(
            update(RAGConnectorRun)
            .where(RAGConnectorRun.connector_name == connector.name)
            .values(started_at=RAGConnectorRun.started_at - timedelta(days=1))
        )
        await session.commit()
    run_id = await run_connector(
        engine=engine,
        session_factory=factory,
        settings=settings,
        connector=connector,
        environ=os.environ,
    )
    assert run_id is not None, f"{connector.name}: run skipped"
    async with factory() as session:
        return await session.get(RAGConnectorRun, run_id)


async def _documents(factory, name: str) -> dict[str, RAGDocument]:
    async with factory() as session:
        rows = await session.scalars(select(RAGDocument).where(RAGDocument.connector_name == name))
        return {row.external_id: row for row in rows if row.status != "deleted"}


async def _wait_served(factory, name: str, version_of: dict[str, int]) -> None:
    deadline = time.monotonic() + READY_SECONDS
    while True:
        documents = await _documents(factory, name)
        served = {key: doc.active_version for key, doc in documents.items()}
        failed = [doc.error_code or doc.update_error_code for doc in documents.values()]
        assert not any(failed), f"{name}: ingestion failed {sorted(set(filter(None, failed)))}"
        if served == version_of:
            return
        assert time.monotonic() < deadline, f"{name}: documents not ready in time"
        await asyncio.sleep(0.5)


async def _change_sources(admin_dsn: str, api: httpx.Client) -> None:
    connection = await asyncpg.connect(admin_dsn, timeout=10)
    try:
        await connection.execute("UPDATE notices SET body = body || ' Updated.' WHERE id = 1")
        await connection.execute("DELETE FROM notices WHERE id = 3")
    finally:
        await connection.close()
    changed = {"title": "Water shutoff", "content": "Water is off on Tuesday from 9 to 11."}
    assert api.put("/control/records/notice-1", json=changed).status_code == 200
    assert api.delete("/control/records/notice-3").status_code == 204


async def _restore_sources(admin_dsn: str, api: httpx.Client) -> None:
    connection = await asyncpg.connect(admin_dsn, timeout=10)
    try:
        await connection.execute(
            "UPDATE notices SET body = 'Water is off on Monday from 9 to 11 for pipe repairs.' "
            "WHERE id = 1"
        )
        await connection.execute(
            "INSERT INTO notices (id, title, body) VALUES "
            "(3, 'Recycling', 'Glass goes in the green bin behind block B.') "
            "ON CONFLICT (id) DO NOTHING"
        )
    finally:
        await connection.close()
    seeded = {
        "notice-1": ("Water shutoff", "Water is off on Monday from 9 to 11."),
        "notice-3": ("Recycling", "Glass goes in the green bin."),
    }
    for key, (title, content) in seeded.items():
        api.put(f"/control/records/{key}", json={"title": title, "content": content})


async def run_smoke(base_url: str) -> None:
    if urlsplit(base_url).hostname != "api":
        raise ValueError("run inside the AI stack against http://api:8000")
    admin_dsn = os.environ.get(ADMIN_DSN_ENV)
    if not admin_dsn:
        raise ValueError(f"{ADMIN_DSN_ENV} is required (the local fake's admin DSN)")
    settings = RAGSettings()
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    try:
        with (
            httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client,
            httpx.Client(base_url=SOURCE_API, timeout=10, trust_env=False) as api,
        ):
            building_id, owner_id = _manager(client)
            connectors = _connectors(building_id, owner_id, uuid4().hex[:8])

            # First runs: every record becomes a document, served once ingested.
            for connector in connectors:
                run = await _run(engine, factory, settings, connector)
                assert (run.status, run.created) == ("complete", RECORDS), (
                    f"{connector.name}: {run.status} {run.error_code} created={run.created}"
                )
            for connector in connectors:
                ids = await _documents(factory, connector.name)
                await _wait_served(factory, connector.name, dict.fromkeys(ids, 1))

            # Change one record and remove one in each source, then run again.
            await _change_sources(admin_dsn, api)
            try:
                for connector in connectors:
                    run = await _run(engine, factory, settings, connector)
                    assert (run.status, run.updated, run.deleted, run.unchanged) == (
                        "complete",
                        1,
                        1,
                        1,
                    ), f"{connector.name}: {run.status} {run.error_code}"
                for connector in connectors:
                    ids = sorted(await _documents(factory, connector.name))
                    assert len(ids) == RECORDS - 1
                    expected = {key: 2 if key in {"1", "notice-1"} else 1 for key in ids}
                    await _wait_served(factory, connector.name, expected)
            finally:
                await _restore_sources(admin_dsn, api)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://api:8000")
    args = parser.parse_args()
    try:
        asyncio.run(run_smoke(args.base_url))
    except ValueError as error:
        parser.error(str(error))
    except (httpx.HTTPError, OSError, asyncpg.PostgresError):
        raise SystemExit("Connector smoke transport failed (details withheld)") from None
    print("PASS: postgres and http_json connectors created, updated and deleted documents")


if __name__ == "__main__":
    main()
