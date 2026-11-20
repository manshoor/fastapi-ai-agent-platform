"""Connector config, sources and sync against real PostgreSQL and the fake JSON source."""

import asyncio
from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select, text, update

from app.ai.processing import HandleResult
from app.models import Building, Membership, RAGConnectorRun, RAGDocument, RAGOutbox
from app.rag.connectors import sources
from app.rag.connectors.config import (
    Connector,
    ConnectorConfigError,
    load_connectors,
    parse_connectors,
)
from app.rag.connectors.sources import Fetch, Record, fetch_http, fetch_postgres
from app.rag.connectors.sync import lock_key, run_connector
from app.rag.contracts import ContentType, IngestError
from app.rag.documents import delete_document, list_documents, update_document
from app.rag.uploads import Upload
from tests.support.source_api import make_source_app
from tests.test_rag_ingestion import ControlledEmbedder, rag_runtime  # noqa: F401

SCHEMA = "connector_fixture"
DSN_ENV = "SOURCE_DSN"
TOKEN_ENV = "SOURCE_TOKEN"
SECRET_DSN = "postgresql://reader:hunter2@source-db:5432/notices"


# --- Fixtures --------------------------------------------------------------------------


@pytest.fixture
async def building(app, runtime):
    """A building the runtime's owner manages."""
    building_id = uuid4()
    async with app.state.session_factory() as session:
        session.add(Building(id=building_id, name="Connector House"))
        await session.flush()
        session.add(Membership(user_id=runtime.owner_id, building_id=building_id, role="manager"))
        await session.commit()
    return building_id


@pytest.fixture
def connector(runtime, building):
    def make(source: dict, *, name: str = "notices", max_records: int = 100) -> Connector:
        raw = {
            "connectors": [
                {
                    "name": name,
                    "building_id": str(building),
                    "owner_user_id": str(runtime.owner_id),
                    "interval_seconds": 60,
                    "max_records": max_records,
                    "source": source,
                }
            ]
        }
        return parse_connectors(raw)[0]

    return make


@pytest.fixture
def plain(connector):
    """A connector whose fetches the tests supply directly."""
    return connector({"kind": "http_json", "url": "https://source.example/records"})


@pytest.fixture
def sync(app, runtime):
    """Run a connector once, due now: earlier runs are moved back past the interval."""

    async def run(connector: Connector, fetch: Fetch | None = None, fetcher=None):
        async with runtime.factory() as session:
            await session.execute(
                update(RAGConnectorRun)
                .where(RAGConnectorRun.connector_name == connector.name)
                .values(started_at=RAGConnectorRun.started_at - timedelta(days=1))
            )
            await session.commit()

        async def fixed(_connector, _environ):
            return fetch

        run_id = await run_connector(
            engine=app.state.engine,
            session_factory=runtime.factory,
            settings=runtime.settings,
            connector=connector,
            environ={},
            fetcher=fetcher or fixed,
        )
        if run_id is None:
            return None
        async with runtime.factory() as session:
            return await session.get(RAGConnectorRun, run_id)

    return run


@pytest.fixture
async def source_db(app, settings):
    """A scratch schema in app_test standing in for an external database."""
    async with app.state.engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        await connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))
        await connection.execute(
            text(f"CREATE TABLE {SCHEMA}.notices (id int PRIMARY KEY, title text, body text)")
        )
        await connection.execute(
            text(
                f"INSERT INTO {SCHEMA}.notices VALUES "
                "(1, 'Water', 'Water is off on Monday.'), (2, 'Lift', 'The lift is serviced.')"
            )
        )
    url = settings.sqlalchemy_url.set(drivername="postgresql")
    yield {DSN_ENV: url.render_as_string(hide_password=False)}
    async with app.state.engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


async def documents(runtime, name: str = "notices") -> dict[str, RAGDocument]:
    async with runtime.factory() as session:
        rows = await session.scalars(select(RAGDocument).where(RAGDocument.connector_name == name))
        return {row.external_id: row for row in rows}


async def events(runtime, document_id) -> list[tuple[str, int]]:
    async with runtime.factory() as session:
        rows = await session.execute(
            select(RAGOutbox.event_kind, RAGOutbox.version)
            .where(RAGOutbox.document_id == document_id)
            .order_by(RAGOutbox.created_at)
        )
        return [tuple(row) for row in rows]


async def make_ready(runtime, document_id, version: int = 1) -> None:
    event = await runtime.event_for(document_id, version=version)
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED


def records(*pairs: tuple[str, str]) -> Fetch:
    return Fetch(
        records=[Record(key, f"Title {key}", content) for key, content in pairs], complete=True
    )


# --- Config ----------------------------------------------------------------------------


def config(source: dict, **fields) -> dict:
    connector = {
        "name": "notices",
        "building_id": str(uuid4()),
        "owner_user_id": str(uuid4()),
        "interval_seconds": 60,
        "max_records": 10,
        "source": source,
    }
    return {"connectors": [connector | fields]}


POSTGRES = {"kind": "postgres", "dsn_env": DSN_ENV, "query": "SELECT 1"}


@pytest.mark.parametrize(
    "raw",
    [
        config(POSTGRES, extra="field"),
        config(POSTGRES | {"password": "x"}),
        config(POSTGRES | {"dsn_env": SECRET_DSN}),
        config(POSTGRES | {"query": "SELECT 1; DROP TABLE users"}),
        config(POSTGRES | {"query": "DELETE FROM users"}),
        config(POSTGRES | {"query": "SELECT 1 AS external_id -- cut"}),
        config(POSTGRES | {"query": "SELECT 1 /* cut */ AS external_id"}),
        config(POSTGRES | {SECRET_DSN: "x"}),
        config({"kind": "http_json", "url": "https://source.example:abc/x"}),
        config(POSTGRES, interval_seconds=10),
        config(POSTGRES, max_records=5001),
        config(POSTGRES, name="Bad Name"),
        config({"kind": "http_json", "url": "http://source.example/x"}),
        config({"kind": "http_json", "url": "https://user:pass@source.example/x"}),
        config({"kind": "http_json", "url": "ftp://source.example/x"}),
        config({"kind": "http_json", "url": "https://x", "auth_header_env": "Bearer abc"}),
        config({"kind": "http_json", "url": "https://x", "pagination": {"mode": "cursor"}}),
        config({"kind": "http_json", "url": "https://x", "max_pages": 101}),
        config({"kind": "smb", "path": "x"}),
        {"connectors": config(POSTGRES)["connectors"] * 2},
    ],
)
def test_config_rejects(raw):
    with pytest.raises(ConnectorConfigError) as caught:
        parse_connectors(raw)
    # Field locations only: a secret in the wrong field never reaches the message.
    message = str(caught.value)
    assert "hunter2" not in message and "Bearer" not in message and "pass@" not in message


def test_config_accepts_plain_http_only_to_listed_hosts():
    source = {
        "kind": "http_json",
        "url": "http://source-api:8080/r",
        "allowed_hosts": ["source-api"],
    }
    assert parse_connectors(config(source))[0].source.url == "http://source-api:8080/r"


def test_config_file_errors_name_no_values(tmp_path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConnectorConfigError, match="cannot be read"):
        load_connectors(str(missing))
    broken = tmp_path / "broken.yaml"
    broken.write_text(f"connectors: [{{name: {SECRET_DSN}", encoding="utf-8")
    with pytest.raises(ConnectorConfigError) as caught:
        load_connectors(str(broken))
    assert "hunter2" not in str(caught.value)


# --- PostgreSQL source -----------------------------------------------------------------


def pg_source(query: str) -> dict:
    return {"kind": "postgres", "dsn_env": DSN_ENV, "query": query}


NOTICES = f"SELECT id AS external_id, title, body AS content FROM {SCHEMA}.notices"


async def test_postgres_source_reads_rows(connector, source_db):
    source = connector(pg_source(NOTICES)).source
    fetch = await fetch_postgres(source, 10, source_db)
    assert fetch.complete and fetch.error_code is None
    assert {record.external_id: record.title for record in fetch.records} == {
        "1": "Water",
        "2": "Lift",
    }


async def test_postgres_source_refuses_writes(app, connector, source_db):
    writing = (
        f"WITH gone AS (DELETE FROM {SCHEMA}.notices RETURNING id) "
        "SELECT id AS external_id, 't' AS title, 'c' AS content FROM gone"
    )
    fetch = await fetch_postgres(connector(pg_source(writing)).source, 10, source_db)
    assert (fetch.complete, fetch.error_code, fetch.records) == (False, "source_query_failed", [])
    async with app.state.engine.connect() as connection:
        assert await connection.scalar(text(f"SELECT count(*) FROM {SCHEMA}.notices")) == 2


async def test_postgres_source_caps_and_ids(connector, source_db):
    source = connector(pg_source(NOTICES)).source
    capped = await fetch_postgres(source, 1, source_db)
    assert (capped.complete, capped.error_code, len(capped.records)) == (False, "record_limit", 1)
    null_ids = connector(
        pg_source(f"SELECT NULL AS external_id, title, body AS content FROM {SCHEMA}.notices")
    ).source
    invalid = await fetch_postgres(null_ids, 10, source_db)
    assert (invalid.complete, invalid.error_code) == (False, "invalid_record_id")


async def test_postgres_source_fails_closed(connector, source_db):
    source = connector(pg_source(NOTICES)).source
    assert (await fetch_postgres(source, 10, {})).error_code == "dsn_missing"
    bad = {DSN_ENV: "not a dsn"}
    assert (await fetch_postgres(source, 10, bad)).error_code == "source_unreachable"
    missing = connector(pg_source(f"SELECT 1 AS external_id FROM {SCHEMA}.nothing")).source
    assert (await fetch_postgres(missing, 10, source_db)).error_code == "source_query_failed"


# --- HTTP source -----------------------------------------------------------------------


@pytest.fixture
def source_api():
    return make_source_app()


def http_source(connector, path: str, **fields):
    return connector({"kind": "http_json", "url": f"https://source.example{path}", **fields}).source


async def fetch(source, api, max_records: int = 100, environ=None) -> Fetch:
    transport = httpx.ASGITransport(app=api)
    return await fetch_http(source, max_records, environ or {}, transport=transport)


async def test_http_source_pages(connector, source_api):
    paged = http_source(connector, "/records", pagination={"mode": "page"})
    result = await fetch(paged, source_api)
    assert result.complete and [r.external_id for r in result.records] == [
        "notice-1",
        "notice-2",
        "notice-3",
    ]
    # Page 3 came back empty, which ends the fetch.
    assert [query for _, query, _ in source_api.state.requests] == [
        {"page": "1"},
        {"page": "2"},
        {"page": "3"},
    ]


async def test_http_source_follows_cursor(connector, source_api):
    paging = {"mode": "cursor", "cursor_path": "next"}
    cursor = http_source(connector, "/cursor", items_path="data", pagination=paging)
    result = await fetch(cursor, source_api)
    assert result.complete and len(result.records) == 3
    assert all(path == "/cursor" for path, _, _ in source_api.state.requests)


async def test_http_source_incomplete_at_caps(connector, source_api):
    paged = http_source(connector, "/records", pagination={"mode": "page"}, max_pages=1)
    pages = await fetch(paged, source_api)
    assert (pages.complete, pages.error_code, len(pages.records)) == (False, "page_limit", 2)
    everything = http_source(connector, "/records", pagination={"mode": "page"})
    capped = await fetch(everything, source_api, max_records=2)
    assert (capped.complete, capped.error_code) == (False, "record_limit")


async def test_http_source_sends_auth_from_environment(connector, source_api):
    source = http_source(connector, "/records", auth_header_env=TOKEN_ENV)
    assert (await fetch(source, source_api)).error_code == "auth_missing"
    assert not source_api.state.requests
    result = await fetch(source, source_api, environ={TOKEN_ENV: "Bearer test-token"})
    assert result.complete
    assert source_api.state.requests[-1][2] == "Bearer test-token"


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("/redirect", "source_http_status"),
        ("/status/500", "source_http_status"),
        ("/status/204", "source_http_status"),
        ("/large", "source_too_large"),
        ("/gzip", "source_too_large"),
        ("/cursor", "source_malformed"),
    ],
)
async def test_http_source_fails_closed(connector, source_api, path, code):
    # /cursor answers an object: with no items_path there is no list, never "zero records".
    result = await fetch(http_source(connector, path), source_api)
    assert (result.complete, result.error_code, result.records) == (False, code, [])
    # No redirect was followed.
    assert [request[0] for request in source_api.state.requests] == [path]


async def test_http_source_keeps_the_url_query(connector, source_api):
    source = http_source(connector, "/records?building=5", pagination={"mode": "page"})
    assert (await fetch(source, source_api)).complete
    assert source_api.state.requests[0][1] == {"building": "5", "page": "1"}


async def test_http_source_missing_cursor_key_fails(connector, source_api):
    paging = {"mode": "cursor", "cursor_path": "next_cursor"}
    cursor = http_source(connector, "/cursor", items_path="data", pagination=paging)
    result = await fetch(cursor, source_api)
    assert (result.complete, result.error_code, result.records) == (False, "source_malformed", [])


@pytest.mark.parametrize("path", ["/brotli", "/gzip-bomb"])
async def test_http_source_bounds_decoding(connector, source_api, path):
    # Brotli is refused before decoding; gzip inflates only up to the cap.
    result = await fetch(http_source(connector, path), source_api)
    assert result.error_code in {"source_malformed", "source_too_large"} and not result.records


async def test_http_source_refuses_unencodable_ids(connector, source_api):
    result = await fetch(http_source(connector, "/surrogate"), source_api)
    assert (result.complete, result.error_code, len(result.records)) == (
        False,
        "invalid_record_id",
        1,
    )


async def test_http_source_invalid_url_is_a_code(connector, source_api):
    source = http_source(connector, "/records").model_copy(update={"url": "https://x:abc/"})
    assert (await fetch(source, source_api)).error_code == "source_unreachable"


async def test_postgres_source_bounds_bytes(app, connector, source_db, monkeypatch):
    async with app.state.engine.begin() as connection:
        await connection.execute(
            text(f"INSERT INTO {SCHEMA}.notices VALUES (3, 'Big', repeat('x', 6 * 1024 * 1024))")
        )
    source = connector(pg_source(NOTICES)).source
    result = await fetch_postgres(source, 10, source_db)
    # The oversize row arrives without content: seen, then skipped by sync.
    assert result.complete
    assert {r.external_id: r.content is None for r in result.records}["3"] is True
    monkeypatch.setattr(sources, "MAX_FETCH_BYTES", 10)
    capped = await fetch_postgres(source, 10, source_db)
    assert (capped.complete, capped.error_code, capped.records) == (
        False,
        "source_too_large",
        [],
    )


# --- Sync ------------------------------------------------------------------------------


async def test_sync_creates_updates_and_deletes(runtime, plain, sync):
    run = await sync(plain, records(("a", "Alpha text."), ("b", "Beta text.")))
    assert (run.status, run.fetched, run.created, run.error_code) == ("complete", 2, 2, None)
    created = await documents(runtime)
    assert set(created) == {"a", "b"}
    first = created["a"]
    assert (first.owner_id, first.source_kind, first.status) == (
        runtime.owner_id,
        "connector",
        "pending",
    )
    assert await events(runtime, first.id) == [("created", 1)]
    for document in created.values():
        await make_ready(runtime, document.id)

    run = await sync(plain, records(("a", "Alpha text, changed."), ("b", "Beta text.")))
    assert (run.status, run.updated, run.unchanged, run.deleted) == ("complete", 1, 1, 0)
    changed = (await documents(runtime))["a"]
    assert (changed.version, changed.active_version, changed.status) == (2, 1, "pending")
    assert await events(runtime, first.id) == [("created", 1), ("updated", 2)]

    run = await sync(plain, records(("a", "Alpha text, changed.")))
    assert (run.status, run.deleted) == ("complete", 1)
    gone = (await documents(runtime))["b"]
    assert gone.status == "deleted"
    assert (await events(runtime, gone.id))[-1] == ("deleted", 1)


async def test_sync_recreates_a_deleted_record_id(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    await sync(plain, records(("b", "Beta.")))
    # The tombstone frees the record id: the partial unique index skips deleted rows.
    run = await sync(plain, records(("a", "Alpha again.")))
    assert run.created == 1
    async with runtime.factory() as session:
        statuses = await session.scalars(
            select(RAGDocument.status).where(RAGDocument.external_id == "a")
        )
        assert sorted(statuses) == ["deleted", "pending"]


@pytest.mark.parametrize("code", ["record_limit", "page_limit", "invalid_record_id"])
async def test_partial_fetch_never_deletes(runtime, plain, sync, code):
    await sync(plain, records(("a", "Alpha."), ("b", "Beta.")))
    partial = Fetch(records=[Record("a", "A", "Alpha.")], complete=False, error_code=code)
    run = await sync(plain, partial)
    assert (run.status, run.error_code, run.deleted) == ("partial", code, 0)
    assert {doc.status for doc in (await documents(runtime)).values()} == {"pending"}


async def test_empty_fetch_deletes_nothing(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    run = await sync(plain, records())
    assert (run.status, run.error_code, run.deleted) == ("partial", "empty_source", 0)
    assert (await documents(runtime))["a"].status == "pending"


async def test_duplicate_ids_apply_the_first_only(runtime, plain, sync):
    twice = records(("a", "Alpha one."), ("a", "Alpha two."))
    run = await sync(plain, twice)
    assert (run.created, run.skipped) == (1, 1)
    await make_ready(runtime, (await documents(runtime))["a"].id)
    run = await sync(plain, twice)
    assert (run.unchanged, run.updated, run.skipped) == (1, 0, 1)
    assert (await documents(runtime))["a"].version == 1


async def test_moved_connector_removes_old_building_documents(runtime, connector, sync):
    old = connector({"kind": "http_json", "url": "https://source.example/records"})
    await sync(old, records(("a", "Alpha.")))
    building_id = uuid4()
    async with runtime.factory() as session:
        session.add(Building(id=building_id, name="Second House"))
        await session.flush()
        session.add(Membership(user_id=runtime.owner_id, building_id=building_id, role="manager"))
        await session.commit()
    moved = old.model_copy(update={"building_id": building_id})
    run = await sync(moved, records(("a", "Alpha.")))
    assert (run.status, run.deleted, run.created) == ("complete", 1, 1)
    async with runtime.factory() as session:
        rows = (
            await session.execute(
                select(RAGDocument.building_id, RAGDocument.status).where(
                    RAGDocument.external_id == "a"
                )
            )
        ).all()
    assert sorted(rows, key=lambda row: row.status) == [
        (old.building_id, "deleted"),
        (building_id, "pending"),
    ]


async def test_failed_fetch_applies_nothing(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    run = await sync(plain, Fetch(error_code="source_unreachable"))
    assert (run.status, run.error_code, run.created, run.deleted) == (
        "failed",
        "source_unreachable",
        0,
        0,
    )
    assert (await documents(runtime))["a"].status == "pending"


async def test_skipped_records_are_seen_not_deleted(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    bad = Fetch(
        records=[Record("a", "A", "   "), Record("b", "B", None), Record("c", "", "Text.")],
        complete=True,
    )
    run = await sync(plain, bad)
    assert (run.status, run.skipped, run.deleted, run.created) == ("complete", 3, 0, 0)
    assert set(await documents(runtime)) == {"a"}


async def test_owner_must_manage_the_building(runtime, building, plain, sync):
    async with runtime.factory() as session:
        await session.execute(delete(Membership).where(Membership.building_id == building))
        await session.commit()
    run = await sync(plain, records(("a", "Alpha.")))
    assert (run.status, run.error_code, run.created) == ("failed", "owner_not_manager", 0)
    assert await documents(runtime) == {}


async def test_advisory_lock_stops_an_overlapping_run(app, plain, sync):
    async with app.state.engine.connect() as other:
        await other.scalar(select(func.pg_advisory_lock(lock_key(plain.name))))
        await other.commit()
        assert await sync(plain, records(("a", "Alpha."))) is None
        await other.scalar(select(func.pg_advisory_unlock(lock_key(plain.name))))
        await other.commit()
    assert (await sync(plain, records(("a", "Alpha.")))).status == "complete"


async def test_concurrent_runs_apply_once(runtime, plain, sync):
    release = asyncio.Event()

    async def slow(_connector, _environ):
        await release.wait()
        return records(("a", "Alpha."))

    first = asyncio.create_task(sync(plain, fetcher=slow))
    await asyncio.sleep(0)
    while not await _running(runtime, plain.name):
        await asyncio.sleep(0.01)
    assert await sync(plain, fetcher=slow) is None
    release.set()
    assert (await first).created == 1


async def _running(runtime, name: str) -> bool:
    async with runtime.factory() as session:
        status = await session.scalar(
            select(RAGConnectorRun.status).where(RAGConnectorRun.connector_name == name)
        )
        return status == "running"


async def test_run_waits_for_its_interval(app, runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    again = await run_connector(
        engine=app.state.engine,
        session_factory=runtime.factory,
        settings=runtime.settings,
        connector=plain,
        environ={},
        fetcher=None,
    )
    assert again is None


async def test_stale_running_row_is_closed(runtime, plain, sync):
    async with runtime.factory() as session:
        session.add(
            RAGConnectorRun(
                id=uuid4(),
                connector_name=plain.name,
                started_at=func.clock_timestamp() - timedelta(days=2),
            )
        )
        await session.commit()
    run = await sync(plain, records(("a", "Alpha.")))
    assert run.status == "complete"
    async with runtime.factory() as session:
        stale = await session.scalars(
            select(RAGConnectorRun.error_code).where(RAGConnectorRun.id != run.id)
        )
        assert list(stale) == ["interrupted"]


async def test_capacity_stops_the_run_and_deletes_nothing(runtime, plain, sync):
    runtime.settings.rag_connector_unsettled_limit = 1
    run = await sync(plain, records(("a", "Alpha."), ("b", "Beta.")))
    assert (run.status, run.error_code, run.created) == ("partial", "capacity", 1)
    assert set(await documents(runtime)) == {"a"}


async def test_same_content_updates_title_in_place(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    document = (await documents(runtime))["a"]
    await make_ready(runtime, document.id)
    renamed = Fetch(records=[Record("a", "New title", "Alpha.")], complete=True)
    run = await sync(plain, renamed)
    assert (run.updated, run.unchanged) == (1, 0)
    after = (await documents(runtime))["a"]
    assert (after.title, after.version, after.status) == ("New title", 1, "ready")
    assert await events(runtime, document.id) == [("created", 1)]
    run = await sync(plain, renamed)
    assert (run.updated, run.unchanged) == (0, 1)


async def test_busy_version_is_left_alone(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    run = await sync(plain, records(("a", "Alpha changed.")))
    assert (run.unchanged, run.updated) == (1, 0)
    assert (await documents(runtime))["a"].version == 1


@pytest.mark.parametrize(
    ("code", "retried"),
    [
        (IngestError.ATTEMPTS_EXHAUSTED, True),
        (IngestError.QUEUE_EXPIRED, True),
        (IngestError.NO_TEXT, False),
    ],
)
async def test_same_content_retries_only_time_failures(runtime, plain, sync, code, retried):
    await sync(plain, records(("a", "Alpha.")))
    document = (await documents(runtime))["a"]
    await make_ready(runtime, document.id)
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document.id)
            .values(update_error_code=str(code))
        )
        await session.commit()
    run = await sync(plain, records(("a", "Alpha.")))
    assert (run.updated, run.unchanged) == ((1, 0) if retried else (0, 1))
    assert (await documents(runtime))["a"].version == (2 if retried else 1)


async def test_failed_first_version_retries_after_queue_expiry(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    document = (await documents(runtime))["a"]
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document.id)
            .values(
                status="failed",
                error_code=str(IngestError.QUEUE_EXPIRED),
                finished_at=func.clock_timestamp(),
                source=None,
            )
        )
        await session.commit()
    run = await sync(plain, records(("a", "Alpha.")))
    assert run.updated == 1
    after = (await documents(runtime))["a"]
    assert (after.version, after.status) == (2, "pending")


# --- Connector documents and the document API ------------------------------------------


async def test_api_refuses_to_change_connector_documents(runtime, plain, sync):
    await sync(plain, records(("a", "Alpha.")))
    document = (await documents(runtime))["a"]
    await make_ready(runtime, document.id)
    async with runtime.factory() as session:
        with pytest.raises(HTTPException) as caught:
            await update_document(
                session=session,
                settings=runtime.settings,
                owner_id=runtime.owner_id,
                document_id=document.id,
                upload=Upload(title=None, content_type=ContentType.TEXT, data=b"Other."),
                idempotency_key=uuid4().hex,
                trace_context={},
            )
        assert caught.value.status_code == 409
    async with runtime.factory() as session:
        with pytest.raises(HTTPException) as caught:
            await delete_document(
                session=session, owner_id=runtime.owner_id, document_id=document.id
            )
        assert caught.value.status_code == 409
    assert (await documents(runtime))["a"].status == "ready"


async def test_connector_documents_do_not_use_upload_quota(runtime, plain, sync):
    runtime.settings.rag_owner_document_limit = 1
    runtime.settings.rag_owner_unsettled_limit = 1
    await sync(plain, records(("a", "Alpha."), ("b", "Beta.")))
    # Uploads still fit, and the owner's list shows uploads only.
    event = await runtime.submit(b"An uploaded note.")
    async with runtime.factory() as session:
        listed = await list_documents(
            session=session, settings=runtime.settings, owner_id=runtime.owner_id
        )
    assert [document.id for document in listed] == [event.document_id]


async def test_sync_from_the_fake_http_source(runtime, connector, sync, source_api):
    source = {
        "kind": "http_json",
        "url": "https://source.example/records",
        "pagination": {"mode": "page"},
    }
    live = connector(source)
    transport = httpx.ASGITransport(app=source_api)

    async def through_fake(item, environ):
        return await fetch_http(item.source, item.max_records, environ, transport=transport)

    run = await sync(live, fetcher=through_fake)
    assert (run.status, run.created) == ("complete", 3)
    source_api.state.records.pop("notice-2")
    run = await sync(live, fetcher=through_fake)
    assert (run.status, run.deleted, run.unchanged) == ("complete", 1, 2)
