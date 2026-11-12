"""Upload, list, read and delete RAG documents through the real app and database."""

import asyncio
from contextlib import AsyncExitStack
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, HTTPException
from sqlalchemy import func, select

from app.config import Settings
from app.models import RAGDocument, RAGOutbox
from app.rag.contracts import ContentType
from app.rag.documents import submit_document
from app.rag.uploads import Upload

URL = "/api/v1/rag/documents"
SECRET = "sk-must-never-echo-7f3a"
TEXT_LIMIT = 4096
PDF_LIMIT = 8192
OWNER_UNSETTLED = 3
OWNER_DOCUMENTS = 5


@pytest.fixture
def settings(monkeypatch) -> Settings:
    # RAG settings are read from the environment when the lifespan builds the runtime.
    for name, value in {
        "RAG_TEXT_MAX_BYTES": TEXT_LIMIT,
        "RAG_PDF_MAX_BYTES": PDF_LIMIT,
        "RAG_OWNER_UNSETTLED_LIMIT": OWNER_UNSETTLED,
        "RAG_OWNER_DOCUMENT_LIMIT": OWNER_DOCUMENTS,
        "RAG_UPLOAD_TIMEOUT_SECONDS": 1,
        # The runtime builds a provider client; these tests never call it.
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
    }.items():
        monkeypatch.setenv(name, str(value))
    return Settings(ai_enabled=True, rag_enabled=True)


def headers(owner: dict, key: str | None = None) -> dict:
    return {**owner["headers"], "Idempotency-Key": key or uuid4().hex}


async def upload(
    client: httpx.AsyncClient,
    owner: dict,
    *,
    key: str | None = None,
    title: str = "Pump notes",
    name: str = "notes.md",
    data: bytes = b"# Pump\n\nThe pump runs at 40 litres per minute.",
):
    return await client.post(
        URL, headers=headers(owner, key), data={"title": title}, files={"file": (name, data)}
    )


async def test_upload_stores_pending_document_and_outbox_event(
    app: FastAPI, client: httpx.AsyncClient, account
):
    owner = await account()
    response = await upload(client, owner, title="  Pump notes  ")
    assert response.status_code == 202
    body = response.json()
    assert response.headers["location"] == body["status_url"] == f"{URL}/{body['id']}"
    assert response.headers["cache-control"] == "no-store"
    assert body["title"] == "Pump notes" and body["content_type"] == "text/markdown"
    assert body["status"] == "pending" and body["chunk_count"] is None
    assert "source" not in body
    async with app.state.session_factory() as session:
        document = await session.get(RAGDocument, UUID(body["id"]))
        assert document.owner_id == UUID(owner["id"])
        assert (await session.scalar(select(RAGDocument.source))).startswith(b"# Pump")
        event = await session.scalar(select(RAGOutbox))
        assert (event.document_id, event.event_kind, event.attempt) == (
            document.id,
            "requested",
            1,
        )


async def test_replay_returns_original_and_other_content_conflicts(
    app: FastAPI, client: httpx.AsyncClient, account
):
    owner = await account()
    key = uuid4().hex
    first = await upload(client, owner, key=key)
    again = await upload(client, owner, key=key)
    assert again.status_code == 202 and again.json()["id"] == first.json()["id"]
    conflict = await upload(client, owner, key=key, data=b"Different content")
    assert conflict.status_code == 409
    # The same key belongs to each owner separately.
    other = await upload(client, await account(), key=key)
    assert other.status_code == 202 and other.json()["id"] != first.json()["id"]
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(RAGDocument)) == 2


async def test_replay_of_finished_document_returns_200(
    app: FastAPI, client: httpx.AsyncClient, account
):
    owner = await account()
    key = uuid4().hex
    created = (await upload(client, owner, key=key)).json()
    async with app.state.session_factory() as session:
        document = await session.get(RAGDocument, UUID(created["id"]))
        document.status, document.source, document.chunk_count = "ready", None, 1
        document.finished_at = func.now()
        await session.commit()
    replay = await upload(client, owner, key=key)
    assert replay.status_code == 200 and replay.json()["status"] == "ready"


async def test_list_get_delete_are_owner_scoped(app: FastAPI, client: httpx.AsyncClient, account):
    owner, stranger = await account(), await account()
    first = (await upload(client, owner, title="First")).json()
    second = (await upload(client, owner, title="Second")).json()
    await upload(client, stranger, title="Foreign")

    listed = await client.get(URL, headers=owner["headers"])
    assert listed.status_code == 200 and listed.headers["cache-control"] == "no-store"
    assert [item["id"] for item in listed.json()["items"]] == [second["id"], first["id"]]

    read = await client.get(f"{URL}/{first['id']}", headers=owner["headers"])
    assert read.status_code == 200 and read.json() == first

    # Foreign and missing documents are indistinguishable.
    foreign = await client.get(f"{URL}/{first['id']}", headers=stranger["headers"])
    missing = await client.get(f"{URL}/{uuid4()}", headers=owner["headers"])
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    denied = await client.delete(f"{URL}/{first['id']}", headers=stranger["headers"])
    assert denied.status_code == 404

    deleted = await client.delete(f"{URL}/{first['id']}", headers=owner["headers"])
    assert deleted.status_code == 204
    assert (await client.get(f"{URL}/{first['id']}", headers=owner["headers"])).status_code == 404
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(RAGOutbox)) == 2


async def test_owner_quotas_hold_under_concurrent_uploads(
    app: FastAPI, client: httpx.AsyncClient, account
):
    owner = await account()
    runtime = app.state.rag

    # The route allows one upload per owner per process, so race the quota lock directly,
    # as separate API processes would.
    async def submit(index: int) -> int:
        async with app.state.session_factory() as session:
            try:
                _, status_code = await submit_document(
                    session=session,
                    settings=runtime.settings,
                    owner_id=UUID(owner["id"]),
                    upload=Upload(
                        title="Doc", content_type=ContentType.TEXT, data=f"Doc {index}".encode()
                    ),
                    idempotency_key=uuid4().hex,
                    trace_context={},
                )
            except HTTPException as exc:
                return exc.status_code
        return status_code

    codes = sorted(await asyncio.gather(*(submit(index) for index in range(OWNER_UNSETTLED + 1))))
    assert codes == [202] * OWNER_UNSETTLED + [429]

    # Settled documents free the unsettled quota but still count toward the total.
    async with app.state.session_factory() as session:
        for document in await session.scalars(select(RAGDocument)):
            document.status, document.source, document.error_code = "failed", None, "no_text"
            document.finished_at = func.now()
        await session.commit()
    accepted = [
        (await upload(client, owner, data=f"Later {index}".encode())).status_code
        for index in range(3)
    ]
    assert accepted == [202, 202, 429]


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("notes.html", b"<p>hi</p>", 415),
        ("notes", b"no suffix", 415),
        ("notes.md", b"x" * (TEXT_LIMIT + 1), 413),
        ("report.pdf", b"%PDF-" + b"x" * PDF_LIMIT, 413),
        ("notes.md", b"", 422),
        ("notes.txt", b"\xff\xfe" + SECRET.encode(), 422),
        ("notes.txt", SECRET.encode() + b"\x00", 422),
        ("report.pdf", SECRET.encode(), 422),
    ],
    ids=["html", "no-suffix", "text-size", "pdf-size", "empty", "utf8", "nul", "pdf-header"],
)
async def test_invalid_files_are_rejected_without_echo(
    app: FastAPI, client: httpx.AsyncClient, account, name, data, status
):
    owner = await account()
    response = await upload(client, owner, name=name, data=data)
    assert response.status_code == status
    assert SECRET not in response.text
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(RAGDocument)) == 0


async def test_valid_pdf_and_bom_text_are_accepted(client: httpx.AsyncClient, account):
    owner = await account()
    pdf = await upload(client, owner, name="Report.PDF", data=b"%PDF-1.7\n%stub")
    text = await upload(client, owner, name="a.txt", data=b"\xef\xbb\xbfplain caf\xc3\xa9")
    assert pdf.status_code == text.status_code == 202
    assert pdf.json()["content_type"] == "application/pdf"
    assert text.json()["content_type"] == "text/plain"


@pytest.mark.parametrize(
    ("parts", "status"),
    [
        ([("title", (None, "")), ("file", ("a.md", b"text"))], 422),
        ([("title", (None, "x" * 201)), ("file", ("a.md", b"text"))], 422),
        ([("title", (None, SECRET + "\x00")), ("file", ("a.md", b"text"))], 422),
        ([("file", ("a.md", b"text"))], 400),
        ([("title", (None, "No file"))], 400),
        ([("title", (None, "Extra")), ("other", (None, SECRET)), ("file", ("a.md", b"t"))], 400),
        ([("title", (None, "Two")), ("file", ("a.md", b"one")), ("file", ("b.md", b"two"))], 400),
    ],
    ids=["empty-title", "long-title", "nul-title", "no-title", "no-file", "extra", "two-files"],
)
async def test_malformed_forms_are_rejected_without_echo(
    client: httpx.AsyncClient, account, parts, status
):
    owner = await account()
    response = await client.post(URL, headers=headers(owner), files=parts)
    assert response.status_code == status
    assert SECRET not in response.text


async def test_non_multipart_body_is_unsupported(client: httpx.AsyncClient, account):
    owner = await account()
    response = await client.post(URL, headers=headers(owner), json={"title": "x"})
    assert response.status_code == 415


async def test_missing_idempotency_key_is_rejected(client: httpx.AsyncClient, account):
    owner = await account()
    response = await client.post(
        URL, headers=owner["headers"], data={"title": "x"}, files={"file": ("a.md", b"t")}
    )
    assert response.status_code == 422


def multipart(title: str, name: str, payload: bytes) -> tuple[bytes, bytes]:
    boundary = uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="title"\r\n\r\n{title}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return f"multipart/form-data; boundary={boundary}".encode(), body


async def raw_post(
    app: FastAPI, request_headers: list, chunks: list[bytes | None], pause: float = 0
):
    """Drive the ASGI app directly: no Content-Length, chunked body, observable reads.

    A None chunk means the client disconnects at that point.
    """
    pending = list(chunks)
    reads = 0
    sent: list[dict] = []

    async def receive():
        nonlocal reads
        reads += 1
        if pause:
            await asyncio.sleep(pause)
        if not pending:
            return {"type": "http.disconnect"}
        chunk = pending.pop(0)
        if chunk is None:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": chunk, "more_body": bool(pending)}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": URL,
        "raw_path": URL.encode(),
        "query_string": b"",
        "headers": request_headers,
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    await app(scope, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    return status, reads


async def test_unauthenticated_upload_never_reads_the_body(app: FastAPI):
    content_type, body = multipart("Title", "a.md", b"x" * 1000)
    status, reads = await raw_post(
        app,
        [(b"content-type", content_type), (b"idempotency-key", uuid4().hex.encode())],
        [body],
    )
    assert (status, reads) == (401, 0)


async def test_streamed_body_past_ceiling_is_413(app: FastAPI, account):
    owner = await account()
    token = owner["headers"]["Authorization"].encode()
    ceiling = app.state.rag_upload_ceiling
    content_type, body = multipart("Big", "a.md", b"x" * (2 * ceiling))
    chunks = [body[index : index + 16384] for index in range(0, len(body), 16384)]
    status, reads = await raw_post(
        app,
        [
            (b"authorization", token),
            (b"content-type", content_type),
            (b"idempotency-key", uuid4().hex.encode()),
        ],
        chunks,
    )
    assert status == 413
    assert reads < len(chunks)


async def test_slow_body_times_out_with_408(app: FastAPI, account):
    owner = await account()
    content_type, body = multipart("Slow", "a.md", b"text")
    status, _ = await raw_post(
        app,
        [
            (b"authorization", owner["headers"]["Authorization"].encode()),
            (b"content-type", content_type),
            (b"idempotency-key", uuid4().hex.encode()),
        ],
        [body[:10], body[10:]],
        pause=0.7,
    )
    assert status == 408


def owner_headers(owner: dict, content_type: bytes) -> list:
    return [
        (b"authorization", owner["headers"]["Authorization"].encode()),
        (b"content-type", content_type),
        (b"idempotency-key", uuid4().hex.encode()),
    ]


async def test_client_disconnect_mid_upload_is_a_fixed_400(app: FastAPI, account):
    owner = await account()
    content_type, body = multipart("Gone", "a.md", b"x" * 1000)
    status, _ = await raw_post(app, owner_headers(owner, content_type), [body[:50], None])
    assert status == 400


async def test_unknown_multipart_charset_is_a_fixed_400(app: FastAPI, account):
    owner = await account()
    content_type, body = multipart("Title", "a.md", b"text")
    status, _ = await raw_post(
        app, owner_headers(owner, content_type + b"; charset=undefined"), [body]
    )
    assert status == 400


async def test_one_owner_holds_one_upload_slot(app: FastAPI, client: httpx.AsyncClient, account):
    owner, other = await account(), await account()
    runtime = app.state.rag
    async with runtime.upload_slot(UUID(owner["id"])):
        held = await upload(client, owner)
        free = await upload(client, other)
    assert held.status_code == 429 and held.headers["retry-after"] == "1"
    assert free.status_code == 202


async def test_busy_upload_slots_shed_load(app: FastAPI, client: httpx.AsyncClient, account):
    owner = await account()
    runtime = app.state.rag
    async with AsyncExitStack() as stack:
        for _ in range(runtime.settings.rag_upload_concurrency):
            await stack.enter_async_context(runtime.upload_slot(uuid4()))
        busy = await upload(client, owner)
    assert busy.status_code == 503 and busy.headers["retry-after"] == "1"
    assert (await upload(client, owner)).status_code == 202


async def test_disabled_rag_returns_503_and_crud_still_works(monkeypatch):
    monkeypatch.delenv("RAG_ENABLED", raising=False)
    from app.main import create_app

    application = create_app(Settings(rag_enabled=False))
    async with LifespanManager(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as http:
            email = f"test-{uuid4().hex}@example.com"
            password = "a-secure-password-123"
            await http.post("/api/v1/auth/register", json={"email": email, "password": password})
            login = await http.post(
                "/api/v1/auth/login", data={"username": email, "password": password}
            )
            auth = {"Authorization": f"Bearer {login.json()['access_token']}"}
            disabled = await http.get(URL, headers=auth)
            assert disabled.status_code == 503
            created = await http.post("/api/v1/items", headers=auth, json={"title": "Still"})
            assert created.status_code == 201
