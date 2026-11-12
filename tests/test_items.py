"""HTTP CRUD, ownership, persistence, keyset pagination, and body-limit regressions."""

import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, update

from app.models import Item

AccountFactory = Callable[..., Awaitable[dict]]
ItemFactory = Callable[..., Awaitable[dict]]
ITEM_FIELDS = {"id", "title", "description", "created_at", "updated_at"}


async def test_create_list_read_patch_and_delete_persist_in_postgresql(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory
) -> None:
    owner = await account()
    headers = owner["headers"]
    created = await client.post(
        "/api/v1/items", headers=headers, json={"title": "  Original title  "}
    )
    assert created.status_code == 201
    item = created.json()
    assert set(item) == ITEM_FIELDS
    assert item["title"] == "Original title"
    assert item["description"] == ""
    item_id = UUID(item["id"])
    async with app.state.session_factory() as session:
        stored = await session.get(Item, item_id)
        assert stored is not None
        assert str(stored.owner_id) == owner["id"]
        assert stored.title == "Original title"
        assert stored.description == ""
        assert stored.created_at.utcoffset() == timedelta(0)
        assert stored.updated_at.utcoffset() == timedelta(0)

    read = await client.get(f"/api/v1/items/{item_id}", headers=headers)
    assert read.status_code == 200
    assert read.json() == item
    page = await client.get("/api/v1/items", headers=headers)
    assert page.status_code == 200
    assert page.json() == {"items": [item], "next_cursor": None}

    changed = await client.patch(
        f"/api/v1/items/{item_id}", headers=headers, json={"description": "persisted change"}
    )
    assert changed.status_code == 200
    patched = changed.json()
    assert set(patched) == ITEM_FIELDS
    assert patched["title"] == "Original title"
    assert patched["description"] == "persisted change"
    assert patched["created_at"] == item["created_at"]
    assert datetime.fromisoformat(patched["updated_at"]) > datetime.fromisoformat(
        item["updated_at"]
    )
    async with app.state.session_factory() as session:
        stored = await session.get(Item, item_id)
        assert stored is not None
        assert stored.title == "Original title"
        assert stored.description == "persisted change"
        assert stored.updated_at == datetime.fromisoformat(patched["updated_at"])
    refreshed = await client.get(f"/api/v1/items/{item_id}", headers=headers)
    assert refreshed.status_code == 200
    assert refreshed.json() == patched

    renamed = await client.patch(
        f"/api/v1/items/{item_id}", headers=headers, json={"title": "  Renamed  "}
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert renamed.json()["description"] == "persisted change"
    async with app.state.session_factory() as session:
        stored = await session.get(Item, item_id)
        assert stored is not None
        assert stored.title == "Renamed"
        assert stored.description == "persisted change"

    deleted = await client.delete(f"/api/v1/items/{item_id}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""
    async with app.state.session_factory() as session:
        assert await session.get(Item, item_id) is None
    for response in (
        await client.delete(f"/api/v1/items/{item_id}", headers=headers),
        await client.get(f"/api/v1/items/{item_id}", headers=headers),
        await client.patch(f"/api/v1/items/{item_id}", headers=headers, json={"title": "Gone"}),
    ):
        assert response.status_code == 404
        assert response.json() == {"detail": "Item not found"}
    empty = await client.get("/api/v1/items", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "next_cursor": None}


async def test_foreign_owner_cannot_read_or_mutate_even_with_warmed_owner_cache(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory, item: ItemFactory
) -> None:
    owner = await account()
    outsider = await account()
    private = await item(owner, title="Private title", description="Private description")
    path = f"/api/v1/items/{private['id']}"
    warmed = await client.get(path, headers=owner["headers"])
    assert warmed.status_code == 200
    assert warmed.json() == private
    key = f"item:v1:{owner['id']}:{private['id']}"
    assert await app.state.redis.exists(key) == 1

    missing_path = f"/api/v1/items/{uuid4()}"
    for method, payload in (("GET", None), ("PATCH", {"title": "Stolen"}), ("DELETE", None)):
        kwargs = {"json": payload} if payload is not None else {}
        foreign = await client.request(method, path, headers=outsider["headers"], **kwargs)
        missing = await client.request(method, missing_path, headers=outsider["headers"], **kwargs)
        assert foreign.status_code == missing.status_code == 404
        assert foreign.json() == missing.json() == {"detail": "Item not found"}
        assert "Private" not in foreign.text
    listed = await client.get("/api/v1/items", headers=outsider["headers"])
    assert listed.status_code == 200
    assert listed.json() == {"items": [], "next_cursor": None}
    own_read = await client.get(path, headers=owner["headers"])
    assert own_read.status_code == 200
    assert own_read.json() == private
    async with app.state.session_factory() as session:
        stored = await session.get(Item, UUID(private["id"]))
        assert stored is not None
        assert str(stored.owner_id) == owner["id"]
        assert stored.title == "Private title"
        assert stored.description == "Private description"


async def test_json_cannot_forge_or_transfer_ownership(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory, item: ItemFactory
) -> None:
    owner = await account()
    outsider = await account()
    private = await item(owner)
    forged_create = await client.post(
        "/api/v1/items",
        headers=outsider["headers"],
        json={"owner_id": owner["id"], "title": "Forged"},
    )
    forged_patch = await client.patch(
        f"/api/v1/items/{private['id']}",
        headers=owner["headers"],
        json={"owner_id": outsider["id"], "description": "Transferred"},
    )
    assert forged_create.status_code == 422
    assert forged_patch.status_code == 422
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Item)) == 1
        stored = await session.get(Item, UUID(private["id"]))
        assert stored is not None
        assert str(stored.owner_id) == owner["id"]
        assert stored.description == "before"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"title": None},
        {"description": None},
        {"title": ""},
        {"title": " \t\n "},
        {"title": "x" * 201},
        {"description": "x" * 5001},
        {"title": 123},
        {"description": []},
        {"title": "Valid", "unexpected": "reject the entire patch"},
    ],
)
async def test_invalid_patch_is_atomic_and_does_not_modify_persisted_item(
    client: httpx.AsyncClient,
    app: FastAPI,
    account: AccountFactory,
    item: ItemFactory,
    payload: dict,
) -> None:
    owner = await account()
    original = await item(owner)
    response = await client.patch(
        f"/api/v1/items/{original['id']}", headers=owner["headers"], json=payload
    )
    assert response.status_code == 422
    async with app.state.session_factory() as session:
        stored = await session.get(Item, UUID(original["id"]))
        assert stored is not None
        assert stored.title == original["title"]
        assert stored.description == original["description"]
        assert stored.updated_at == datetime.fromisoformat(original["updated_at"])


def assert_nul_validation_is_private(
    response: httpx.Response, payload: dict[str, str], field: str
) -> None:
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors
    assert any(error["loc"] == ["body", field] for error in errors)
    for error in errors:
        assert "input" not in error
        assert "ctx" not in error
    assert "\x00" not in response.text
    assert "\\u0000" not in response.text
    for value in payload.values():
        assert value not in response.text
        assert json.dumps(value, ensure_ascii=True)[1:-1] not in response.text
        assert value.replace("\x00", "") not in response.text


@pytest.mark.parametrize("field", ["title", "description"])
@pytest.mark.parametrize("position", ["beginning", "middle", "end"])
async def test_create_rejects_nul_item_text_without_inserting_or_echoing_input(
    client: httpx.AsyncClient,
    app: FastAPI,
    account: AccountFactory,
    field: str,
    position: str,
) -> None:
    owner = await account()
    parts = {
        "beginning": "\x00private-create-text",
        "middle": "private-create\x00-text",
        "end": "private-create-text\x00",
    }
    payload = {"title": "Private valid create title", "description": "Private valid description"}
    payload[field] = parts[position]
    response = await client.post("/api/v1/items", headers=owner["headers"], json=payload)
    assert_nul_validation_is_private(response, payload, field)
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Item)) == 0


@pytest.mark.parametrize("field", ["title", "description"])
@pytest.mark.parametrize("position", ["beginning", "middle", "end"])
@pytest.mark.parametrize("change_other_field", [False, True], ids=["alone", "with-valid-change"])
async def test_patch_rejects_nul_item_text_atomically_without_echoing_input(
    client: httpx.AsyncClient,
    app: FastAPI,
    account: AccountFactory,
    item: ItemFactory,
    field: str,
    position: str,
    change_other_field: bool,
) -> None:
    owner = await account()
    original = await item(owner)
    parts = {
        "beginning": "\x00private-patch-text",
        "middle": "private-patch\x00-text",
        "end": "private-patch-text\x00",
    }
    payload = {field: parts[position]}
    if change_other_field:
        other_field = "description" if field == "title" else "title"
        payload[other_field] = "Private valid change must not persist"
    response = await client.patch(
        f"/api/v1/items/{original['id']}", headers=owner["headers"], json=payload
    )
    assert_nul_validation_is_private(response, payload, field)
    async with app.state.session_factory() as session:
        stored = await session.get(Item, UUID(original["id"]))
        assert stored is not None
        assert stored.title == original["title"]
        assert stored.description == original["description"]
        assert stored.updated_at == datetime.fromisoformat(original["updated_at"])


async def test_unicode_item_text_roundtrips_on_create_and_patch(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory
) -> None:
    owner = await account()
    initial = {"title": "Café 東京 🌍", "description": "مرحبا\n雪 ❄️ e\u0301"}
    created = await client.post("/api/v1/items", headers=owner["headers"], json=initial)
    assert created.status_code == 201
    original = created.json()
    assert original["title"] == initial["title"]
    assert original["description"] == initial["description"]
    async with app.state.session_factory() as session:
        stored = await session.get(Item, UUID(original["id"]))
        assert stored is not None
        assert stored.title == initial["title"]
        assert stored.description == initial["description"]

    changed = {"title": "新しい名前 🚀 naïve", "description": "Καλημέρα\nहिन्दी 🧪"}
    patched = await client.patch(
        f"/api/v1/items/{original['id']}", headers=owner["headers"], json=changed
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == changed["title"]
    assert patched.json()["description"] == changed["description"]
    async with app.state.session_factory() as session:
        stored = await session.get(Item, UUID(original["id"]))
        assert stored is not None
        assert stored.title == changed["title"]
        assert stored.description == changed["description"]
        assert stored.updated_at == datetime.fromisoformat(patched.json()["updated_at"])
    fetched = await client.get(f"/api/v1/items/{original['id']}", headers=owner["headers"])
    assert fetched.status_code == 200
    assert fetched.json() == patched.json()


async def test_create_validation_and_maximum_field_lengths(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory
) -> None:
    owner = await account()
    for payload in (
        {},
        {"title": ""},
        {"title": "  "},
        {"title": None},
        {"title": "x" * 201},
        {"title": "Title", "description": None},
        {"title": "Title", "description": "x" * 5001},
        {"title": "Title", "unknown": "not allowed"},
    ):
        response = await client.post("/api/v1/items", headers=owner["headers"], json=payload)
        assert response.status_code == 422
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Item)) == 0
    maximum = await client.post(
        "/api/v1/items",
        headers=owner["headers"],
        json={"title": "x" * 200, "description": "y" * 5000},
    )
    assert maximum.status_code == 201
    assert maximum.json()["title"] == "x" * 200
    assert maximum.json()["description"] == "y" * 5000
    cleared = await client.patch(
        f"/api/v1/items/{maximum.json()['id']}",
        headers=owner["headers"],
        json={"description": ""},
    )
    assert cleared.status_code == 200
    assert cleared.json()["description"] == ""
    assert cleared.json()["title"] == "x" * 200


async def test_tied_timestamp_keyset_pages_are_stable_complete_and_owner_scoped(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory, item: ItemFactory
) -> None:
    owner = await account()
    outsider = await account()
    records = [await item(owner, title=f"Item {index}") for index in range(7)]
    foreign = await item(outsider)
    tied_at = datetime(2026, 1, 1, 12, tzinfo=UTC)
    async with app.state.session_factory() as session:
        await session.execute(
            update(Item).where(Item.owner_id == UUID(owner["id"])).values(created_at=tied_at)
        )
        await session.execute(
            update(Item)
            .where(Item.id == UUID(foreign["id"]))
            .values(created_at=tied_at - timedelta(days=1))
        )
        await session.commit()
    expected = sorted((record["id"] for record in records), key=lambda value: UUID(value).int)
    expected.reverse()
    seen: list[str] = []
    cursor = None
    cursors: set[str] = set()
    page_sizes: list[int] = []
    while True:
        params: dict = {"limit": 3}
        if cursor is not None:
            params["cursor"] = cursor
        response = await client.get("/api/v1/items", headers=owner["headers"], params=params)
        assert response.status_code == 200
        page = response.json()
        page_sizes.append(len(page["items"]))
        seen.extend(record["id"] for record in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert cursor not in cursors
        cursors.add(cursor)
        decoded = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        assert datetime.fromisoformat(decoded[0]) == tied_at
        assert decoded[1] == page["items"][-1]["id"]
        if len(cursors) == 1:
            # Moving updated_at across the page boundary must not move its keyset position.
            patched = await client.patch(
                f"/api/v1/items/{expected[-1]}",
                headers=owner["headers"],
                json={"description": "Changed between pages"},
            )
            assert patched.status_code == 200
            foreign_page = await client.get(
                "/api/v1/items",
                headers=outsider["headers"],
                params={"limit": 3, "cursor": cursor},
            )
            assert foreign_page.status_code == 200
            assert [row["id"] for row in foreign_page.json()["items"]] == [foreign["id"]]
            assert foreign_page.json()["next_cursor"] is None
    assert page_sizes == [3, 3, 1]
    assert seen == expected
    assert len(set(seen)) == 7
    empty = await client.get(
        "/api/v1/items",
        headers=owner["headers"],
        params={"cursor": encode_cursor([tied_at.isoformat(), expected[-1]])},
    )
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "next_cursor": None}


def encode_cursor(value: object) -> str:
    return (
        base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )


async def test_malformed_cursors_and_page_limits_are_rejected(
    client: httpx.AsyncClient, account: AccountFactory
) -> None:
    owner = await account()
    identifier = "12345678-abcd-4321-abcd-123456789abc"
    timestamp = "2026-01-01T12:00:00+00:00"
    malformed = [
        "",
        "!not-base64!",
        "a",
        "x" * 513,
        "e30=",
        encode_cursor({"created_at": timestamp, "id": identifier}),
        encode_cursor([]),
        encode_cursor([timestamp]),
        encode_cursor([timestamp, identifier, "extra"]),
        encode_cursor(["2026-01-01T12:00:00", identifier]),
        encode_cursor(["not-a-time", identifier]),
        encode_cursor([timestamp, "not-a-uuid"]),
        encode_cursor([timestamp, identifier.upper()]),
        encode_cursor([timestamp, identifier.replace("-", "")]),
        encode_cursor([123, identifier]),
        encode_cursor([timestamp, None]),
        encode_cursor([timestamp, {}]),
    ]
    for cursor in malformed:
        response = await client.get(
            "/api/v1/items", headers=owner["headers"], params={"cursor": cursor}
        )
        assert response.status_code == 422, cursor
    for limit in ("0", "101", "-1", "1.5", "not-a-number"):
        response = await client.get(
            "/api/v1/items", headers=owner["headers"], params={"limit": limit}
        )
        assert response.status_code == 422, limit
    for limit in (1, 100):
        response = await client.get(
            "/api/v1/items", headers=owner["headers"], params={"limit": limit}
        )
        assert response.status_code == 200
        assert response.json() == {"items": [], "next_cursor": None}
    invalid_id = await client.get("/api/v1/items/not-a-uuid", headers=owner["headers"])
    assert invalid_id.status_code == 422


@pytest.mark.parametrize("mode", ["content_length", "chunked", "false_small_length"])
async def test_oversized_request_is_413_before_json_or_form_validation(
    client: httpx.AsyncClient, app: FastAPI, account: AccountFactory, mode: str
) -> None:
    owner = await account()
    body = json.dumps({"title": "Oversized", "description": "x" * (33 * 1024)}).encode()
    headers = {**owner["headers"], "Content-Type": "application/json"}
    if mode == "content_length":
        request = client.build_request("POST", "/api/v1/items", headers=headers, content=body)
        assert int(request.headers["content-length"]) > 32 * 1024
    else:

        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(body), 1024):
                yield body[offset : offset + 1024]

        if mode == "false_small_length":
            headers["Content-Length"] = "1"
        request = client.build_request("POST", "/api/v1/items", headers=headers, content=chunks())
        if mode == "chunked":
            assert "content-length" not in request.headers
            assert request.headers["transfer-encoding"] == "chunked"
    response = await client.send(request)
    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Item)) == 0


async def test_exact_body_limit_and_small_chunked_json_are_replayed_correctly(
    client: httpx.AsyncClient, account: AccountFactory
) -> None:
    owner = await account()
    payload = b'{"title":"At the body limit"}'
    body = payload + b" " * (32 * 1024 - len(payload))
    exact = await client.post(
        "/api/v1/items",
        headers={**owner["headers"], "Content-Type": "application/json"},
        content=body,
    )
    assert exact.status_code == 201
    assert exact.json()["title"] == "At the body limit"

    async def chunks() -> AsyncIterator[bytes]:
        yield b'{"title":'
        yield b'"Chunked title",'
        yield b'"description":"Replayed body"}'

    chunked = await client.post(
        "/api/v1/items",
        headers={**owner["headers"], "Content-Type": "application/json"},
        content=chunks(),
    )
    assert chunked.status_code == 201
    assert chunked.json()["title"] == "Chunked title"
    assert chunked.json()["description"] == "Replayed body"
