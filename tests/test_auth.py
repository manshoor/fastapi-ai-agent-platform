"""Account, credential secrecy, JWT, and bounded Argon2 integration regressions."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import httpx
import jwt
import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import func, insert, select

from app.config import Settings
from app.models import AuthFailure, User
from app.security import PasswordService

AccountFactory = Callable[..., Awaitable[dict]]


def assert_validation_is_private(response: httpx.Response, *secrets: str) -> None:
    assert response.status_code == 422
    for secret in secrets:
        assert secret not in response.text
    for error in response.json()["detail"]:
        assert "input" not in error
        assert "ctx" not in error


async def test_registration_login_and_persisted_argon2_hash(
    client: httpx.AsyncClient, app: FastAPI, settings: Settings
) -> None:
    password = "  unmodified-password-123  "
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "  Mixed.Case@Example.COM  ", "password": password},
    )
    assert response.status_code == 201
    user = response.json()
    assert set(user) == {"id", "email", "created_at"}
    assert user["email"] == "mixed.case@example.com"
    assert datetime.fromisoformat(user["created_at"]).utcoffset() == timedelta(0)
    async with app.state.session_factory() as session:
        stored = await session.scalar(select(User).where(User.id == UUID(user["id"])))
        assert stored is not None
        assert stored.email == user["email"]
        password_hash = stored.password_hash
    assert password_hash.startswith("$argon2id$")
    assert password not in password_hash
    assert await app.state.passwords.verify(password, password_hash)
    assert not await app.state.passwords.verify(password.strip(), password_hash)
    assert password_hash not in response.text
    assert password not in response.text

    login = await client.post(
        "/api/v1/auth/login",
        data={"username": "  MIXED.CASE@EXAMPLE.COM  ", "password": password},
    )
    assert login.status_code == 200
    token = login.json()
    assert set(token) == {"access_token", "token_type", "expires_in"}
    assert token["token_type"] == "bearer"
    assert token["expires_in"] == settings.auth_token_ttl_seconds
    assert password not in login.text
    assert password_hash not in login.text
    claims = jwt.decode(
        token["access_token"],
        bytes.fromhex(settings.jwt_secret),
        algorithms=["HS256"],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    assert claims["sub"] == user["id"]
    assert claims["exp"] - claims["iat"] == settings.auth_token_ttl_seconds
    authorized = await client.get(
        "/api/v1/items", headers={"Authorization": f"Bearer {token['access_token']}"}
    )
    assert authorized.status_code == 200
    assert authorized.json() == {"items": [], "next_cursor": None}


async def test_normalized_duplicate_email(
    client: httpx.AsyncClient, account: AccountFactory, app: FastAPI
) -> None:
    owner = await account(email="duplicate@example.com")
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "  DUPLICATE@EXAMPLE.COM  ", "password": "another-password-123"},
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "Email already registered"}
    async with app.state.session_factory() as session:
        users = (await session.scalars(select(User))).all()
        assert [str(user.id) for user in users] == [owner["id"]]


async def test_concurrent_duplicate_registration_commits_exactly_one_user(
    client: httpx.AsyncClient, app: FastAPI
) -> None:
    responses = await asyncio.gather(
        client.post(
            "/api/v1/auth/register",
            json={"email": "race@example.com", "password": "concurrent-password-123"},
        ),
        client.post(
            "/api/v1/auth/register",
            json={"email": "  RACE@EXAMPLE.COM  ", "password": "concurrent-password-123"},
        ),
    )
    assert sorted(response.status_code for response in responses) == [201, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json() == {"detail": "Email already registered"}
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(User)) == 1
    login = await client.post(
        "/api/v1/auth/login",
        data={"username": "race@example.com", "password": "concurrent-password-123"},
    )
    assert login.status_code == 200


@pytest.mark.parametrize("length", [12, 128])
async def test_password_boundary_lengths_are_accepted(
    client: httpx.AsyncClient, length: int
) -> None:
    password = "p" * length
    response = await client.post(
        "/api/v1/auth/register", json={"email": "bounds@example.com", "password": password}
    )
    assert response.status_code == 201
    login = await client.post(
        "/api/v1/auth/login", data={"username": "bounds@example.com", "password": password}
    )
    assert login.status_code == 200


@pytest.mark.parametrize("length", [0, 11, 129])
async def test_password_outside_bounds_is_rejected_without_echo(
    client: httpx.AsyncClient, app: FastAPI, length: int
) -> None:
    password = "s" * length
    register = await client.post(
        "/api/v1/auth/register", json={"email": "invalid@example.com", "password": password}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": "invalid@example.com", "password": password}
    )
    secrets = (password,) if password else ()
    assert_validation_is_private(register, *secrets)
    assert_validation_is_private(login, *secrets)
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(User)) == 0


@pytest.mark.parametrize("location", ["extra_name", "extra_value", "email", "password_type"])
async def test_validation_errors_never_echo_secret_fields_or_values(
    client: httpx.AsyncClient, location: str
) -> None:
    secret = "private-marker-do-not-reflect"
    payload: dict = {"email": "private@example.com", "password": "valid-password-123"}
    if location == "extra_name":
        payload[secret] = "unexpected"
    elif location == "extra_value":
        payload["password_hash"] = secret
    elif location == "email":
        payload["email"] = f"{secret}@invalid-domain"
    else:
        payload["password"] = {"private": secret}
    response = await client.post("/api/v1/auth/register", json=payload)
    assert_validation_is_private(response, secret, "valid-password-123")


async def test_invalid_login_email_is_private(client: httpx.AsyncClient) -> None:
    secret = "secret-email-validation-marker"
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": f"{secret}@invalid-domain", "password": "private-password-123"},
    )
    assert_validation_is_private(response, secret, "private-password-123")


async def test_wrong_password_and_unknown_email_have_identical_unauthorized_response(
    client: httpx.AsyncClient, account: AccountFactory
) -> None:
    await account(email="known@example.com")
    wrong, unknown = await asyncio.gather(
        client.post(
            "/api/v1/auth/login",
            data={"username": "known@example.com", "password": "incorrect-password-123"},
        ),
        client.post(
            "/api/v1/auth/login",
            data={"username": "unknown@example.com", "password": "incorrect-password-123"},
        ),
    )
    for response in (wrong, unknown):
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid credentials"}
        assert response.headers["www-authenticate"] == "Bearer"
    assert wrong.content == unknown.content


@pytest.mark.parametrize("authorization", [None, "Basic credentials", "Bearer not.a.jwt", "Bearer"])
async def test_missing_or_malformed_bearer_token(
    client: httpx.AsyncClient, authorization: str | None
) -> None:
    headers = {} if authorization is None else {"Authorization": authorization}
    response = await client.get("/api/v1/items", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}
    assert response.headers["www-authenticate"] == "Bearer"


def signed_claims(settings: Settings) -> dict:
    now = datetime.now(UTC)
    return {
        "sub": str(uuid4()),
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "expired",
        "tampered",
        "audience",
        "issuer",
        "algorithm",
        "unsigned",
        "future_iat",
        "missing_sub",
        "missing_iat",
        "missing_exp",
        "missing_iss",
        "missing_aud",
    ],
)
async def test_invalid_jwt_is_rejected(
    client: httpx.AsyncClient, settings: Settings, defect: str
) -> None:
    claims = signed_claims(settings)
    algorithm = "HS256"
    key: bytes | str = bytes.fromhex(settings.jwt_secret)
    if defect == "expired":
        claims["iat"] = datetime.now(UTC) - timedelta(minutes=10)
        claims["exp"] = datetime.now(UTC) - timedelta(minutes=5)
    elif defect == "audience":
        claims["aud"] = "another-api"
    elif defect == "issuer":
        claims["iss"] = "another-issuer"
    elif defect == "algorithm":
        algorithm = "HS384"
        settings.jwt_secret = "ab" * 64
        key = bytes.fromhex(settings.jwt_secret)
    elif defect == "unsigned":
        algorithm = "none"
        key = ""
    elif defect == "future_iat":
        claims["iat"] = datetime.now(UTC) + timedelta(minutes=1)
    elif defect.startswith("missing_"):
        del claims[defect.removeprefix("missing_")]
    token = jwt.encode(claims, key, algorithm=algorithm)
    if defect == "tampered":
        header, payload, signature = token.split(".")
        changed = ("A" if signature[0] != "A" else "B") + signature[1:]
        token = ".".join((header, payload, changed))
    response = await client.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert token not in response.text


@pytest.mark.parametrize(
    "subject",
    [
        "not-a-uuid",
        "",
        "12345678-ABCD-4321-ABCD-123456789ABC",
        "12345678abcd4321abcd123456789abc",
        "{12345678-abcd-4321-abcd-123456789abc}",
        123,
        None,
        [],
        {},
    ],
)
async def test_jwt_subject_must_be_a_canonical_uuid_string(
    client: httpx.AsyncClient, settings: Settings, subject: object
) -> None:
    claims = signed_claims(settings)
    claims["sub"] = subject
    token = jwt.encode(claims, bytes.fromhex(settings.jwt_secret), algorithm="HS256")
    response = await client.get("/api/v1/items", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication credentials"}
    assert response.headers["www-authenticate"] == "Bearer"


async def test_cancelled_argon2_worker_holds_admission_until_thread_finishes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = PasswordService(
        settings.model_copy(
            update={"auth_hash_concurrency": 1, "auth_hash_queue_timeout_seconds": 0.02}
        )
    )
    await service.initialize()
    entered = threading.Event()
    release = threading.Event()
    original_hash = service._hasher.hash
    worker_threads: list[int] = []
    main_thread = threading.get_ident()

    def paused_hash(password: str) -> str:
        worker_threads.append(threading.get_ident())
        if password == "paused-password-123":
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Test did not release the admitted hashing worker")
        return original_hash(password)

    monkeypatch.setattr(service._hasher, "hash", paused_hash)
    caller = asyncio.create_task(service.hash("paused-password-123"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert len(worker_threads) == 1
        assert worker_threads[0] != main_thread
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done()
        with pytest.raises(HTTPException) as rejected:
            await service.hash("replacement-password-123")
        assert rejected.value.status_code == 503
        assert rejected.value.headers == {"Retry-After": "1"}
        assert len(worker_threads) == 1
        assert not caller.done()
    finally:
        # Never await the cancelled caller while its real worker is still paused.
        release.set()
        if not caller.done():
            caller.cancel()
        await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), timeout=10)
    assert caller.cancelled()
    restored_hash = await service.hash("replacement-password-123")
    assert len(worker_threads) == 2
    assert restored_hash.startswith("$argon2id$")
    assert await service.verify("replacement-password-123", restored_hash)


# AUTH_PASSWORD_FAILURE_LIMIT default; the conftest account's password.
PASSWORD_FAILURE_LIMIT = 10
ACCOUNT_PASSWORD = "a-secure-password-123"


async def login(client: httpx.AsyncClient, email: str, password: str) -> httpx.Response:
    return await client.post("/api/v1/auth/login", data={"username": email, "password": password})


async def test_password_failures_throttle_the_account_before_argon2(
    app: FastAPI, client: httpx.AsyncClient, account: AccountFactory, monkeypatch
) -> None:
    owner, other = await account(), await account()
    for _ in range(PASSWORD_FAILURE_LIMIT):
        assert (await login(client, owner["email"], "incorrect-password-123")).status_code == 401

    verified = []
    verify = app.state.passwords.verify

    async def counting(password: str, password_hash: str) -> bool:
        verified.append(password_hash)
        return await verify(password, password_hash)

    monkeypatch.setattr(app.state.passwords, "verify", counting)
    throttled = await login(client, owner["email"], ACCOUNT_PASSWORD)

    assert throttled.status_code == 429
    assert throttled.json() == {"detail": "Too many failed attempts"}
    assert 1 <= int(throttled.headers["retry-after"]) <= 900
    assert verified == []
    # One account's failures never throttle another.
    assert (await login(client, other["email"], ACCOUNT_PASSWORD)).status_code == 200


async def test_unknown_email_is_never_throttled_or_recorded(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    for _ in range(PASSWORD_FAILURE_LIMIT + 2):
        response = await login(client, "nobody@example.com", "incorrect-password-123")
        assert response.status_code == 401
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(AuthFailure)) == 0


async def test_failures_outside_the_window_are_not_counted_and_get_pruned(
    app: FastAPI, client: httpx.AsyncClient, account: AccountFactory
) -> None:
    owner = await account()
    user_id = UUID(owner["id"])
    stale = datetime.now(UTC) - timedelta(minutes=16)
    async with app.state.session_factory() as session:
        await session.execute(
            insert(AuthFailure),
            [
                {"user_id": user_id, "kind": "password", "created_at": stale}
                for _ in range(PASSWORD_FAILURE_LIMIT)
            ],
        )
        await session.commit()

    accepted = await login(client, owner["email"], ACCOUNT_PASSWORD)
    refused = await login(client, owner["email"], "incorrect-password-123")

    assert (accepted.status_code, refused.status_code) == (200, 401)
    async with app.state.session_factory() as session:
        kept = (
            await session.scalars(
                select(AuthFailure.created_at).where(AuthFailure.user_id == user_id)
            )
        ).all()
    assert len(kept) == 1 and kept[0] > stale


async def test_retry_after_counts_down_from_the_oldest_counted_failure(
    app: FastAPI, client: httpx.AsyncClient, account: AccountFactory
) -> None:
    owner = await account()
    # The oldest counted failure is 100 s old: it leaves the 900 s window in 800 s.
    ages = [100] + [5] * (PASSWORD_FAILURE_LIMIT - 1)
    async with app.state.session_factory() as session:
        for age in ages:
            await session.execute(
                insert(AuthFailure).values(
                    user_id=UUID(owner["id"]),
                    kind="password",
                    created_at=func.clock_timestamp() - timedelta(seconds=age),
                )
            )
        await session.commit()

    throttled = await login(client, owner["email"], ACCOUNT_PASSWORD)

    assert throttled.status_code == 429
    # Rounded up: only milliseconds passed since the rows were written.
    assert throttled.headers["retry-after"] == "800"
