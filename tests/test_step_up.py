"""Unlock step-up: enroll an authenticator with the password, enable it with a code, and
confirm each unlock with a fresh code. Real PostgreSQL; tests choose the database step."""

import asyncio
import base64
import logging
import time
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, update

import app.assistant.step_up as step_up
from app.assistant.totp import STEP_SECONDS, code_at
from app.config import Settings
from app.models import AssistantToolCall, AuthFailure, Device, PendingAction, TotpCredential
from tests.support.assistant import (
    ACCOUNT_PASSWORD,
    SIMULATOR,
    TOTP_KEY,
    TOTP_PATH,
    StepClock,
    demo_residence,
    enable_authenticator,
    pending_unlock,
    serve,
)

WRONG_PASSWORD = "not-the-account-password"
BALCONY_DOOR = "Balcony door"
# Defaults of ASSISTANT_TOTP_FAILURE_LIMIT and AUTH_PASSWORD_FAILURE_LIMIT.
CODE_LIMIT = 5
PASSWORD_LIMIT = 10


@pytest.fixture
async def simulator():
    application = SIMULATOR.create_app()
    async with serve(application) as base_url:
        yield base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    monkeypatch.setenv("OPENAI_BASE_URL", simulator + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-never-external")
    monkeypatch.setenv("ASSISTANT_TOTP_KEY", TOTP_KEY)
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


@pytest.fixture
def clock(monkeypatch) -> StepClock:
    clock = StepClock()
    monkeypatch.setattr(step_up, "_database_step", clock)
    return clock


@pytest.fixture
def resident(client: httpx.AsyncClient, account):
    """A demo resident without an authenticator."""

    async def create() -> dict:
        owner = await account()
        await demo_residence(client, owner)
        return owner

    return create


def wrong_code(secret: bytes, step: int) -> str:
    """A six-digit code that matches no step of the window around `step`."""
    window = {code_at(secret, step + offset) for offset in (-1, 0, 1)}
    return next(code for n in range(4) if (code := f"{n:06d}") not in window)


async def enroll(client, owner: dict, password: str = ACCOUNT_PASSWORD) -> httpx.Response:
    return await client.post(
        f"{TOTP_PATH}/enroll", headers=owner["headers"], json={"password": password}
    )


async def activate(client, owner: dict, code: str) -> httpx.Response:
    return await client.post(f"{TOTP_PATH}/activate", headers=owner["headers"], json={"code": code})


async def confirm(client, owner: dict, action_id, code: str) -> httpx.Response:
    return await client.post(
        f"/api/v1/assistant/actions/{action_id}/confirm",
        headers=owner["headers"],
        json={"code": code},
    )


async def failures(app: FastAPI, owner: dict, kind: str) -> int:
    async with app.state.session_factory() as session:
        return await session.scalar(
            select(func.count())
            .select_from(AuthFailure)
            .where(AuthFailure.user_id == UUID(owner["id"]), AuthFailure.kind == kind)
        )


async def confirmation_audit(app: FastAPI, owner: dict) -> list[tuple]:
    async with app.state.session_factory() as session:
        rows = await session.execute(
            select(
                AssistantToolCall.decision,
                AssistantToolCall.result_code,
                AssistantToolCall.pending_action_id,
                AssistantToolCall.device_id,
            )
            .where(
                AssistantToolCall.user_id == UUID(owner["id"]),
                AssistantToolCall.source == "confirmation",
            )
            .order_by(AssistantToolCall.created_at)
        )
        return [tuple(row) for row in rows]


async def still_pending(app: FastAPI, action_id: UUID) -> bool:
    """The action waits and its door is still locked."""
    async with app.state.session_factory() as session:
        row = (
            await session.execute(
                select(PendingAction.status, Device.locked)
                .join(Device, Device.id == PendingAction.device_id)
                .where(PendingAction.id == action_id)
            )
        ).one()
    return tuple(row) == ("pending", True)


async def test_enrollment_needs_the_password_and_shows_the_secret_once(
    app: FastAPI, client, account
):
    owner = await account()
    refused = await enroll(client, owner, WRONG_PASSWORD)
    assert (refused.status_code, refused.json()["detail"]) == (403, "Invalid password")
    assert await failures(app, owner, "password") == 1

    response = await enroll(client, owner)
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    secret = base64.b32decode(body["secret"])
    assert len(secret) == 20
    assert body["otpauth_uri"].startswith("otpauth://totp/Resident%20assistant%3Atest-")
    assert f"secret={body['secret']}&" in body["otpauth_uri"]
    async with app.state.session_factory() as session:
        row = await session.scalar(
            select(TotpCredential).where(TotpCredential.user_id == UUID(owner["id"]))
        )
    # Stored sealed only, and not enabled until a first code matches.
    assert secret not in row.secret_ciphertext
    assert len(row.nonce) == 12
    assert (row.enabled_at, row.last_used_step) == (None, None)


async def test_reenrolling_replaces_an_unused_secret_but_never_an_enabled_one(
    client, account, clock
):
    owner = await account()
    first = base64.b32decode((await enroll(client, owner)).json()["secret"])
    second = base64.b32decode((await enroll(client, owner)).json()["secret"])

    assert first != second
    assert (await activate(client, owner, code_at(first, clock.step))).status_code == 403
    assert (await activate(client, owner, code_at(second, clock.step))).status_code == 200
    again = await enroll(client, owner)
    assert (again.status_code, again.json()["detail"]) == (409, "Authenticator already enabled")


async def test_activation_needs_an_enrollment_and_happens_once(client, account, clock):
    owner = await account()
    early = await activate(client, owner, "123456")
    assert (early.status_code, early.json()["detail"]) == (409, "Authenticator not enrolled")

    secret = await enable_authenticator(client, owner, clock)
    again = await activate(client, owner, code_at(secret, clock.step))
    assert (again.status_code, again.json()["detail"]) == (409, "Authenticator already enabled")


async def sealed_secret(app: FastAPI, owner: dict) -> bytes:
    async with app.state.session_factory() as session:
        return await session.scalar(
            select(TotpCredential.secret_ciphertext).where(
                TotpCredential.user_id == UUID(owner["id"])
            )
        )


async def test_enrollment_never_replaces_a_secret_enabled_after_its_read(
    app: FastAPI, client, account, clock, monkeypatch
):
    owner = await account()
    await enable_authenticator(client, owner, clock)
    before = await sealed_secret(app, owner)

    # As if another request inserted and enabled the row after this one's locked read.
    async def unseen(session, user_id):
        return None

    monkeypatch.setattr(step_up, "_locked_credential", unseen)
    response = await enroll(client, owner)

    assert (response.status_code, response.json()["detail"]) == (
        409,
        "Authenticator already enabled",
    )
    assert await sealed_secret(app, owner) == before


async def test_parallel_activations_enable_once(client, account, clock):
    owner = await account()
    secret = base64.b32decode((await enroll(client, owner)).json()["secret"])
    code = code_at(secret, clock.step)

    responses = await asyncio.gather(*[activate(client, owner, code) for _ in range(3)])

    assert sorted(response.status_code for response in responses) == [200, 409, 409]


@pytest.mark.parametrize("stage", ["never-enrolled", "enrolled-not-enabled"])
async def test_confirm_needs_an_enabled_authenticator(app: FastAPI, client, resident, clock, stage):
    owner = await resident()
    action_id = await pending_unlock(app, owner)
    if stage == "enrolled-not-enabled":
        assert (await enroll(client, owner)).status_code == 201

    # The caller's own action and an id that exists nowhere get the same refusal.
    for target in (action_id, uuid4()):
        response = await confirm(client, owner, target, "123456")
        assert (response.status_code, response.json()["detail"]) == (
            403,
            "Authenticator not enabled",
        )

    assert await still_pending(app, action_id)
    # No action or device id: neither is proven to be the caller's.
    expected = [("denied", "step_up_required", None, None)] * 2
    assert await confirmation_audit(app, owner) == expected


async def test_wrong_code_is_refused_counted_and_audited(app: FastAPI, client, resident, clock):
    owner = await resident()
    secret = await enable_authenticator(client, owner, clock)
    action_id = await pending_unlock(app, owner)

    response = await confirm(client, owner, action_id, wrong_code(secret, clock.step))

    assert (response.status_code, response.json()["detail"]) == (403, "Invalid authenticator code")
    assert await still_pending(app, action_id)
    assert await failures(app, owner, "totp") == 1
    assert await confirmation_audit(app, owner) == [("denied", "step_up_failed", None, None)]
    # The action still waits for the right code.
    assert (await confirm(client, owner, action_id, code_at(secret, clock.step))).status_code == 200


async def test_code_failures_stop_at_the_limit_before_the_code_is_checked(
    app: FastAPI, client, resident, clock
):
    owner = await resident()
    secret = await enable_authenticator(client, owner, clock)
    action_id = await pending_unlock(app, owner)
    for _ in range(CODE_LIMIT):
        refused = await confirm(client, owner, action_id, wrong_code(secret, clock.step))
        assert refused.status_code == 403

    # Even the right code is not checked now.
    response = await confirm(client, owner, action_id, code_at(secret, clock.step))

    assert (response.status_code, response.json()["detail"]) == (429, "Too many failed attempts")
    assert 1 <= int(response.headers["retry-after"]) <= 900
    assert await still_pending(app, action_id)
    assert await failures(app, owner, "totp") == CODE_LIMIT


async def test_parallel_wrong_codes_stop_at_exactly_the_limit(
    app: FastAPI, client, resident, clock
):
    owner = await resident()
    secret = await enable_authenticator(client, owner, clock)
    action_id = await pending_unlock(app, owner)
    guess = wrong_code(secret, clock.step)

    responses = await asyncio.gather(
        *[confirm(client, owner, action_id, guess) for _ in range(CODE_LIMIT + 2)]
    )

    assert sorted(response.status_code for response in responses) == [403] * CODE_LIMIT + [429] * 2
    assert await failures(app, owner, "totp") == CODE_LIMIT


async def test_one_code_confirms_one_action(app: FastAPI, client, resident, clock):
    owner = await resident()
    secret = await enable_authenticator(client, owner, clock)
    front = await pending_unlock(app, owner)
    balcony = await pending_unlock(app, owner, BALCONY_DOOR)
    code = code_at(secret, clock.step)

    responses = await asyncio.gather(
        confirm(client, owner, front, code), confirm(client, owner, balcony, code)
    )

    assert sorted(response.status_code for response in responses) == [200, 403]
    waiting = front if responses[0].status_code == 403 else balcony
    assert (await confirm(client, owner, waiting, code)).status_code == 403
    clock.step += 1
    assert (await confirm(client, owner, waiting, code_at(secret, clock.step))).status_code == 200


@pytest.mark.parametrize(("refusal", "status_code"), [("missing", 404), ("expired", 409)])
async def test_a_matched_code_is_spent_even_when_the_confirm_fails(
    app: FastAPI, client, resident, clock, refusal, status_code
):
    owner = await resident()
    secret = await enable_authenticator(client, owner, clock)
    action_id = await pending_unlock(app, owner)
    code = code_at(secret, clock.step)
    target = uuid4()
    if refusal == "expired":
        target = action_id
        async with app.state.session_factory() as session:
            await session.execute(
                update(PendingAction)
                .where(PendingAction.id == action_id)
                .values(
                    created_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 2),
                    expires_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 1),
                )
            )
            await session.commit()

    assert (await confirm(client, owner, target, code)).status_code == status_code
    replay = await confirm(client, owner, action_id, code)

    # RFC 6238 section 5.2: a matched code never matches again.
    assert (replay.status_code, replay.json()["detail"]) == (403, "Invalid authenticator code")


@pytest.mark.parametrize("damage", ["other-users-row", "changed-byte"])
async def test_a_sealed_secret_opens_only_for_its_own_user(
    app: FastAPI, client, resident, clock, caplog, damage
):
    other = await resident()
    owner = await resident()
    await enable_authenticator(client, other, clock)
    secret = await enable_authenticator(client, owner, clock)
    action_id = await pending_unlock(app, owner)
    async with app.state.session_factory() as session:
        sealed = await session.scalar(
            select(TotpCredential).where(TotpCredential.user_id == UUID(other["id"]))
        )
        own = await session.scalar(
            select(TotpCredential.secret_ciphertext).where(
                TotpCredential.user_id == UUID(owner["id"])
            )
        )
        values = {"nonce": sealed.nonce, "secret_ciphertext": sealed.secret_ciphertext}
        if damage == "changed-byte":
            values = {"secret_ciphertext": bytes([own[0] ^ 1]) + own[1:]}
        await session.execute(
            update(TotpCredential)
            .where(TotpCredential.user_id == UUID(owner["id"]))
            .values(**values)
        )
        await session.commit()

    response = await confirm(client, owner, action_id, code_at(secret, clock.step))

    assert (response.status_code, response.json()["detail"]) == (500, "Authenticator unavailable")
    assert "Authenticator secret failed to open" in caplog.text
    # Not the caller's failure: nothing counted, nothing audited, nothing unlocked.
    assert await failures(app, owner, "totp") == 0
    assert await confirmation_audit(app, owner) == []
    assert await still_pending(app, action_id)


async def test_no_secret_code_or_password_reaches_logs_or_spans(
    app: FastAPI, client, resident, clock, caplog, monkeypatch
):
    caplog.set_level(logging.DEBUG, logger="app")
    spans = []
    span = app.state.telemetry.span

    def recording(name, **attributes):
        spans.append((name, attributes))
        return span(name, **attributes)

    monkeypatch.setattr(app.state.telemetry, "span", recording)
    owner = await resident()

    assert (await enroll(client, owner, WRONG_PASSWORD)).status_code == 403
    encoded = (await enroll(client, owner)).json()["secret"]
    secret = base64.b32decode(encoded)
    guess = wrong_code(secret, clock.step)
    assert (await activate(client, owner, guess)).status_code == 403
    first = code_at(secret, clock.step)
    assert (await activate(client, owner, first)).status_code == 200
    clock.step += 1
    action_id = await pending_unlock(app, owner)
    assert (
        await confirm(client, owner, action_id, wrong_code(secret, clock.step))
    ).status_code == 403
    second = code_at(secret, clock.step)
    assert (await confirm(client, owner, action_id, second)).status_code == 200

    assert [name for name, _ in spans].count("assistant.confirm") == 2
    recorded = caplog.text + repr(spans)
    for private in (ACCOUNT_PASSWORD, WRONG_PASSWORD, encoded, guess, first, second):
        assert private not in recorded


@pytest.mark.parametrize(
    "body",
    [
        {"code": "12345"},
        {"code": "1234567"},
        {"code": "12345a"},
        {"code": "123456\n"},
        {"code": " 123456"},
        {"code": "١٢٣٤٥٦"},
        {"code": "123456", "action": "unlock"},
    ],
    ids=["short", "long", "letter", "newline", "space", "arabic-digits", "extra-field"],
)
async def test_codes_are_six_ascii_digits_and_never_echoed(client, resident, body):
    owner = await resident()
    for path in (f"{TOTP_PATH}/activate", f"/api/v1/assistant/actions/{uuid4()}/confirm"):
        response = await client.post(path, headers=owner["headers"], json=body)
        assert response.status_code == 422
        assert body["code"].strip() not in response.text


async def test_step_up_routes_need_a_token_and_the_feature(app: FastAPI, client, account):
    routes = [
        (f"{TOTP_PATH}/enroll", {"password": ACCOUNT_PASSWORD}),
        (f"{TOTP_PATH}/activate", {"code": "123456"}),
    ]
    for path, body in routes:
        assert (await client.post(path, json=body)).status_code == 401
    owner = await account()
    app.state.assistant_settings = None
    for path, body in routes:
        response = await client.post(path, headers=owner["headers"], json=body)
        assert response.status_code == 503


async def test_enrollment_failures_count_against_login(client, account):
    owner = await account()
    for _ in range(PASSWORD_LIMIT):
        assert (await enroll(client, owner, WRONG_PASSWORD)).status_code == 403

    throttled = await enroll(client, owner)
    login = await client.post(
        "/api/v1/auth/login", data={"username": owner["email"], "password": ACCOUNT_PASSWORD}
    )
    assert (throttled.status_code, login.status_code) == (429, 429)


async def test_database_step_follows_the_database_clock(app: FastAPI):
    async with app.state.session_factory() as session:
        step = await step_up._database_step(session)
    assert abs(step - int(time.time()) // STEP_SECONDS) <= 1
