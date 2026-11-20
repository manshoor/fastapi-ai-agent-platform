"""Authenticator-code step-up before a pending unlock runs.

    enroll    password -> failure limit reached: 429, Argon2 not run
                       -> wrong: a password failure (shared with login), 403
              lock row -> already enabled: 409
              -> new secret, sealed, replaces any unconfirmed one -> secret + otpauth URI
    activate  lock row -> none, or already enabled: 409
              -> failure limit reached: 429, code not checked
              -> code matches -> enabled, its step spent; else a code failure, 403
    verify    lock row -> not enabled: audit step_up_required, 403
              -> failure limit reached: 429, code not checked
              -> code matches a step after the last spent one -> step spent, committed
                 else a code failure and audit step_up_failed, 403
              the confirm then claims the action in its own transaction

A matching code is spent and committed before the claim (RFC 6238 section 5.2): a 404
or 409 after the match cannot hand the same code to a second try. The row lock
serializes every check for one user, so parallel guesses cannot pass the failure count
together and one code cannot confirm twice. Steps come from the database clock.
Secrets are sealed with AES-256-GCM under ASSISTANT_TOTP_KEY, with the user id as
associated data: a row copied to another user does not open.
Secrets, codes and passwords are never logged, traced or stored in plain text.
"""

import base64
import logging
import os
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException
from sqlalchemy import BigInteger, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as upsert
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant import totp
from app.assistant.config import AssistantSettings
from app.assistant.contracts import Decision, ResultCode, ToolName
from app.assistant.residence import Resident
from app.config import Settings
from app.models import AssistantToolCall, TotpCredential, User
from app.security import PasswordService
from app.throttle import FailureKind, add_failure, check_failures

logger = logging.getLogger(__name__)
NONCE_BYTES = 12
HTTP_FORBIDDEN = 403
HTTP_CONFLICT = 409
HTTP_INTERNAL_ERROR = 500
WRONG_PASSWORD = "Invalid password"
NOT_ENROLLED = "Authenticator not enrolled"
NOT_ENABLED = "Authenticator not enabled"
ALREADY_ENABLED = "Authenticator already enabled"
WRONG_CODE = "Invalid authenticator code"
UNAVAILABLE = "Authenticator unavailable"


class SecretBox:
    """AES-256-GCM sealing for TOTP secrets, bound to their user."""

    def __init__(self, key: bytes) -> None:
        self._aead = AESGCM(key)

    @classmethod
    def from_settings(cls, settings: AssistantSettings) -> "SecretBox":
        # The settings validator already checked: base64 for exactly 32 bytes.
        key = settings.assistant_totp_key.get_secret_value()
        return cls(base64.b64decode(key, validate=True))

    def seal(self, user_id: UUID, secret: bytes) -> tuple[bytes, bytes]:
        nonce = os.urandom(NONCE_BYTES)
        return nonce, self._aead.encrypt(nonce, secret, user_id.bytes)

    def open(self, user_id: UUID, nonce: bytes, ciphertext: bytes) -> bytes:
        # Raises InvalidTag for another user's row, a changed byte or another key.
        return self._aead.decrypt(nonce, ciphertext, user_id.bytes)


async def check_password(
    session: AsyncSession,
    passwords: PasswordService,
    settings: Settings,
    user_id: UUID,
    password: str,
) -> str:
    """The account's email once its password checks out. Login shares the failure count."""
    await check_failures(
        session,
        user_id,
        FailureKind.PASSWORD,
        settings.auth_password_failure_limit,
        settings.auth_password_failure_window_seconds,
    )
    account = (
        await session.execute(select(User.email, User.password_hash).where(User.id == user_id))
    ).one_or_none()
    # No transaction stays open while Argon2 runs.
    await session.rollback()
    if account is None:
        raise HTTPException(HTTP_FORBIDDEN, WRONG_PASSWORD)
    if await passwords.verify(password, account.password_hash):
        return account.email
    await add_failure(
        session, user_id, FailureKind.PASSWORD, settings.auth_password_failure_window_seconds
    )
    await session.commit()
    raise HTTPException(HTTP_FORBIDDEN, WRONG_PASSWORD)


async def _database_step(session: AsyncSession) -> int:
    epoch = func.extract("epoch", func.clock_timestamp())
    return await session.scalar(select(func.floor(epoch / totp.STEP_SECONDS).cast(BigInteger)))


async def _locked_credential(session: AsyncSession, user_id: UUID) -> TotpCredential | None:
    return await session.scalar(
        select(TotpCredential).where(TotpCredential.user_id == user_id).with_for_update()
    )


async def enroll(session: AsyncSession, box: SecretBox, user_id: UUID) -> bytes:
    """A new, not yet enabled secret for this user. The caller shows it once."""
    credential = await _locked_credential(session, user_id)
    if credential is not None and credential.enabled_at is not None:
        await session.rollback()
        raise HTTPException(HTTP_CONFLICT, ALREADY_ENABLED)
    secret = totp.new_secret()
    nonce, ciphertext = box.seal(user_id, secret)
    statement = upsert(TotpCredential).values(
        user_id=user_id, secret_ciphertext=ciphertext, nonce=nonce
    )
    # The WHERE covers a row another request inserted and enabled after the lock read
    # found none: an enabled secret is never replaced.
    replaced = await session.scalar(
        statement.on_conflict_do_update(
            index_elements=[TotpCredential.user_id],
            set_={"secret_ciphertext": ciphertext, "nonce": nonce, "created_at": func.now()},
            where=TotpCredential.enabled_at.is_(None),
        ).returning(TotpCredential.user_id)
    )
    if replaced is None:
        await session.rollback()
        raise HTTPException(HTTP_CONFLICT, ALREADY_ENABLED)
    await session.commit()
    return secret


async def _matched_step(
    session: AsyncSession, box: SecretBox, credential: TotpCredential, code: str
) -> int | None:
    try:
        secret = box.open(credential.user_id, credential.nonce, credential.secret_ciphertext)
    except InvalidTag:
        # A changed key or row: no code can match. Not the caller's failure, so not counted.
        await session.rollback()
        logger.error("Authenticator secret failed to open")
        raise HTTPException(HTTP_INTERNAL_ERROR, UNAVAILABLE) from None
    step = await _database_step(session)
    return totp.matching_step(secret, code, step, credential.last_used_step)


async def _check_code_failures(
    session: AsyncSession, settings: AssistantSettings, user_id: UUID
) -> None:
    await check_failures(
        session,
        user_id,
        FailureKind.TOTP,
        settings.assistant_totp_failure_limit,
        settings.assistant_totp_failure_window_seconds,
    )


async def _add_code_failure(
    session: AsyncSession, settings: AssistantSettings, user_id: UUID
) -> None:
    await add_failure(
        session, user_id, FailureKind.TOTP, settings.assistant_totp_failure_window_seconds
    )


def _spend(user_id: UUID, step: int, **values):
    return (
        update(TotpCredential)
        .where(TotpCredential.user_id == user_id)
        .values(last_used_step=step, **values)
    )


async def activate(
    session: AsyncSession, box: SecretBox, settings: AssistantSettings, user_id: UUID, code: str
) -> None:
    """Enable the enrolled secret with its first matching code."""
    credential = await _locked_credential(session, user_id)
    if credential is None or credential.enabled_at is not None:
        await session.rollback()
        raise HTTPException(HTTP_CONFLICT, ALREADY_ENABLED if credential else NOT_ENROLLED)
    await _check_code_failures(session, settings, user_id)
    matched = await _matched_step(session, box, credential, code)
    if matched is None:
        await _add_code_failure(session, settings, user_id)
        await session.commit()
        raise HTTPException(HTTP_FORBIDDEN, WRONG_CODE)
    await session.execute(_spend(user_id, matched, enabled_at=func.clock_timestamp()))
    await session.commit()


def _denial(resident: Resident, result: ResultCode):
    # No action or device id: the action is not yet proven to belong to the caller.
    return insert(AssistantToolCall).values(
        turn_id=None,
        user_id=resident.user_id,
        apartment_id=resident.apartment_id,
        source="confirmation",
        tool=ToolName.UNLOCK_DOOR,
        decision=Decision.DENIED,
        result_code=result,
        details={},
    )


async def verify(
    session: AsyncSession,
    box: SecretBox,
    settings: AssistantSettings,
    resident: Resident,
    code: str,
) -> None:
    """Spend a matching code, or raise after recording the refusal. Commits either way."""
    credential = await _locked_credential(session, resident.user_id)
    if credential is None or credential.enabled_at is None:
        await session.execute(_denial(resident, ResultCode.STEP_UP_REQUIRED))
        await session.commit()
        raise HTTPException(HTTP_FORBIDDEN, NOT_ENABLED)
    await _check_code_failures(session, settings, resident.user_id)
    matched = await _matched_step(session, box, credential, code)
    if matched is None:
        await _add_code_failure(session, settings, resident.user_id)
        await session.execute(_denial(resident, ResultCode.STEP_UP_FAILED))
        await session.commit()
        raise HTTPException(HTTP_FORBIDDEN, WRONG_CODE)
    await session.execute(_spend(resident.user_id, matched))
    await session.commit()
