"""Bounded password work and stateless bearer-token authentication."""

import asyncio
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Annotated
from uuid import UUID

import anyio
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.config import Settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


class PasswordService:
    """Keep permits occupied until admitted Argon2 workers have actually finished."""

    def __init__(self, settings: Settings) -> None:
        self._hasher = PasswordHash((Argon2Hasher(),))
        self._semaphore = asyncio.Semaphore(settings.auth_hash_concurrency)
        self._queue_timeout = settings.auth_hash_queue_timeout_seconds
        self._dummy_hash: str | None = None

    @property
    def dummy_hash(self) -> str:
        if self._dummy_hash is None:
            raise RuntimeError("PasswordService must be initialized before use")
        return self._dummy_hash

    async def initialize(self) -> None:
        if self._dummy_hash is None:
            self._dummy_hash = await self.hash(secrets.token_hex(32))

    async def hash(self, password: str) -> str:
        return await self._run(partial(self._hasher.hash, password))

    async def verify(self, password: str, password_hash: str) -> bool:
        return await self._run(partial(self._hasher.verify, password, password_hash))

    async def _worker[T](self, operation: Callable[[], T]) -> T:
        try:
            with anyio.CancelScope(shield=True):
                return await anyio.to_thread.run_sync(operation, abandon_on_cancel=False)
        finally:
            self._semaphore.release()

    async def _run[T](self, operation: Callable[[], T]) -> T:
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._semaphore.acquire()
        except TimeoutError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable",
                headers={"Retry-After": "1"},
            ) from error

        # A separate task owns the permit. asyncio.shield protects that task even
        # from bare Task.cancel(), which an AnyIO cancel scope alone cannot shield.
        worker = asyncio.create_task(self._worker(operation))
        cancellation: asyncio.CancelledError | None = None
        with anyio.CancelScope(shield=True):
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError as error:
                    cancellation = error
            if cancellation is not None:
                # Retrieve any worker exception without masking the caller's
                # cancellation; no hashing work can outlive this request's wait.
                if not worker.cancelled():
                    worker.exception()
                raise cancellation
            return worker.result()


def create_access_token(user_id: UUID, settings: Settings) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(seconds=settings.auth_token_ttl_seconds),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        bytes.fromhex(settings.jwt_secret),
        algorithm="HS256",
    )


def _invalid_authentication() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_id(
    request: Request,
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> UUID:
    if token is None:
        raise _invalid_authentication()
    settings: Settings = request.app.state.settings
    try:
        claims = jwt.decode(
            token,
            bytes.fromhex(settings.jwt_secret),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["sub", "iat", "exp", "iss", "aud"]},
        )
        subject = claims["sub"]
        user_id = UUID(subject)
        if str(user_id) != subject:
            raise ValueError("Noncanonical subject")
        return user_id
    except (jwt.InvalidTokenError, ValueError, TypeError, AttributeError, OverflowError) as error:
        raise _invalid_authentication() from error
