import asyncio
import errno
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import asyncpg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.ai.config import AISettings
from app.ai.router import router as ai_router
from app.auth import router as auth_router
from app.cache import ItemCache
from app.config import Settings
from app.db import create_engine, create_session_factory
from app.health import router as health_router
from app.items import router as items_router
from app.security import PasswordService
from app.telemetry import configure_telemetry

logger = logging.getLogger(__name__)
BODY_LIMIT_BYTES = 32 * 1024
# Only application-owned labels may be reflected: extra-field locations and
# validator messages can themselves contain attacker-supplied secrets.
_VALIDATION_LOCATIONS = frozenset(
    {
        "body",
        "query",
        "path",
        "header",
        "cookie",
        "email",
        "password",
        "username",
        "title",
        "description",
        "item_id",
        "instruction",
        "job_id",
        "Idempotency-Key",
        "limit",
        "cursor",
    }
)
_VALIDATION_MESSAGES = {
    "missing": "Field required",
    "extra_forbidden": "Extra inputs are not permitted",
    "json_invalid": "Invalid JSON",
    "json_type": "JSON input required",
    "model_attributes_type": "Input must be an object",
    "model_type": "Input must be an object",
    "string_type": "Input must be a string",
    "string_too_short": "String is too short",
    "string_too_long": "String is too long",
    "string_pattern_mismatch": "String does not match the required pattern",
    "int_type": "Input must be an integer",
    "int_parsing": "Input must be a valid integer",
    "int_from_float": "Input must be an integer",
    "greater_than": "Input must exceed the minimum",
    "greater_than_equal": "Input must meet the minimum",
    "less_than": "Input must be below the maximum",
    "less_than_equal": "Input must not exceed the maximum",
    "uuid_type": "Input must be a UUID",
    "uuid_parsing": "Input must be a valid UUID",
    "uuid_version": "Input must use the required UUID version",
    "value_error": "Invalid value",
}
_TRANSIENT_STATES = {"57014", "57P01", "57P02", "57P03", "53300"}
_NETWORK_ERRNOS = {
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.ECONNABORTED,
    errno.ETIMEDOUT,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.EPIPE,
}


class BoundedBodyMiddleware:
    def __init__(self, app: ASGIApp, limit: int = BODY_LIMIT_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        lengths = [value for key, value in scope["headers"] if key == b"content-length"]
        if lengths:
            try:
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError
                declared_length = int(lengths[0])
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return
            if declared_length > self.limit:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.limit:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _is_transient_database_error(exc: Exception) -> bool:
    if isinstance(exc, (PoolTimeoutError, TimeoutError)):
        return True
    if isinstance(exc, DBAPIError) and exc.connection_invalidated:
        return True
    original = exc.orig if isinstance(exc, DBAPIError) else exc
    state = getattr(original, "sqlstate", None)
    if state and (state.startswith("08") or state in _TRANSIENT_STATES):
        return True
    if isinstance(original, (TimeoutError, socket.gaierror)):
        return True
    if isinstance(original, OSError):
        return original.errno in _NETWORK_ERRNOS
    return isinstance(original, asyncpg.PostgresConnectionError)


async def request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Never serialize/log exc, its body, input, ctx, or original messages. Even
    # a validator's message or an unknown JSON key can contain a password/JWT.
    detail = []
    for error in exc.errors():
        error_type = error.get("type")
        safe_type = error_type if error_type in _VALIDATION_MESSAGES else "value_error"
        location = [
            part
            if type(part) is int or (isinstance(part, str) and part in _VALIDATION_LOCATIONS)
            else "[field]"
            for part in error.get("loc", ())
        ]
        detail.append({"type": safe_type, "loc": location, "msg": _VALIDATION_MESSAGES[safe_type]})
    return JSONResponse({"detail": detail}, status_code=422)


async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Exception text and tracebacks may include SQL, parameters, or connection credentials.
    logger.error("Database request failure exception=%s", type(exc).__name__)
    if _is_transient_database_error(exc):
        return JSONResponse(
            {"detail": "Service temporarily unavailable"},
            status_code=503,
            headers={"Retry-After": "1"},
        )
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        engine = None
        redis = None
        telemetry = None

        async def close_resources() -> None:
            try:
                if redis is not None:
                    await redis.aclose()
            finally:
                try:
                    if engine is not None:
                        await engine.dispose()
                finally:
                    if telemetry is not None:
                        await asyncio.to_thread(telemetry.shutdown)

        try:
            try:
                runtime_settings = settings if settings is not None else Settings()
                engine = create_engine(runtime_settings)
                redis = Redis.from_url(
                    runtime_settings.redis_url,
                    max_connections=runtime_settings.redis_max_connections,
                    socket_connect_timeout=runtime_settings.redis_timeout_seconds,
                    socket_timeout=runtime_settings.redis_timeout_seconds,
                    retry=Retry(NoBackoff(), 0),
                    retry_on_timeout=False,
                    decode_responses=False,
                )
                application.state.settings = runtime_settings
                application.state.ai_settings = (
                    AISettings(**runtime_settings.model_dump())
                    if runtime_settings.ai_enabled
                    else None
                )
                telemetry = configure_telemetry("rewrite-api")
                application.state.telemetry = telemetry
                application.state.engine = engine
                application.state.session_factory = create_session_factory(engine)
                application.state.redis = redis
                application.state.item_cache = ItemCache(redis, runtime_settings)
                application.state.passwords = PasswordService(runtime_settings)
                await application.state.passwords.initialize()
            except Exception as exc:
                logger.error("Application startup failure exception=%s", type(exc).__name__)
                # Server lifespan tracebacks must not print URL/settings errors.
                raise RuntimeError("Application startup failed") from None
            yield
        finally:
            # Keep cleanup alive even under bare Task.cancel(), and propagate
            # cancellation only after both infrastructure clients have closed.
            cleanup = asyncio.create_task(close_resources())
            cancellation = None
            with anyio.CancelScope(shield=True):
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                    except Exception:
                        break
                try:
                    cleanup.result()
                except Exception as exc:
                    logger.error("Application shutdown failure exception=%s", type(exc).__name__)
                    if cancellation is None:
                        raise RuntimeError("Application shutdown failed") from None
                if cancellation is not None:
                    raise cancellation

    application = FastAPI(title="Owner-scoped CRUD API", lifespan=lifespan)
    application.add_middleware(BoundedBodyMiddleware)
    application.add_exception_handler(RequestValidationError, request_validation_error_handler)
    for exception_type in (SQLAlchemyError, asyncpg.PostgresError, OSError):
        application.add_exception_handler(exception_type, database_error_handler)
    application.include_router(auth_router)
    application.include_router(items_router)
    application.include_router(ai_router)
    application.include_router(health_router)
    return application


app = create_app()
