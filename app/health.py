import asyncio
import logging

import asyncpg
from fastapi import APIRouter, Request
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    dependencies = {"database": "unavailable", "cache": "unavailable"}

    async def check_database() -> None:
        try:
            async with request.app.state.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            dependencies["database"] = "ok"
        except (SQLAlchemyError, asyncpg.PostgresError, OSError) as exc:
            logger.warning("Readiness database failure exception=%s", type(exc).__name__)

    async def check_cache() -> None:
        try:
            async with asyncio.timeout(0.2):
                if await request.app.state.redis.ping():
                    dependencies["cache"] = "ok"
        except (RedisError, OSError) as exc:
            logger.warning("Readiness cache failure exception=%s", type(exc).__name__)

    try:
        async with asyncio.timeout(1):
            await asyncio.gather(check_database(), check_cache())
    except TimeoutError:
        logger.warning("Readiness dependency deadline exceeded")

    if dependencies["database"] != "ok":
        return JSONResponse({"status": "unavailable", **dependencies}, status_code=503)
    status = "ok" if dependencies["cache"] == "ok" else "degraded"
    return JSONResponse({"status": status, **dependencies})
