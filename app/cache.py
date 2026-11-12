"""Disposable detail cache with Redis-clock absolute freshness deadlines."""

import asyncio
import logging
from uuid import UUID

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.config import Settings
from app.schemas import ItemResponse

logger = logging.getLogger(__name__)
_CACHE_ERRORS = (ConnectionError, RedisTimeoutError, ResponseError, TimeoutError)


class ItemCache:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.ttl_ms = settings.cache_ttl_ms

    @staticmethod
    def _key(owner_id: UUID, item_id: UUID) -> str:
        return f"item:v1:{owner_id}:{item_id}"

    @staticmethod
    def _log_failure(operation: str, error: Exception) -> None:
        logger.warning("Item cache %s failed: %s", operation, type(error).__name__)

    async def get(self, owner_id: UUID, item_id: UUID) -> ItemResponse | None:
        try:
            async with asyncio.timeout(0.2):
                value = await self.redis.get(self._key(owner_id, item_id))
        except _CACHE_ERRORS as error:
            self._log_failure("get", error)
            return None
        if value is None:
            return None
        try:
            item = ItemResponse.model_validate_json(value)
            if item.id == item_id:
                return item
        except (ValidationError, ValueError, TypeError):
            pass
        await self.invalidate(owner_id, item_id)
        return None

    async def start_fill(self) -> int | None:
        try:
            async with asyncio.timeout(0.2):
                seconds, microseconds = await self.redis.time()
            return seconds * 1000 + microseconds // 1000 + self.ttl_ms
        except _CACHE_ERRORS as error:
            self._log_failure("time", error)
            return None

    async def fill(self, owner_id: UUID, item: ItemResponse, expires_at_ms: int) -> None:
        value = item.model_dump_json()
        try:
            async with asyncio.timeout(0.2):
                await self.redis.set(self._key(owner_id, item.id), value, pxat=expires_at_ms)
        except _CACHE_ERRORS as error:
            self._log_failure("fill", error)

    async def invalidate(self, owner_id: UUID, item_id: UUID) -> None:
        try:
            async with asyncio.timeout(0.2):
                await self.redis.delete(self._key(owner_id, item_id))
        except _CACHE_ERRORS as error:
            self._log_failure("invalidate", error)
