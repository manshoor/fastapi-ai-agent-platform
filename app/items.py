"""Owner-scoped item CRUD, immutable keyset pagination, and detail caching."""

import base64
import binascii
import json
import re
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import ItemCache
from app.db import get_session
from app.models import Item
from app.schemas import ItemCreate, ItemPage, ItemPatch, ItemResponse
from app.security import get_current_user_id

router = APIRouter(prefix="/api/v1/items", tags=["items"])
Session = Annotated[AsyncSession, Depends(get_session)]
Owner = Annotated[UUID, Depends(get_current_user_id)]


def _cache(request: Request) -> ItemCache:
    return request.app.state.item_cache


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Item not found")


def _encode_cursor(item: ItemResponse) -> str:
    payload = json.dumps(
        [item.created_at.astimezone(UTC).isoformat(), str(item.id)], separators=(",", ":")
    ).encode()
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        if not 1 <= len(cursor) <= 512 or re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None:
            raise ValueError("Invalid encoding")
        raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != cursor:
            raise ValueError("Noncanonical encoding")
        values = json.loads(raw)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("Invalid fields")
        timestamp, identifier = values
        if not isinstance(timestamp, str) or not isinstance(identifier, str):
            raise ValueError("Invalid field types")
        created_at = datetime.fromisoformat(timestamp)
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("Timestamp requires timezone")
        item_id = UUID(identifier)
        if str(item_id) != identifier:
            raise ValueError("Noncanonical UUID")
        return created_at.astimezone(UTC), item_id
    except (ValueError, TypeError, OverflowError, binascii.Error) as error:
        raise HTTPException(status_code=422, detail="Invalid cursor") from error


@router.post("", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def create_item(payload: ItemCreate, owner_id: Owner, session: Session) -> ItemResponse:
    item = Item(owner_id=owner_id, **payload.model_dump())
    session.add(item)
    await session.flush()
    response = ItemResponse.model_validate(item)
    await session.commit()
    return response


@router.get("", response_model=ItemPage)
async def list_items(
    owner_id: Owner,
    session: Session,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
) -> ItemPage:
    statement = select(Item).where(Item.owner_id == owner_id)
    if cursor is not None:
        created_at, item_id = _decode_cursor(cursor)
        statement = statement.where(tuple_(Item.created_at, Item.id) < tuple_(created_at, item_id))
    statement = statement.order_by(Item.created_at.desc(), Item.id.desc()).limit(limit + 1)
    rows = (await session.scalars(statement)).all()
    items = [ItemResponse.model_validate(item) for item in rows[:limit]]
    next_cursor = _encode_cursor(items[-1]) if len(rows) > limit else None
    await session.rollback()
    return ItemPage(items=items, next_cursor=next_cursor)


@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(
    item_id: UUID, request: Request, owner_id: Owner, session: Session
) -> ItemResponse:
    cache = _cache(request)
    cached = await cache.get(owner_id, item_id)
    if cached is not None:
        return cached
    # Establish the absolute Redis deadline before the READ COMMITTED database read.
    expires_at_ms = await cache.start_fill()
    item = await session.scalar(select(Item).where(Item.owner_id == owner_id, Item.id == item_id))
    if item is None:
        raise _not_found()
    response = ItemResponse.model_validate(item)
    await session.rollback()
    if expires_at_ms is not None:
        await cache.fill(owner_id, response, expires_at_ms)
    return response


@router.patch("/{item_id}", response_model=ItemResponse)
async def patch_item(
    item_id: UUID, payload: ItemPatch, request: Request, owner_id: Owner, session: Session
) -> ItemResponse:
    statement = (
        update(Item)
        .where(Item.owner_id == owner_id, Item.id == item_id)
        .values(**payload.model_dump(exclude_unset=True), updated_at=func.now())
        .returning(Item)
    )
    item = await session.scalar(statement)
    if item is None:
        raise _not_found()
    response = ItemResponse.model_validate(item)
    await session.commit()
    await _cache(request).invalidate(owner_id, item_id)
    return response


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(
    item_id: UUID, request: Request, owner_id: Owner, session: Session
) -> Response:
    statement = delete(Item).where(Item.owner_id == owner_id, Item.id == item_id).returning(Item.id)
    deleted_id = await session.scalar(statement)
    if deleted_id is None:
        raise _not_found()
    await session.commit()
    await _cache(request).invalidate(owner_id, item_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
