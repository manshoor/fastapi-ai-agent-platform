"""Per-account failure throttles for passwords and one-time codes.

    check   count this user's failures of one kind inside the window (database clock)
            -> at the limit: roll back, 429, Retry-After = until the oldest counted
               failure leaves the window
    record  in the caller's short transaction: insert one failure, then prune this
            user's rows of that kind that left the window

Only existing accounts are counted: an unknown email records nothing, so nobody can
fill the counter of an account that does not exist. Rows hold no password, code,
address or token: a user, a kind and a time.
"""

import math
from datetime import timedelta
from enum import StrEnum
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuthFailure

HTTP_TOO_MANY_REQUESTS = 429
THROTTLED_DETAIL = "Too many failed attempts"


class FailureKind(StrEnum):
    PASSWORD = "password"
    TOTP = "totp"


def _window_start(window_seconds: float):
    return func.clock_timestamp() - timedelta(seconds=window_seconds)


async def check_failures(
    session: AsyncSession, user_id: UUID, kind: FailureKind, limit: int, window_seconds: float
) -> None:
    """Raise 429 when this user already has `limit` failures of this kind in the window."""
    count, remaining = (
        await session.execute(
            select(
                func.count(),
                func.extract(
                    "epoch",
                    func.min(AuthFailure.created_at)
                    + timedelta(seconds=window_seconds)
                    - func.clock_timestamp(),
                ),
            ).where(
                AuthFailure.user_id == user_id,
                AuthFailure.kind == kind,
                AuthFailure.created_at > _window_start(window_seconds),
            )
        )
    ).one()
    if count < limit:
        return
    await session.rollback()
    retry_after = max(1, math.ceil(float(remaining or 0)))
    raise HTTPException(
        HTTP_TOO_MANY_REQUESTS, THROTTLED_DETAIL, headers={"Retry-After": str(retry_after)}
    )


async def add_failure(
    session: AsyncSession, user_id: UUID, kind: FailureKind, window_seconds: float
) -> None:
    """Record one failure and prune this user's expired ones. The caller commits."""
    await session.execute(insert(AuthFailure).values(user_id=user_id, kind=kind))
    await session.execute(
        delete(AuthFailure).where(
            AuthFailure.user_id == user_id,
            AuthFailure.kind == kind,
            AuthFailure.created_at <= _window_start(window_seconds),
        )
    )
