"""Confirming a pending unlock. Only the resident's own request can run it; no tool can.

The router first runs step_up.verify, which spends a matching authenticator code in its
own committed transaction. Then, in one transaction:
  claim own pending, unexpired row while still a resident  -- won  --> unlock + audit -> 200
  otherwise diagnose:  missing, foreign or membership gone -------> 404
                       expired (mark it, commit)            -------> 409
                       already decided                      -------> 409
"""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.contracts import (
    ConfirmedDevice,
    ConfirmResponse,
    Decision,
    ResultCode,
    ToolName,
)
from app.assistant.residence import Resident, resident_member
from app.models import AssistantToolCall, Device, PendingAction

_NOT_FOUND = "Action not found"
_EXPIRED = "Action expired"
_DECIDED = "Action already decided"


async def _diagnose(session: AsyncSession, resident: Resident, action_id: UUID) -> None:
    """The claim matched nothing: say why without revealing other users' rows."""
    row = (
        await session.execute(
            select(
                PendingAction.status,
                (PendingAction.expires_at <= func.clock_timestamp()).label("late"),
            ).where(
                PendingAction.id == action_id,
                PendingAction.user_id == resident.user_id,
                PendingAction.apartment_id == resident.apartment_id,
            )
        )
    ).one_or_none()
    member = row is not None and await session.scalar(select(resident_member(resident)))
    if not member:
        await session.rollback()
        raise HTTPException(404, _NOT_FOUND)
    if row.status == "pending" and row.late:
        await session.execute(
            update(PendingAction)
            .where(PendingAction.id == action_id, PendingAction.status == "pending")
            .values(status="expired", decided_at=func.clock_timestamp())
        )
        await session.commit()
        raise HTTPException(409, _EXPIRED)
    await session.rollback()
    if row.status == "expired":
        raise HTTPException(409, _EXPIRED)
    raise HTTPException(409, _DECIDED)


async def confirm_action(
    session: AsyncSession, resident: Resident, action_id: UUID
) -> ConfirmResponse:
    claimed = (
        await session.execute(
            update(PendingAction)
            .where(
                PendingAction.id == action_id,
                PendingAction.user_id == resident.user_id,
                PendingAction.apartment_id == resident.apartment_id,
                PendingAction.status == "pending",
                PendingAction.expires_at > func.clock_timestamp(),
                resident_member(resident),
            )
            .values(status="confirmed", decided_at=func.clock_timestamp())
            .returning(PendingAction.device_id, PendingAction.action)
        )
    ).one_or_none()
    if claimed is None:
        await _diagnose(session, resident, action_id)
    # The composite FK pins the device to the action's apartment.
    name = await session.scalar(
        update(Device)
        .where(Device.id == claimed.device_id, Device.apartment_id == resident.apartment_id)
        .values(locked=False, updated_at=func.clock_timestamp())
        .returning(Device.name)
    )
    await session.execute(
        insert(AssistantToolCall).values(
            turn_id=None,
            user_id=resident.user_id,
            apartment_id=resident.apartment_id,
            source="confirmation",
            tool=ToolName.UNLOCK_DOOR,
            decision=Decision.CONFIRMED,
            result_code=ResultCode.OK,
            device_id=claimed.device_id,
            pending_action_id=action_id,
            details={},
        )
    )
    await session.commit()
    return ConfirmResponse(
        id=action_id,
        action=claimed.action,
        status="confirmed",
        device=ConfirmedDevice(name=name, locked=False),
    )
