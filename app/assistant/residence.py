"""Resident identity and residence data. Every statement filters by the server-loaded user.

The model never supplies a user, apartment or building: routes and tools take them from
`Resident`, which comes from the caller's verified token and their membership row.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import exists, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.contracts import (
    ApartmentView,
    BookingView,
    BuildingView,
    DeviceView,
    PendingActionView,
    ResidenceView,
    Role,
    TicketView,
)
from app.models import Apartment, Booking, Building, Device, Membership, PendingAction, Ticket

RESIDENCE_LIST_LIMIT = 10
DEMO_BUILDING = "Demo residence"
DEMO_APARTMENT = "1A"
# Name, kind and state of each simulated demo device.
DEMO_DEVICES = (
    {"name": "Living room thermostat", "kind": "thermostat", "target_celsius": Decimal("21.0")},
    {"name": "Bedroom thermostat", "kind": "thermostat", "target_celsius": Decimal("19.5")},
    {"name": "Front door", "kind": "door_lock", "locked": True},
    {"name": "Balcony door", "kind": "door_lock", "locked": True},
    {"name": "Alarm", "kind": "alarm", "armed": True},
)
# Amenity, start offset from now, length.
DEMO_BOOKINGS = (
    ("Gym", timedelta(days=1), timedelta(hours=1)),
    ("Party room", timedelta(days=7), timedelta(hours=4)),
)


@dataclass(frozen=True)
class Resident:
    user_id: UUID
    apartment_id: UUID
    building_id: UUID
    apartment_label: str
    building_name: str


async def load_resident(session: AsyncSession, user_id: UUID) -> Resident | None:
    row = (
        await session.execute(
            select(Membership.apartment_id, Membership.building_id, Apartment.label, Building.name)
            .join(Apartment, Apartment.id == Membership.apartment_id)
            .join(Building, Building.id == Membership.building_id)
            .where(Membership.user_id == user_id, Membership.role == Role.RESIDENT)
        )
    ).one_or_none()
    if row is None:
        return None
    return Resident(
        user_id=user_id,
        apartment_id=row.apartment_id,
        building_id=row.building_id,
        apartment_label=row.label,
        building_name=row.name,
    )


def resident_member(resident: Resident):
    """SQL condition: the resident still lives in the apartment. Used inside writes."""
    return exists().where(
        Membership.user_id == resident.user_id,
        Membership.apartment_id == resident.apartment_id,
        Membership.role == Role.RESIDENT,
    )


async def is_manager(session: AsyncSession, user_id: UUID, building_id: UUID) -> bool:
    return bool(
        await session.scalar(
            select(
                exists().where(
                    Membership.user_id == user_id,
                    Membership.building_id == building_id,
                    Membership.role == Role.MANAGER,
                )
            )
        )
    )


def _device_view(device: Device) -> DeviceView:
    return DeviceView(
        id=device.id,
        name=device.name,
        kind=device.kind,
        target_celsius=float(device.target_celsius) if device.target_celsius is not None else None,
        locked=device.locked,
        armed=device.armed,
    )


async def residence_view(session: AsyncSession, resident: Resident) -> ResidenceView:
    apartment = resident.apartment_id
    devices = await session.scalars(
        select(Device).where(Device.apartment_id == apartment).order_by(Device.name)
    )
    bookings = await session.scalars(
        select(Booking)
        .where(Booking.apartment_id == apartment, Booking.ends_at > func.now())
        .order_by(Booking.starts_at)
        .limit(RESIDENCE_LIST_LIMIT)
    )
    tickets = await session.scalars(
        select(Ticket)
        .where(Ticket.apartment_id == apartment)
        .order_by(Ticket.created_at.desc(), Ticket.id.desc())
        .limit(RESIDENCE_LIST_LIMIT)
    )
    pending = await session.execute(
        select(PendingAction.id, PendingAction.action, PendingAction.expires_at, Device.name)
        .join(Device, Device.id == PendingAction.device_id)
        .where(
            PendingAction.apartment_id == apartment,
            PendingAction.user_id == resident.user_id,
            PendingAction.status == "pending",
            PendingAction.expires_at > func.clock_timestamp(),
        )
        .order_by(PendingAction.created_at)
    )
    return ResidenceView(
        building=BuildingView(id=resident.building_id, name=resident.building_name),
        apartment=ApartmentView(id=apartment, label=resident.apartment_label),
        manager=await is_manager(session, resident.user_id, resident.building_id),
        devices=[_device_view(device) for device in devices],
        bookings=[BookingView.model_validate(booking) for booking in bookings],
        tickets=[TicketView.model_validate(ticket) for ticket in tickets],
        pending_actions=[
            PendingActionView(
                id=row.id, action=row.action, device=row.name, expires_at=row.expires_at
            )
            for row in pending
        ],
    )


def _already_resident() -> HTTPException:
    return HTTPException(409, "Account already has a residence")


async def create_demo_residence(session: AsyncSession, user_id: UUID) -> Resident:
    """Give the caller a demo building, apartment, devices and bookings, as resident and manager."""
    if await load_resident(session, user_id) is not None:
        await session.rollback()
        raise _already_resident()
    now: datetime = await session.scalar(select(func.now()))
    building_id, apartment_id = uuid4(), uuid4()
    place = {"building_id": building_id, "apartment_id": apartment_id}
    try:
        await session.execute(insert(Building).values(id=building_id, name=DEMO_BUILDING))
        await session.execute(
            insert(Apartment).values(id=apartment_id, building_id=building_id, label=DEMO_APARTMENT)
        )
        await session.execute(
            insert(Membership),
            [
                {"user_id": user_id, "role": Role.RESIDENT, **place},
                {"user_id": user_id, "role": Role.MANAGER, "building_id": building_id},
            ],
        )
        await session.execute(insert(Device), [{**device, **place} for device in DEMO_DEVICES])
        await session.execute(
            insert(Booking),
            [
                {
                    "user_id": user_id,
                    "amenity": amenity,
                    "starts_at": now + offset,
                    "ends_at": now + offset + length,
                    **place,
                }
                for amenity, offset, length in DEMO_BOOKINGS
            ],
        )
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        # A concurrent demo request for the same account won the one-residence index.
        if "uq_memberships_one_residence" in str(error.orig):
            raise _already_resident() from None
        raise
    return Resident(
        user_id=user_id,
        apartment_id=apartment_id,
        building_id=building_id,
        apartment_label=DEMO_APARTMENT,
        building_name=DEMO_BUILDING,
    )
