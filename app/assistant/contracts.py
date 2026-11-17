"""Public assistant and residence schemas, tool argument models, and fixed codes."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from app.schemas import StrictSchema

DeviceKind = Literal["thermostat", "door_lock", "alarm"]


class Role(StrEnum):
    RESIDENT = "resident"
    MANAGER = "manager"


class BuildingView(StrictSchema):
    id: UUID
    name: str


class ApartmentView(StrictSchema):
    id: UUID
    label: str


class DeviceView(StrictSchema):
    id: UUID
    name: str
    kind: DeviceKind
    target_celsius: float | None
    locked: bool | None
    armed: bool | None


class BookingView(StrictSchema):
    id: UUID
    amenity: str
    starts_at: datetime
    ends_at: datetime


class TicketView(StrictSchema):
    id: UUID
    category: str
    summary: str
    status: str
    created_at: datetime


class PendingActionView(StrictSchema):
    id: UUID
    action: Literal["unlock_door"]
    device: str
    expires_at: datetime


class ResidenceView(StrictSchema):
    building: BuildingView
    apartment: ApartmentView
    manager: bool
    devices: list[DeviceView]
    bookings: list[BookingView]
    tickets: list[TicketView]
    pending_actions: list[PendingActionView]
