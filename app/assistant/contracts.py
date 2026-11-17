"""Public assistant and residence schemas, tool argument models, and fixed codes."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, StringConstraints

from app.schemas import StrictSchema, reject_nul

# Thermostat limits the assistant may set; the database has a wider hard bound.
THERMOSTAT_MIN_CELSIUS = 16.0
THERMOSTAT_MAX_CELSIUS = 26.0
THERMOSTAT_STEP_CELSIUS = 0.5

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


class ToolName(StrEnum):
    """The whole tool allowlist. Nothing else can act for the resident."""

    SEARCH_DOCUMENTS = "search_documents"
    LIST_BOOKINGS = "list_bookings"
    CREATE_TICKET = "create_ticket"
    SET_THERMOSTAT = "set_thermostat"
    LOCK_DOOR = "lock_door"
    UNLOCK_DOOR = "unlock_door"


# Audited in place of any name outside the allowlist; the model's string is never stored.
UNKNOWN_TOOL = "unknown"


class Decision(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"


class ResultCode(StrEnum):
    OK = "ok"
    DUPLICATE = "duplicate"
    ALREADY_PENDING = "already_pending"
    NOT_FOUND = "not_found"
    NOT_PERMITTED = "not_permitted"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"
    LIMIT_EXCEEDED = "limit_exceeded"
    NO_DOCUMENTS = "no_documents"


# Tool results that ground an answer on their own. Search grounds only through citations.
GROUNDING_RESULTS = frozenset(
    {
        (Decision.ALLOWED, ResultCode.OK),
        (Decision.ALLOWED, ResultCode.DUPLICATE),
        (Decision.PENDING_CONFIRMATION, ResultCode.OK),
        (Decision.PENDING_CONFIRMATION, ResultCode.ALREADY_PENDING),
    }
)


class TicketCategory(StrEnum):
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    HEATING = "heating"
    APPLIANCE = "appliance"
    OTHER = "other"


def _text(maximum: int):
    return Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=maximum),
        AfterValidator(reject_nul),
    ]


DeviceName = _text(100)


class SearchArguments(StrictSchema):
    question: _text(500)


class NoArguments(StrictSchema):
    pass


class TicketArguments(StrictSchema):
    category: TicketCategory
    summary: _text(200)


class ThermostatArguments(StrictSchema):
    device: DeviceName
    celsius: float = Field(
        ge=THERMOSTAT_MIN_CELSIUS,
        le=THERMOSTAT_MAX_CELSIUS,
        multiple_of=THERMOSTAT_STEP_CELSIUS,
        allow_inf_nan=False,
        # A number, never a numeric string.
        strict=True,
    )


class DeviceArguments(StrictSchema):
    device: DeviceName


Message = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    AfterValidator(reject_nul),
]


class MessageRequest(StrictSchema):
    message: Message = Field(repr=False)


class ActionResult(StrictSchema):
    """One tool call of the turn, built by the server from the tool's result."""

    tool: str
    decision: Decision
    result: ResultCode
    device: str | None = None
    ticket_id: UUID | None = None
    pending_action_id: UUID | None = None


class Citation(StrictSchema):
    document_id: UUID
    title: str
    ordinal: int
    page: int | None
    snippet: str


class TurnResponse(StrictSchema):
    turn_id: UUID
    status: Literal["answered", "no_context", "incomplete"]
    answer: str
    citations: list[Citation]
    actions: list[ActionResult]
    pending_actions: list[PendingActionView]


class ConfirmedDevice(StrictSchema):
    name: str
    locked: bool


class ConfirmResponse(StrictSchema):
    id: UUID
    action: Literal["unlock_door"]
    status: Literal["confirmed"]
    device: ConfirmedDevice
