"""Public assistant and residence schemas, tool argument models, and fixed codes."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, StringConstraints

from app.schemas import Password, StrictSchema, reject_nul

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
    # Offered only while assistant memory is on.
    REMEMBER = "remember"
    FORGET = "forget"


MEMORY_TOOLS = frozenset({ToolName.REMEMBER, ToolName.FORGET})
# The only tools that still run once a search handed out passages. A tool added later is
# refused there until it is listed here.
READ_TOOLS = frozenset({ToolName.SEARCH_DOCUMENTS, ToolName.LIST_BOOKINGS})


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
    # A confirmation refused before its action is claimed: no enabled authenticator, or
    # a code that did not match.
    STEP_UP_REQUIRED = "step_up_required"
    STEP_UP_FAILED = "step_up_failed"


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
    # The verified word-for-word quotes the reply takes from this passage.
    quotes: list[str]


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


# Six ASCII digits. [0-9], not \d: \d also matches digits of other scripts.
TotpCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]


class PasswordRequest(StrictSchema):
    password: Password = Field(repr=False)


class CodeRequest(StrictSchema):
    code: TotpCode = Field(repr=False)


class TotpEnrollment(StrictSchema):
    """Shown once: the secret for the authenticator app, as base32 and as an otpauth URI."""

    secret: str = Field(repr=False)
    otpauth_uri: str = Field(repr=False)


class TotpStatus(StrictSchema):
    enabled: bool


class MemoryScope(StrEnum):
    # Follows the user wherever they live.
    RESIDENT = "resident"
    # Holds only while the user lives in the building it was saved in.
    RESIDENCE = "residence"


MAX_MEMORY_CHARACTERS = 300
# A reply quotes at least three words of a memory, so a shorter one could never be cited.
MIN_MEMORY_WORDS = 3


def _enough_words(value: str) -> str:
    if len(value.split()) < MIN_MEMORY_WORDS:
        raise ValueError("A memory needs at least three words")
    return value


MemoryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=MAX_MEMORY_CHARACTERS),
    AfterValidator(reject_nul),
    AfterValidator(_enough_words),
]


class MemoryRequest(StrictSchema):
    text: MemoryText = Field(repr=False)
    scope: MemoryScope


class MemoryView(StrictSchema):
    id: UUID
    scope: MemoryScope
    text: str = Field(repr=False)
    created_at: datetime
    expires_at: datetime


class MemoryList(StrictSchema):
    """Servable memories, newest first: unexpired, and residence ones of the current home."""

    memories: list[MemoryView]


# A recalled memory's label: M1..M20, one per memory a turn may recall.
MEMORY_LABEL_PATTERN = r"^M(?:[1-9]|1[0-9]|20)$"
MemoryLabel = Annotated[str, StringConstraints(pattern=MEMORY_LABEL_PATTERN)]


class RememberArguments(StrictSchema):
    # Checked server-side: a word-for-word span of the resident's current message.
    text: MemoryText
    scope: MemoryScope


class ForgetArguments(StrictSchema):
    label: MemoryLabel
    # Checked server-side: a word-for-word span of the resident's current message.
    request_quote: _text(MAX_MEMORY_CHARACTERS)
