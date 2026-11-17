"""The tool allowlist, its server-side checks, and the audit trail.

    model tool call -> allowlisted name? -- no --> audit "unknown" denial
                    -> arguments valid?  -- no --> audit invalid_arguments
                    -> handler: read tools read and audit;
                       write tools: one transaction =
                         per-user lock -> limits -> guarded write (membership in WHERE)
                         -> audit row
                    -> result JSON back to the model

The model supplies only names and values. User, apartment and building come from
`Resident`. Every time and window comes from the database clock.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy import func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.processing import SessionFactory
from app.ai.provider import MAX_PASSAGES, ToolCall
from app.assistant.config import AssistantSettings
from app.assistant.contracts import (
    UNKNOWN_TOOL,
    ActionResult,
    Decision,
    DeviceArguments,
    NoArguments,
    ResultCode,
    SearchArguments,
    ThermostatArguments,
    TicketArguments,
    ToolName,
)
from app.assistant.residence import Resident, resident_member
from app.models import AssistantToolCall, Booking, Device, Membership, PendingAction, Ticket
from app.rag.retrieval import BuildingScopedRetriever, has_building_context

# Transaction advisory lock namespace for one user's tool writes ("ASTW").
_WRITE_LOCK_NAMESPACE = 0x41535457
SEARCH_TOP_K = 4
BOOKING_LIMIT = 10
DEVICE_TOOLS = (ToolName.SET_THERMOSTAT, ToolName.LOCK_DOOR)


@dataclass
class ToolContext:
    """Everything a tool may use. Built by the server for one turn."""

    session_factory: SessionFactory
    settings: AssistantSettings
    resident: Resident
    turn_id: UUID
    embedder: Any
    telemetry: Any = None
    # Passages already given to the model this turn; labels P1..P8 continue across searches.
    passages: list[dict] = field(default_factory=list)
    embedding_tokens: int = 0


@dataclass(frozen=True)
class ToolOutcome:
    action: ActionResult
    # Sent back to the model as the function call output.
    output: dict


def _outcome(tool: str, decision: Decision, result: ResultCode, **extra) -> ToolOutcome:
    output = {"result": str(result), **extra.pop("output", {})}
    return ToolOutcome(
        action=ActionResult(tool=tool, decision=decision, result=result, **extra), output=output
    )


def _audit(
    ctx: ToolContext,
    tool: str,
    decision: Decision,
    result: ResultCode,
    *,
    details: dict | None = None,
    **ids,
):
    return insert(AssistantToolCall).values(
        turn_id=ctx.turn_id,
        user_id=ctx.resident.user_id,
        apartment_id=ctx.resident.apartment_id,
        source="assistant",
        tool=tool,
        decision=decision,
        result_code=result,
        details=details or {},
        **ids,
    )


async def _audited(ctx: ToolContext, tool: str, decision: Decision, result: ResultCode, **kw):
    """Record a call that changed nothing, in its own short transaction."""
    async with ctx.session_factory() as session:
        await session.execute(_audit(ctx, tool, decision, result, **kw))
        await session.commit()


async def _lock_user(session: AsyncSession, ctx: ToolContext) -> None:
    await session.execute(
        text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:user))"),
        {"namespace": _WRITE_LOCK_NAMESPACE, "user": str(ctx.resident.user_id)},
    )


def _since(seconds: float):
    return func.clock_timestamp() - timedelta(seconds=seconds)


async def _is_member(session: AsyncSession, ctx: ToolContext) -> bool:
    return bool(await session.scalar(select(resident_member(ctx.resident))))


async def _hold_membership(session: AsyncSession, ctx: ToolContext) -> bool:
    """Share-lock the resident's membership: a revocation waits until this write commits."""
    resident = ctx.resident
    held = await session.scalar(
        select(Membership.id)
        .where(
            Membership.user_id == resident.user_id,
            Membership.apartment_id == resident.apartment_id,
            Membership.role == "resident",
        )
        .with_for_update(read=True)
    )
    return held is not None


async def _deny(
    session: AsyncSession, ctx: ToolContext, tool: str, result: ResultCode, **extra
) -> ToolOutcome:
    """Audit a denial inside the open write transaction, then commit it."""
    await session.execute(_audit(ctx, tool, Decision.DENIED, result))
    await session.commit()
    return _outcome(tool, Decision.DENIED, result, **extra)


async def _missing_device(session: AsyncSession, ctx: ToolContext, tool: str, name: str):
    # A guarded write matched nothing: the resident left, or no such device of that kind.
    result = ResultCode.NOT_FOUND if await _is_member(session, ctx) else ResultCode.NOT_PERMITTED
    return await _deny(session, ctx, tool, result, device=name)


async def search_documents(ctx: ToolContext, args: SearchArguments) -> ToolOutcome:
    tool = ToolName.SEARCH_DOCUMENTS
    remaining = MAX_PASSAGES - len(ctx.passages)
    if remaining <= 0:
        await _audited(ctx, tool, Decision.DENIED, ResultCode.LIMIT_EXCEEDED)
        return _outcome(tool, Decision.DENIED, ResultCode.LIMIT_EXCEEDED)
    settings, resident = ctx.settings, ctx.resident
    async with ctx.session_factory() as session:
        member = await _is_member(session, ctx)
        await session.rollback()
    if not member:
        await _audited(ctx, tool, Decision.DENIED, ResultCode.NOT_PERMITTED)
        return _outcome(tool, Decision.DENIED, ResultCode.NOT_PERMITTED)
    # No ready building chunk: no paid embedding call.
    if not await has_building_context(
        session_factory=ctx.session_factory,
        building_id=resident.building_id,
        embedding_model=settings.ai_embedding_model,
    ):
        await _audited(ctx, tool, Decision.ALLOWED, ResultCode.NO_DOCUMENTS)
        return _outcome(tool, Decision.ALLOWED, ResultCode.NO_DOCUMENTS)
    retriever = BuildingScopedRetriever(
        session_factory=ctx.session_factory,
        embedder=ctx.embedder,
        building_id=resident.building_id,
        embedding_model=settings.ai_embedding_model,
        top_k=min(SEARCH_TOP_K, remaining),
        telemetry=ctx.telemetry,
    )
    # A provider failure propagates: the turn records it; there is nothing to audit.
    documents = await retriever.ainvoke(args.question)
    ctx.embedding_tokens += retriever.embedding_tokens or 0
    passages = []
    for document in documents:
        label = f"P{len(ctx.passages) + 1}"
        passage = {"label": label, "document": document}
        ctx.passages.append(passage)
        passages.append(
            {
                "label": label,
                "title": document.metadata["title"],
                "page": document.metadata["page"],
                "text": document.page_content,
            }
        )
    result = ResultCode.OK if passages else ResultCode.NO_DOCUMENTS
    await _audited(ctx, tool, Decision.ALLOWED, result, details={"passage_count": len(passages)})
    return _outcome(tool, Decision.ALLOWED, result, output={"passages": passages})


async def list_bookings(ctx: ToolContext, args: NoArguments) -> ToolOutcome:
    tool = ToolName.LIST_BOOKINGS
    async with ctx.session_factory() as session:
        if not await _is_member(session, ctx):
            return await _deny(session, ctx, tool, ResultCode.NOT_PERMITTED)
        rows = await session.execute(
            select(Booking.amenity, Booking.starts_at, Booking.ends_at)
            .where(
                Booking.apartment_id == ctx.resident.apartment_id,
                Booking.ends_at > func.clock_timestamp(),
            )
            .order_by(Booking.starts_at)
            .limit(BOOKING_LIMIT)
        )
        bookings = [
            {
                "amenity": row.amenity,
                "starts_at": row.starts_at.isoformat(),
                "ends_at": row.ends_at.isoformat(),
            }
            for row in rows
        ]
        await session.execute(_audit(ctx, tool, Decision.ALLOWED, ResultCode.OK))
        await session.commit()
    return _outcome(tool, Decision.ALLOWED, ResultCode.OK, output={"bookings": bookings})


async def create_ticket(ctx: ToolContext, args: TicketArguments) -> ToolOutcome:
    tool, resident, settings = ToolName.CREATE_TICKET, ctx.resident, ctx.settings
    async with ctx.session_factory() as session:
        await _lock_user(session, ctx)
        if not await _hold_membership(session, ctx):
            return await _deny(session, ctx, tool, ResultCode.NOT_PERMITTED)
        # A retried message returns the recent open ticket instead of a second one.
        existing = await session.scalar(
            select(Ticket.id)
            .where(
                Ticket.apartment_id == resident.apartment_id,
                Ticket.category == args.category,
                Ticket.status == "open",
                Ticket.created_at > _since(settings.assistant_ticket_dedupe_seconds),
            )
            .order_by(Ticket.created_at.desc())
            .limit(1)
        )
        details = {"category": str(args.category)}
        if existing is not None:
            await session.execute(
                _audit(
                    ctx,
                    tool,
                    Decision.ALLOWED,
                    ResultCode.DUPLICATE,
                    details=details,
                    ticket_id=existing,
                )
            )
            await session.commit()
            return _outcome(
                tool,
                Decision.ALLOWED,
                ResultCode.DUPLICATE,
                ticket_id=existing,
                output={"ticket_id": str(existing)},
            )
        recent = await session.scalar(
            select(func.count())
            .select_from(Ticket)
            .where(
                Ticket.user_id == resident.user_id,
                Ticket.created_at > _since(settings.assistant_ticket_window_seconds),
            )
        )
        if recent >= settings.assistant_ticket_limit:
            return await _deny(session, ctx, tool, ResultCode.LIMIT_EXCEEDED)
        ticket_id = await session.scalar(
            insert(Ticket)
            .values(
                apartment_id=resident.apartment_id,
                building_id=resident.building_id,
                user_id=resident.user_id,
                category=args.category,
                summary=args.summary,
                created_at=func.clock_timestamp(),
            )
            .returning(Ticket.id)
        )
        await session.execute(
            _audit(ctx, tool, Decision.ALLOWED, ResultCode.OK, details=details, ticket_id=ticket_id)
        )
        await session.commit()
    return _outcome(
        tool,
        Decision.ALLOWED,
        ResultCode.OK,
        ticket_id=ticket_id,
        output={"ticket_id": str(ticket_id)},
    )


async def _device_limit_reached(session: AsyncSession, ctx: ToolContext) -> bool:
    settings = ctx.settings
    recent = await session.scalar(
        select(func.count())
        .select_from(AssistantToolCall)
        .where(
            AssistantToolCall.user_id == ctx.resident.user_id,
            AssistantToolCall.tool.in_([str(tool) for tool in DEVICE_TOOLS]),
            AssistantToolCall.decision == Decision.ALLOWED,
            AssistantToolCall.created_at > _since(settings.assistant_device_action_window_seconds),
        )
    )
    return recent >= settings.assistant_device_action_limit


async def _set_device(
    ctx: ToolContext, tool: ToolName, name: str, kind: str, values: dict, details: dict
) -> ToolOutcome:
    resident = ctx.resident
    async with ctx.session_factory() as session:
        await _lock_user(session, ctx)
        if await _device_limit_reached(session, ctx):
            return await _deny(session, ctx, tool, ResultCode.LIMIT_EXCEEDED, device=name)
        # One guarded statement: right apartment, name and kind, and still a resident.
        device_id = await session.scalar(
            update(Device)
            .where(
                Device.apartment_id == resident.apartment_id,
                Device.name == name,
                Device.kind == kind,
                resident_member(resident),
            )
            .values(**values, updated_at=func.clock_timestamp())
            .returning(Device.id)
        )
        if device_id is None:
            return await _missing_device(session, ctx, tool, name)
        await session.execute(
            _audit(ctx, tool, Decision.ALLOWED, ResultCode.OK, details=details, device_id=device_id)
        )
        await session.commit()
    return _outcome(tool, Decision.ALLOWED, ResultCode.OK, device=name, output={"device": name})


async def set_thermostat(ctx: ToolContext, args: ThermostatArguments) -> ToolOutcome:
    celsius = float(args.celsius)
    return await _set_device(
        ctx,
        ToolName.SET_THERMOSTAT,
        args.device,
        "thermostat",
        {"target_celsius": celsius},
        {"celsius": celsius},
    )


async def lock_door(ctx: ToolContext, args: DeviceArguments) -> ToolOutcome:
    return await _set_device(
        ctx, ToolName.LOCK_DOOR, args.device, "door_lock", {"locked": True}, {}
    )


async def unlock_door(ctx: ToolContext, args: DeviceArguments) -> ToolOutcome:
    """Never unlocks: records a pending action that only the resident's confirm runs."""
    tool, resident, settings, name = ToolName.UNLOCK_DOOR, ctx.resident, ctx.settings, args.device
    async with ctx.session_factory() as session:
        await _lock_user(session, ctx)
        if not await _hold_membership(session, ctx):
            return await _deny(session, ctx, tool, ResultCode.NOT_PERMITTED, device=name)
        recent = await session.scalar(
            select(func.count())
            .select_from(PendingAction)
            .where(
                PendingAction.user_id == resident.user_id,
                PendingAction.created_at > _since(settings.assistant_unlock_window_seconds),
            )
        )
        if recent >= settings.assistant_unlock_limit:
            return await _deny(session, ctx, tool, ResultCode.LIMIT_EXCEEDED, device=name)
        device_id = await session.scalar(
            select(Device.id).where(
                Device.apartment_id == resident.apartment_id,
                Device.name == name,
                Device.kind == "door_lock",
                resident_member(resident),
            )
        )
        if device_id is None:
            return await _missing_device(session, ctx, tool, name)
        # A stale request must not block a new one.
        await session.execute(
            update(PendingAction)
            .where(
                PendingAction.device_id == device_id,
                PendingAction.status == "pending",
                PendingAction.expires_at <= func.clock_timestamp(),
            )
            .values(status="expired", decided_at=func.clock_timestamp())
        )
        # Another resident of the apartment may hold the live request: no error, no abort.
        action_id = await session.scalar(
            pg_insert(PendingAction)
            .values(
                user_id=resident.user_id,
                apartment_id=resident.apartment_id,
                building_id=resident.building_id,
                device_id=device_id,
                turn_id=ctx.turn_id,
                action="unlock_door",
                created_at=func.clock_timestamp(),
                expires_at=func.clock_timestamp()
                + timedelta(seconds=settings.assistant_unlock_ttl_seconds),
            )
            .on_conflict_do_nothing(
                index_elements=["device_id"], index_where=text("status = 'pending'")
            )
            .returning(PendingAction.id)
        )
        result = ResultCode.OK
        if action_id is None:
            result = ResultCode.ALREADY_PENDING
            # Only the caller's own live request is reported; a co-resident's stays private.
            action_id = await session.scalar(
                select(PendingAction.id).where(
                    PendingAction.device_id == device_id,
                    PendingAction.user_id == resident.user_id,
                    PendingAction.status == "pending",
                    PendingAction.expires_at > func.clock_timestamp(),
                )
            )
        decision = Decision.PENDING_CONFIRMATION if action_id is not None else Decision.DENIED
        await session.execute(
            _audit(ctx, tool, decision, result, device_id=device_id, pending_action_id=action_id)
        )
        await session.commit()
    output = {"device": name}
    if action_id is not None:
        output["status"] = "confirmation_required"
    return _outcome(tool, decision, result, device=name, pending_action_id=action_id, output=output)


@dataclass(frozen=True)
class _Spec:
    arguments: type[BaseModel]
    handler: Callable[[ToolContext, Any], Awaitable[ToolOutcome]]
    description: str
    # Provider-facing parameters: types, enums and required fields only. The server's
    # argument models enforce every limit.
    properties: dict


_STRING = {"type": "string"}
_SPECS: dict[str, _Spec] = {
    ToolName.SEARCH_DOCUMENTS: _Spec(
        SearchArguments,
        search_documents,
        "Search the resident's building documents (house rules, FAQs, guides).",
        {"question": _STRING},
    ),
    ToolName.LIST_BOOKINGS: _Spec(
        NoArguments, list_bookings, "List the apartment's upcoming amenity bookings.", {}
    ),
    ToolName.CREATE_TICKET: _Spec(
        TicketArguments,
        create_ticket,
        "Open a maintenance ticket for the apartment.",
        {
            "category": {
                "type": "string",
                "enum": ["plumbing", "electrical", "heating", "appliance", "other"],
            },
            "summary": _STRING,
        },
    ),
    ToolName.SET_THERMOSTAT: _Spec(
        ThermostatArguments,
        set_thermostat,
        "Set a thermostat's target temperature in degrees Celsius (16 to 26, steps of 0.5).",
        {"device": _STRING, "celsius": {"type": "number"}},
    ),
    ToolName.LOCK_DOOR: _Spec(DeviceArguments, lock_door, "Lock a door.", {"device": _STRING}),
    ToolName.UNLOCK_DOOR: _Spec(
        DeviceArguments,
        unlock_door,
        "Request a door unlock. The resident must confirm it in the app.",
        {"device": _STRING},
    ),
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": str(name),
        "description": spec.description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": spec.properties,
            "required": list(spec.properties),
            "additionalProperties": False,
        },
    }
    for name, spec in _SPECS.items()
]


async def execute(ctx: ToolContext, call: ToolCall) -> ToolOutcome:
    """Run one model tool call under the server's rules. Always audited."""
    spec = _SPECS.get(call.name)
    if spec is None:
        await _audited(ctx, UNKNOWN_TOOL, Decision.DENIED, ResultCode.UNKNOWN_TOOL)
        return _outcome(UNKNOWN_TOOL, Decision.DENIED, ResultCode.UNKNOWN_TOOL)
    tool = ToolName(call.name)
    try:
        args = spec.arguments.model_validate_json(call.arguments)
    except ValidationError:
        await _audited(ctx, tool, Decision.DENIED, ResultCode.INVALID_ARGUMENTS)
        return _outcome(tool, Decision.DENIED, ResultCode.INVALID_ARGUMENTS)
    return await spec.handler(ctx, args)


def output_text(outcome: ToolOutcome) -> str:
    return json.dumps(outcome.output, ensure_ascii=False)
