"""The tool allowlist, its server-side checks, and the audit trail.

    model tool call -> a search handed out passages and not in READ_TOOLS
                       (unknown names included)?  -- yes --> audit not_permitted
                    -> allowlisted name? -- no --> audit "unknown" denial
                    -> arguments valid?  -- no --> audit invalid_arguments
                    -> handler: read tools read and audit;
                       write tools: one transaction =
                         per-user lock -> limits -> guarded write (membership in WHERE)
                         -> audit row
                       memory tools (only while memory is on):
                         text or request quote must be a span of the current message ->
                         remember: user row lock -> membership -> memory limits -> insert
                         forget: delete the recalled M label's row -> audit row
                    -> result JSON back to the model; a result that grounds a reply
                       carries a server-written fact with the next label, T1..T8

The model supplies only names and values. User, apartment and building come from
`Resident`. Every time and window comes from the database clock.
"""

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, timedelta
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
    GROUNDING_RESULTS,
    MEMORY_TOOLS,
    READ_TOOLS,
    UNKNOWN_TOOL,
    ActionResult,
    Decision,
    DeviceArguments,
    ForgetArguments,
    MemoryScope,
    NoArguments,
    RememberArguments,
    ResultCode,
    SearchArguments,
    ThermostatArguments,
    TicketArguments,
    ToolName,
)
from app.assistant.memory import (
    MemoryBox,
    Recalled,
    Refusal,
    Refused,
    delete_memory,
    lock_writer,
    write_memory,
)
from app.assistant.residence import Resident, resident_member
from app.models import AssistantToolCall, Booking, Device, Membership, PendingAction, Ticket
from app.rag.retrieval import BuildingScopedRetriever, graph_search, has_building_context
from app.rag.verification import MIN_QUOTE_WORDS, normalize, whole_quote

# Transaction advisory lock namespace for one user's tool writes ("ASTW").
_WRITE_LOCK_NAMESPACE = 0x41535457
SEARCH_TOP_K = 4
BOOKING_LIMIT = 10
DEVICE_TOOLS = (ToolName.SET_THERMOSTAT, ToolName.LOCK_DOOR)
# A forget request names its intent in the quoted words, and is not "don't forget".
_FORGET_INTENT = re.compile(r"\b(?:forget|delete|remove|erase)\b")
_NEGATED_FORGET = re.compile(r"\b(?:don't|do not|never|not)\s+(?:forget|delete|remove|erase)\b")
_WORD = re.compile(r"\w+")
# Labels T1..T8, as for passages: a turn makes at most eight model calls.
MAX_FACTS = 8
FACT_PREFIX = "T"
# Sent to the model with a refused change, so it still answers instead of giving up.
REFUSED_AFTER_SEARCH = (
    "No change runs after a search returned passages. Do not retry. Answer from the "
    "passages; the resident is told that the action was not run."
)


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
    # Facts of grounding tool results this turn, in label order: facts[0] is T1. A fact
    # that a later result for the same device replaced is blank and grounds nothing.
    facts: list[str] = field(default_factory=list)
    # (tool, device) -> index of that device's current fact in `facts`.
    _device_facts: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
    # A change was refused because passages were handed out: the reply must say so.
    refused_after_search: bool = False
    embedding_tokens: int = 0
    # The resident's current message: memory tools may store or act only on its words.
    message: str = field(default="", repr=False)
    # Seals memory text; None while memory is off, and then no memory tool runs.
    memory_box: MemoryBox | None = field(default=None, repr=False)
    # Memories recalled for this turn: memories[0] is M1. A forgotten one is blanked,
    # never removed, so labels keep their meaning.
    memories: list[Recalled] = field(default_factory=list, repr=False)


@dataclass(frozen=True)
class ToolOutcome:
    action: ActionResult
    # Sent back to the model as the function call output.
    output: dict
    # The one sentence a reply may quote about this result. The server writes it
    # from checked values only: a stored device name, a category, a number.
    # Only a result in GROUNDING_RESULTS gets a label for it.
    fact: str | None = None


def _outcome(tool: str, decision: Decision, result: ResultCode, **extra) -> ToolOutcome:
    output = {"result": str(result), **extra.pop("output", {})}
    fact = extra.pop("fact", None)
    return ToolOutcome(
        action=ActionResult(tool=tool, decision=decision, result=result, **extra),
        output=output,
        fact=fact,
    )


def _labelled(ctx: ToolContext, outcome: ToolOutcome) -> ToolOutcome:
    """Give a grounding result's fact the next T label, in the output the model reads.

    A later result of the same tool for the same device replaces the earlier fact: 21 °C
    then 23 °C leaves only 23 °C citable. The old fact is blanked, never removed, so
    labels already handed out keep their meaning. Other tools keep their facts: the door
    stays locked while an unlock waits, and a ticket stays opened after a duplicate.
    """
    action = outcome.action
    grounds = (action.decision, action.result) in GROUNDING_RESULTS
    if outcome.fact is None or not grounds or len(ctx.facts) >= MAX_FACTS:
        return outcome
    if action.device is not None:
        subject = (action.tool, action.device)
        replaced = ctx._device_facts.get(subject)
        if replaced is not None:
            ctx.facts[replaced] = ""
        ctx._device_facts[subject] = len(ctx.facts)
    ctx.facts.append(outcome.fact)
    label = f"{FACT_PREFIX}{len(ctx.facts)}"
    return replace(outcome, output={**outcome.output, "label": label, "fact": outcome.fact})


def _bookings_fact(rows: Sequence[Any]) -> str:
    """Upcoming bookings in UTC, one clause each, in start order."""
    clauses = []
    for row in rows:
        starts, ends = row.starts_at.astimezone(UTC), row.ends_at.astimezone(UTC)
        end = f"{ends:%H:%M}" if ends.date() == starts.date() else f"{ends:%Y-%m-%d %H:%M}"
        clauses.append(f"{row.amenity} on {starts:%Y-%m-%d} from {starts:%H:%M} to {end} UTC")
    if not clauses:
        return "No upcoming bookings."
    return "Upcoming bookings: " + "; ".join(clauses) + "."


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
        graph=graph_search(settings),
        max_passages=remaining,
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
        rows = rows.all()
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
    return _outcome(
        tool,
        Decision.ALLOWED,
        ResultCode.OK,
        output={"bookings": bookings},
        fact=_bookings_fact(rows),
    )


async def create_ticket(ctx: ToolContext, args: TicketArguments) -> ToolOutcome:
    tool, resident, settings = ToolName.CREATE_TICKET, ctx.resident, ctx.settings
    if _repeats_a_note(ctx, args.summary):
        await _audited(ctx, tool, Decision.DENIED, ResultCode.NOT_PERMITTED)
        return _outcome(tool, Decision.DENIED, ResultCode.NOT_PERMITTED)
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
            minutes = settings.assistant_ticket_dedupe_seconds / 60
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
                fact=f"An open {args.category} ticket from the last {minutes:g} minutes "
                "already exists.",
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
        fact=f"Maintenance ticket opened in category {args.category}.",
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
    ctx: ToolContext,
    tool: ToolName,
    name: str,
    kind: str,
    values: dict,
    details: dict,
    fact: str,
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
    return _outcome(
        tool, Decision.ALLOWED, ResultCode.OK, device=name, output={"device": name}, fact=fact
    )


async def set_thermostat(ctx: ToolContext, args: ThermostatArguments) -> ToolOutcome:
    celsius = float(args.celsius)
    return await _set_device(
        ctx,
        ToolName.SET_THERMOSTAT,
        args.device,
        "thermostat",
        {"target_celsius": celsius},
        {"celsius": celsius},
        f"{args.device} target set to {celsius:g} °C.",
    )


async def lock_door(ctx: ToolContext, args: DeviceArguments) -> ToolOutcome:
    return await _set_device(
        ctx,
        ToolName.LOCK_DOOR,
        args.device,
        "door_lock",
        {"locked": True},
        {},
        f"{args.device} is locked.",
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
    fact = None
    if action_id is not None:
        output["status"] = "confirmation_required"
        fact = f"Unlock request for {name} is waiting for your confirmation in the app."
    return _outcome(
        tool,
        decision,
        result,
        device=name,
        pending_action_id=action_id,
        output=output,
        fact=fact,
    )


SAVED_FACTS = {
    MemoryScope.RESIDENT: "Saved your note to your memories.",
    MemoryScope.RESIDENCE: "Saved your note for this residence.",
}


def _asks_to_forget(quote: str, message: str) -> bool:
    """The quoted words name a forget request, and the message never says "don't forget"."""
    return bool(_FORGET_INTENT.search(normalize(quote))) and not _NEGATED_FORGET.search(
        normalize(message)
    )


def _windows(text: str) -> set[tuple[str, ...]]:
    """Every run of three words in `text`, normalized and stripped of punctuation."""
    words = _WORD.findall(normalize(text))
    return {
        tuple(words[start : start + MIN_QUOTE_WORDS])
        for start in range(len(words) - MIN_QUOTE_WORDS + 1)
    }


def _repeats_a_note(ctx: ToolContext, text: str) -> bool:
    """`text` holds three words in a row of a recalled note that the message does not.

    Recalled notes are private: a ticket summary, which co-residents can read, may hold
    what the resident says now but not what the model read from a note. Both sides are
    compared as bare words, so punctuation cannot hide a note. A paraphrase still can:
    this bounds copying, it is not a guarantee.
    """
    if not ctx.memories:
        return False
    copied = _windows(text) - _windows(ctx.message)
    return any(_windows(memory.text) & copied for memory in ctx.memories)


async def _memory_gate(ctx: ToolContext, tool: str, quote: str) -> ToolOutcome | None:
    """Refuse a memory tool whose quoted words the resident never said."""
    result = None
    if not whole_quote(quote, ctx.message):
        result = ResultCode.INVALID_ARGUMENTS
    elif tool == ToolName.FORGET and not _asks_to_forget(quote, ctx.message):
        result = ResultCode.INVALID_ARGUMENTS
    if result is None:
        return None
    await _audited(ctx, tool, Decision.DENIED, result)
    return _outcome(tool, Decision.DENIED, result)


async def remember(ctx: ToolContext, args: RememberArguments) -> ToolOutcome:
    """Store a span of the resident's message as a memory, under the memory limits."""
    tool, scope = ToolName.REMEMBER, args.scope
    refused = await _memory_gate(ctx, tool, args.text)
    if refused is not None:
        return refused
    async with ctx.session_factory() as session:
        # User row first, then membership: the same order as the memory route.
        now = await lock_writer(session, ctx.resident.user_id)
        if not await _hold_membership(session, ctx):
            return await _deny(session, ctx, tool, ResultCode.NOT_PERMITTED)
        result = await write_memory(
            session, ctx.memory_box, ctx.settings, ctx.resident.user_id, scope, args.text, now
        )
        if isinstance(result, Refused):
            code = (
                ResultCode.NOT_PERMITTED
                if result.reason is Refusal.NO_RESIDENCE
                else ResultCode.LIMIT_EXCEEDED
            )
            return await _deny(session, ctx, tool, code)
        details = {"scope": str(scope)}
        await session.execute(_audit(ctx, tool, Decision.ALLOWED, ResultCode.OK, details=details))
        await session.commit()
    return _outcome(
        tool,
        Decision.ALLOWED,
        ResultCode.OK,
        output={"scope": str(scope)},
        fact=SAVED_FACTS[scope],
    )


async def forget(ctx: ToolContext, args: ForgetArguments) -> ToolOutcome:
    """Hard delete one memory recalled this turn, named by its M label."""
    tool = ToolName.FORGET
    refused = await _memory_gate(ctx, tool, args.request_quote)
    if refused is not None:
        return refused
    index = int(args.label[1:]) - 1
    recalled = ctx.memories[index] if index < len(ctx.memories) else None
    if recalled is None or not recalled.text:
        await _audited(ctx, tool, Decision.DENIED, ResultCode.NOT_FOUND)
        return _outcome(tool, Decision.DENIED, ResultCode.NOT_FOUND)
    # Gone or going: either way it grounds no later claim this turn.
    ctx.memories[index] = Recalled(memory_id=recalled.memory_id, text="")
    async with ctx.session_factory() as session:
        # Already gone: the resident deleted it through the route during this turn.
        if not await delete_memory(session, ctx.resident.user_id, recalled.memory_id):
            return await _deny(session, ctx, tool, ResultCode.NOT_FOUND)
        await session.execute(_audit(ctx, tool, Decision.ALLOWED, ResultCode.OK))
        await session.commit()
    # The fact names the note in the resident's own words, so they see what went.
    return _outcome(
        tool,
        Decision.ALLOWED,
        ResultCode.OK,
        output={"label": args.label},
        fact=f'Forgot your note: "{recalled.text}".',
    )


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
    ToolName.REMEMBER: _Spec(
        RememberArguments,
        remember,
        "Save a note the resident asks you to remember. text is copied word for word from "
        "the resident's message. scope residence keeps it only while they live here.",
        {"text": _STRING, "scope": {"type": "string", "enum": [str(s) for s in MemoryScope]}},
    ),
    ToolName.FORGET: _Spec(
        ForgetArguments,
        forget,
        "Delete one recalled memory, by its label M1 to M20, when the resident asks. "
        "request_quote is copied word for word from the resident's message.",
        {"label": _STRING, "request_quote": _STRING},
    ),
}


def _offered(settings: AssistantSettings) -> dict[str, _Spec]:
    """The tools of this deployment: the memory tools only while memory is on."""
    if settings.assistant_memory_enabled:
        return _SPECS
    return {name: spec for name, spec in _SPECS.items() if name not in MEMORY_TOOLS}


def tool_schemas(settings: AssistantSettings) -> list[dict]:
    return [
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
        for name, spec in _offered(settings).items()
    ]


def _spec(ctx: ToolContext, name: str) -> _Spec | None:
    spec = _offered(ctx.settings).get(name)
    # A memory tool also needs the box: a context built without one runs none.
    if name in MEMORY_TOOLS and ctx.memory_box is None:
        return None
    return spec


async def execute(ctx: ToolContext, call: ToolCall) -> ToolOutcome:
    """Run one model tool call under the server's rules. Always audited."""
    telemetry = ctx.telemetry
    if telemetry is None:
        return await _execute(ctx, call)
    # The span names only allowlisted tools; anything else is "unknown".
    tool = call.name if _spec(ctx, call.name) is not None else UNKNOWN_TOOL
    with telemetry.span("assistant.tool", operation=tool, turn_id=str(ctx.turn_id)):
        outcome = await _execute(ctx, call)
        decision, result = str(outcome.action.decision), str(outcome.action.result)
        telemetry.annotate(status=decision, outcome=result)
        telemetry.increment("assistant.tool_calls", operation=tool, status=decision, outcome=result)
    return outcome


async def _refuse_after_search(ctx: ToolContext, tool: str) -> ToolOutcome:
    ctx.refused_after_search = True
    await _audited(ctx, tool, Decision.DENIED, ResultCode.NOT_PERMITTED)
    return _outcome(
        tool,
        Decision.DENIED,
        ResultCode.NOT_PERMITTED,
        output={"reason": REFUSED_AFTER_SEARCH},
    )


async def _execute(ctx: ToolContext, call: ToolCall) -> ToolOutcome:
    spec = _spec(ctx, call.name)
    tool = ToolName(call.name) if spec is not None else UNKNOWN_TOOL
    # Once a search handed out passages, a document could be steering the model: only
    # reads run for the rest of the turn. Decided before the arguments are read, so a
    # bad argument or an unknown name is refused, and reported, like any other change.
    if ctx.passages and tool not in READ_TOOLS:
        return await _refuse_after_search(ctx, tool)
    if spec is None:
        await _audited(ctx, UNKNOWN_TOOL, Decision.DENIED, ResultCode.UNKNOWN_TOOL)
        return _outcome(UNKNOWN_TOOL, Decision.DENIED, ResultCode.UNKNOWN_TOOL)
    try:
        args = spec.arguments.model_validate_json(call.arguments)
    except ValidationError:
        await _audited(ctx, tool, Decision.DENIED, ResultCode.INVALID_ARGUMENTS)
        return _outcome(tool, Decision.DENIED, ResultCode.INVALID_ARGUMENTS)
    return _labelled(ctx, await spec.handler(ctx, args))


def output_text(outcome: ToolOutcome) -> str:
    return json.dumps(outcome.output, ensure_ascii=False)
