"""One assistant turn: admit, run the agent graph, ground the reply, record usage.

    turn slot (one per user, bounded per process)
      -> admit: per-user lock, window count, turn row
      -> resident context (devices) read, transaction released
      -> graph under deadline backstop, LangSmith tracing off
      -> grounding rule -> usage row -> TurnResponse

    Failure after a committed write: 200 "incomplete" with the actions.
    Failure with nothing committed: 503 / 504 / 502, as for RAG queries.

No transaction is open during a provider call. Rows hold counts and codes only:
never the message, a tool argument, a passage or the answer.
"""

import asyncio
import logging
import time
from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from langsmith import tracing_context
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import SQLAlchemyError

from app.ai.processing import SessionFactory
from app.ai.provider import ASSISTANT_PROMPT_VERSION, ProviderFailure
from app.assistant.contracts import (
    GROUNDING_RESULTS,
    ActionResult,
    Citation,
    PendingActionView,
    ToolName,
    TurnResponse,
)
from app.assistant.graph import BUDGET_CODE, TurnRun, first_input, recursion_limit
from app.assistant.residence import Resident
from app.assistant.runtime import AssistantRuntime
from app.assistant.tools import ToolContext
from app.models import AssistantTurn, Device, PendingAction
from app.telemetry import Telemetry

logger = logging.getLogger(__name__)
# Two-key advisory lock namespace for per-user turn admission ("ASTA").
_TURN_LOCK_NAMESPACE = 0x41535441
SNIPPET_CHARACTERS = 300
# The graph's own budget fires first; this bounds a stuck dependency, with room for one
# write (pool checkout plus a few statements at the two-second statement timeout).
BACKSTOP_SECONDS = 15
NO_CONTEXT_ANSWER = "I could not find that in your building's documents."
INCOMPLETE_ANSWER = "I could not finish. The completed steps are listed."
WRITE_TOOLS = frozenset(
    {ToolName.CREATE_TICKET, ToolName.SET_THERMOSTAT, ToolName.LOCK_DOOR, ToolName.UNLOCK_DOOR}
)
# Device kinds the model may see. The alarm is never exposed.
VISIBLE_KINDS = ("thermostat", "door_lock")
DEADLINE_CODE = "turn_deadline"
DATABASE_CODE = "database_unavailable"
CANCELLED_CODE = "turn_cancelled"
_UNAVAILABLE_CODES = frozenset({"provider_rate_limit"})
_TIMEOUT_CODES = frozenset({"provider_timeout", DEADLINE_CODE, BUDGET_CODE})


def _http_error(code: str) -> HTTPException:
    if code in _UNAVAILABLE_CODES:
        return HTTPException(503, "Assistant provider busy", headers={"Retry-After": "5"})
    if code in _TIMEOUT_CODES:
        return HTTPException(504, "Assistant timed out")
    return HTTPException(502, "Assistant provider failed")


async def _admit(session_factory: SessionFactory, runtime: AssistantRuntime, user_id: UUID):
    settings = runtime.settings
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:user))"),
                {"namespace": _TURN_LOCK_NAMESPACE, "user": str(user_id)},
            )
            recent = await session.scalar(
                select(func.count())
                .select_from(AssistantTurn)
                .where(
                    AssistantTurn.user_id == user_id,
                    AssistantTurn.created_at
                    > func.clock_timestamp()
                    - timedelta(seconds=settings.assistant_turn_window_seconds),
                )
            )
            if recent >= settings.assistant_turn_limit:
                retry_after = max(1, round(settings.assistant_turn_window_seconds))
                raise HTTPException(
                    429, "Turn limit reached", headers={"Retry-After": str(retry_after)}
                )
            turn_id = uuid4()
            session.add(
                AssistantTurn(
                    id=turn_id,
                    user_id=user_id,
                    model=settings.ai_model,
                    prompt_version=ASSISTANT_PROMPT_VERSION,
                    created_at=func.clock_timestamp(),
                )
            )
    return turn_id


async def _resident_context(session_factory: SessionFactory, resident: Resident) -> dict:
    async with session_factory() as session:
        rows = await session.execute(
            select(Device.name, Device.kind, Device.target_celsius, Device.locked)
            .where(Device.apartment_id == resident.apartment_id, Device.kind.in_(VISIBLE_KINDS))
            .order_by(Device.name)
        )
        devices = []
        for row in rows:
            device = {"name": row.name, "kind": row.kind}
            if row.kind == "thermostat":
                device["target_celsius"] = float(row.target_celsius)
            else:
                device["locked"] = row.locked
            devices.append(device)
        await session.rollback()
    return {
        "apartment": resident.apartment_label,
        "building": resident.building_name,
        "devices": devices,
    }


async def _record(
    telemetry: Telemetry,
    session_factory: SessionFactory,
    turn_id: UUID,
    status: str,
    run: TurnRun | None,
    error_code: str | None = None,
) -> None:
    """Best effort: a failed usage write never turns a committed result into an error."""
    counts = {"model_calls": run.model_calls, "tool_calls": len(run.actions)} if run else {}
    telemetry.annotate(status=status, error_code=error_code, **counts)
    telemetry.increment("assistant.turns", status=status, error_code=error_code)
    values = {"status": status, "error_code": error_code, "finished_at": func.clock_timestamp()}
    if run is not None:
        values.update(
            model_calls=run.model_calls,
            tool_calls=len(run.actions),
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            embedding_tokens=run.context.embedding_tokens,
        )
        if run.response_model is not None:
            values["model"] = run.response_model

    async def write() -> None:
        try:
            async with session_factory() as session:
                await session.execute(
                    update(AssistantTurn)
                    .where(AssistantTurn.id == turn_id, AssistantTurn.status == "admitted")
                    .values(**values)
                )
                await session.commit()
        except (SQLAlchemyError, OSError) as exc:
            logger.error("Assistant usage update failure exception=%s", type(exc).__name__)

    # A cancel cannot interrupt the close: the write finishes in the background.
    await asyncio.shield(write())


def _committed(actions: list[ActionResult]) -> bool:
    return any(
        action.tool in WRITE_TOOLS and (action.decision, action.result) in GROUNDING_RESULTS
        for action in actions
    )


def _grounded_by_tools(actions: list[ActionResult]) -> bool:
    # Search grounds an answer only through citations.
    return any(
        action.tool != ToolName.SEARCH_DOCUMENTS
        and (action.decision, action.result) in GROUNDING_RESULTS
        for action in actions
    )


def _citations(run: TurnRun) -> list[Citation]:
    passages = run.context.passages
    citations: list[Citation] = []
    seen: set[str] = set()
    for label in run.reply.citations:
        index = int(label[1:]) - 1
        if index >= len(passages):
            # A label the server never gave out is invalid output.
            raise ProviderFailure("provider_invalid_output")
        if label in seen:
            continue
        seen.add(label)
        document = passages[index]["document"]
        citations.append(
            Citation(
                document_id=document.metadata["document_id"],
                title=document.metadata["title"],
                ordinal=document.metadata["ordinal"],
                page=document.metadata["page"],
                snippet=document.page_content[:SNIPPET_CHARACTERS],
            )
        )
    return citations


async def _pending_views(session_factory: SessionFactory, run: TurnRun) -> list[PendingActionView]:
    ids = [action.pending_action_id for action in run.actions if action.pending_action_id]
    if not ids:
        return []
    async with session_factory() as session:
        rows = await session.execute(
            select(PendingAction.id, PendingAction.action, PendingAction.expires_at, Device.name)
            .join(Device, Device.id == PendingAction.device_id)
            .where(
                PendingAction.id.in_(ids),
                PendingAction.user_id == run.context.resident.user_id,
                PendingAction.status == "pending",
                PendingAction.expires_at > func.clock_timestamp(),
            )
            .order_by(PendingAction.created_at)
        )
        views = [
            PendingActionView(
                id=row.id, action=row.action, device=row.name, expires_at=row.expires_at
            )
            for row in rows
        ]
        await session.rollback()
    return views


class _Deadline(Exception):
    """The turn's backstop fired. Other timeouts are dependency failures."""


async def _run_graph(runtime: AssistantRuntime, run: TurnRun) -> None:
    settings = runtime.settings
    config = {
        "configurable": {"turn": run},
        "recursion_limit": recursion_limit(settings),
    }
    try:
        async with asyncio.timeout(
            settings.assistant_turn_deadline_seconds + BACKSTOP_SECONDS
        ) as scope:
            with tracing_context(enabled=False):
                await runtime.graph.ainvoke({"route": "model"}, config=config)
    except TimeoutError:
        if scope.expired():
            raise _Deadline from None
        raise


def _response(turn_id: UUID, run: TurnRun, status: str, answer: str, citations, pending):
    return TurnResponse(
        turn_id=turn_id,
        status=status,
        answer=answer,
        citations=citations,
        actions=run.actions,
        pending_actions=pending,
    )


async def _finish(
    session_factory: SessionFactory, runtime: AssistantRuntime, turn_id: UUID, run: TurnRun
) -> TurnResponse:
    """Run the graph and ground the reply. Raises only when nothing was committed."""
    code = None
    citations: list[Citation] = []
    try:
        await _run_graph(runtime, run)
        code = run.stop_code
        if code is None:
            citations = _citations(run)
    except ProviderFailure as exc:
        code = exc.code
    except _Deadline:
        code = DEADLINE_CODE
    if code is not None:
        if not _committed(run.actions):
            await _record(runtime.telemetry, session_factory, turn_id, "failed", run, code)
            raise _http_error(code)
        # Never hide a committed action behind an error.
        status, answer, citations = "incomplete", INCOMPLETE_ANSWER, []
    elif citations or _grounded_by_tools(run.actions):
        status, answer = "answered", run.reply.answer
    else:
        status, answer = "no_context", NO_CONTEXT_ANSWER
    pending = await _pending_views(session_factory, run)
    await _record(runtime.telemetry, session_factory, turn_id, status, run, code)
    return _response(turn_id, run, status, answer, citations, pending)


async def run_turn(
    *,
    session_factory: SessionFactory,
    runtime: AssistantRuntime,
    resident: Resident,
    message: str,
) -> TurnResponse:
    settings = runtime.settings
    async with runtime.turn_slot(resident.user_id):
        turn_id = await _admit(session_factory, runtime, resident.user_id)
        with runtime.telemetry.span(
            "assistant.turn",
            turn_id=str(turn_id),
            model=settings.ai_model,
            prompt_version=ASSISTANT_PROMPT_VERSION,
        ):
            started = time.monotonic()
            try:
                return await _turn(session_factory, runtime, resident, message, turn_id)
            finally:
                runtime.telemetry.observe("assistant.turn.duration", time.monotonic() - started)


async def _turn(
    session_factory: SessionFactory,
    runtime: AssistantRuntime,
    resident: Resident,
    message: str,
    turn_id: UUID,
) -> TurnResponse:
    settings = runtime.settings
    run: TurnRun | None = None
    try:
        context = await _resident_context(session_factory, resident)
        run = TurnRun(
            context=ToolContext(
                session_factory=session_factory,
                settings=settings,
                resident=resident,
                turn_id=turn_id,
                embedder=runtime.embedder,
                telemetry=runtime.telemetry,
            ),
            model=runtime.model,
            deadline=time.monotonic() + settings.assistant_turn_deadline_seconds,
            items=[first_input(context, message)],
        )
        return await _finish(session_factory, runtime, turn_id, run)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        await _record(runtime.telemetry, session_factory, turn_id, "failed", run, CANCELLED_CODE)
        raise
    except Exception as exc:
        # Database outages (including builtin connect timeouts) and programming errors.
        database = isinstance(exc, (SQLAlchemyError, OSError, TimeoutError))
        code = DATABASE_CODE if database else "internal_error"
        if run is not None and _committed(run.actions):
            # The write stands; report it. Pending ids are in the actions.
            await _record(runtime.telemetry, session_factory, turn_id, "incomplete", run, code)
            return _response(turn_id, run, "incomplete", INCOMPLETE_ANSWER, [], [])
        await _record(runtime.telemetry, session_factory, turn_id, "failed", run, code)
        raise
