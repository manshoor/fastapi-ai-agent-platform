from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant import memory, step_up, totp
from app.assistant.actions import confirm_action
from app.assistant.config import AssistantSettings
from app.assistant.contracts import (
    CodeRequest,
    ConfirmResponse,
    MemoryList,
    MemoryRequest,
    MemoryView,
    MessageRequest,
    PasswordRequest,
    ResidenceView,
    TotpEnrollment,
    TotpStatus,
    TurnResponse,
)
from app.assistant.residence import Resident, create_demo_residence, load_resident, residence_view
from app.assistant.runtime import AssistantRuntime
from app.assistant.turns import run_turn
from app.config import Settings
from app.db import get_session
from app.security import get_current_user_id

router = APIRouter(prefix="/api/v1", tags=["assistant"])
Owner = Annotated[UUID, Depends(get_current_user_id)]
Session = Annotated[AsyncSession, Depends(get_session)]


def _settings(request: Request) -> AssistantSettings:
    settings = request.app.state.assistant_settings
    if settings is None:
        raise HTTPException(503, "Assistant is disabled", headers={"Retry-After": "60"})
    return settings


async def _resident(session: AsyncSession, owner_id: UUID) -> Resident:
    resident = await load_resident(session, owner_id)
    if resident is None:
        await session.rollback()
        raise HTTPException(403, "No residence for this account")
    return resident


@router.get("/residence", response_model=ResidenceView, responses={403: {}, 503: {}})
async def read_residence(
    request: Request, response: Response, owner_id: Owner, session: Session
) -> ResidenceView:
    _settings(request)
    resident = await _resident(session, owner_id)
    view = await residence_view(session, resident)
    await session.rollback()
    response.headers["Cache-Control"] = "no-store"
    return view


def _demo_enabled(request: Request) -> None:
    # Runs before authentication: with the flag off the route is absent, not refused.
    runtime_settings: Settings = request.app.state.settings
    if not runtime_settings.assistant_demo_enabled:
        raise HTTPException(404, "Not Found")


@router.post(
    "/residence/demo",
    response_model=ResidenceView,
    status_code=201,
    responses={404: {}, 409: {}, 503: {}},
    dependencies=[Depends(_demo_enabled)],
    include_in_schema=False,
)
async def create_demo(
    request: Request, response: Response, owner_id: Owner, session: Session
) -> ResidenceView:
    _settings(request)
    resident = await create_demo_residence(session, owner_id)
    view = await residence_view(session, resident)
    await session.rollback()
    response.headers["Cache-Control"] = "no-store"
    return view


@router.post(
    "/assistant/totp/enroll",
    response_model=TotpEnrollment,
    status_code=201,
    responses={403: {}, 409: {}, 429: {}, 503: {}},
)
async def enroll_totp(
    payload: PasswordRequest,
    request: Request,
    response: Response,
    owner_id: Owner,
    session: Session,
) -> TotpEnrollment:
    settings = _settings(request)
    response.headers["Cache-Control"] = "no-store"
    runtime: AssistantRuntime = request.app.state.assistant
    email = await step_up.check_password(
        session, request.app.state.passwords, request.app.state.settings, owner_id, payload.password
    )
    secret = await step_up.enroll(session, runtime.secret_box, owner_id)
    return TotpEnrollment(
        secret=totp.encode_secret(secret),
        otpauth_uri=totp.provisioning_uri(secret, email, settings.assistant_totp_issuer),
    )


@router.post(
    "/assistant/totp/activate",
    response_model=TotpStatus,
    responses={403: {}, 409: {}, 429: {}, 503: {}},
)
async def activate_totp(
    payload: CodeRequest, request: Request, response: Response, owner_id: Owner, session: Session
) -> TotpStatus:
    settings = _settings(request)
    response.headers["Cache-Control"] = "no-store"
    runtime: AssistantRuntime = request.app.state.assistant
    await step_up.activate(session, runtime.secret_box, settings, owner_id, payload.code)
    return TotpStatus(enabled=True)


@router.post(
    "/assistant/actions/{action_id}/confirm",
    response_model=ConfirmResponse,
    responses={403: {}, 404: {}, 409: {}, 429: {}, 503: {}},
)
async def confirm(
    payload: CodeRequest,
    request: Request,
    response: Response,
    action_id: UUID,
    owner_id: Owner,
    session: Session,
) -> ConfirmResponse:
    settings = _settings(request)
    response.headers["Cache-Control"] = "no-store"
    runtime: AssistantRuntime = request.app.state.assistant
    telemetry = request.app.state.telemetry
    with telemetry.span("assistant.confirm", action_id=str(action_id)):
        resident = await _resident(session, owner_id)
        # The code is spent and committed first: a 404 or 409 below cannot reuse it.
        await step_up.verify(session, runtime.secret_box, settings, resident, payload.code)
        return await confirm_action(session, resident, action_id)


@router.post(
    "/assistant/messages",
    response_model=TurnResponse,
    responses={403: {}, 422: {}, 429: {}, 502: {}, 503: {}, 504: {}},
)
async def send_message(
    payload: MessageRequest, request: Request, response: Response, owner_id: Owner
) -> TurnResponse:
    # No Session dependency: no request-scoped session stays open across provider calls.
    _settings(request)
    runtime: AssistantRuntime = request.app.state.assistant
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        resident = await _resident(session, owner_id)
        await session.rollback()
    result = await run_turn(
        session_factory=session_factory,
        runtime=runtime,
        resident=resident,
        message=payload.message,
    )
    response.headers["Cache-Control"] = "no-store"
    return result


def _memory_enabled(request: Request) -> None:
    # Runs before authentication: with memory off the routes are absent, not refused.
    settings = request.app.state.assistant_settings
    if settings is None or not settings.assistant_memory_enabled:
        raise HTTPException(404, "Not Found")


MEMORIES = "/assistant/memories"


@router.get(
    MEMORIES,
    response_model=MemoryList,
    responses={404: {}, 500: {}},
    dependencies=[Depends(_memory_enabled)],
    include_in_schema=False,
)
async def read_memories(
    request: Request, response: Response, owner_id: Owner, session: Session
) -> MemoryList:
    response.headers["Cache-Control"] = "no-store"
    runtime: AssistantRuntime = request.app.state.assistant
    return await memory.list_memories(session, runtime.memory_box, owner_id)


@router.post(
    MEMORIES,
    response_model=MemoryView,
    status_code=201,
    responses={403: {}, 404: {}, 422: {}, 429: {}},
    dependencies=[Depends(_memory_enabled)],
    include_in_schema=False,
)
async def add_memory(
    payload: MemoryRequest,
    request: Request,
    response: Response,
    owner_id: Owner,
    session: Session,
) -> MemoryView:
    response.headers["Cache-Control"] = "no-store"
    runtime: AssistantRuntime = request.app.state.assistant
    return await memory.add_memory(
        session, runtime.memory_box, runtime.settings, owner_id, payload.scope, payload.text
    )


@router.delete(
    MEMORIES + "/{memory_id}",
    status_code=204,
    responses={404: {}},
    dependencies=[Depends(_memory_enabled)],
    include_in_schema=False,
)
async def forget_memory(memory_id: UUID, owner_id: Owner, session: Session) -> Response:
    await memory.forget_memory(session, owner_id, memory_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@router.delete(
    MEMORIES,
    status_code=204,
    dependencies=[Depends(_memory_enabled)],
    include_in_schema=False,
)
async def forget_memories(owner_id: Owner, session: Session) -> Response:
    await memory.forget_all(session, owner_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
