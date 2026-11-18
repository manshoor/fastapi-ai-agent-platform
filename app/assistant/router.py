from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.actions import confirm_action
from app.assistant.config import AssistantSettings
from app.assistant.contracts import ConfirmResponse, MessageRequest, ResidenceView, TurnResponse
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
    "/assistant/actions/{action_id}/confirm",
    response_model=ConfirmResponse,
    responses={403: {}, 404: {}, 409: {}, 503: {}},
)
async def confirm(
    request: Request, response: Response, action_id: UUID, owner_id: Owner, session: Session
) -> ConfirmResponse:
    _settings(request)
    response.headers["Cache-Control"] = "no-store"
    telemetry = request.app.state.telemetry
    with telemetry.span("assistant.confirm", action_id=str(action_id)):
        resident = await _resident(session, owner_id)
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
