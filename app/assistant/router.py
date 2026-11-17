from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.config import AssistantSettings
from app.assistant.contracts import ResidenceView
from app.assistant.residence import Resident, create_demo_residence, load_resident, residence_view
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


@router.post(
    "/residence/demo",
    response_model=ResidenceView,
    status_code=201,
    responses={404: {}, 409: {}, 503: {}},
)
async def create_demo(
    request: Request, response: Response, owner_id: Owner, session: Session
) -> ResidenceView:
    # Demo-only route: absent unless explicitly enabled.
    runtime_settings: Settings = request.app.state.settings
    if not runtime_settings.assistant_demo_enabled:
        raise HTTPException(404, "Not Found")
    _settings(request)
    resident = await create_demo_residence(session, owner_id)
    view = await residence_view(session, resident)
    await session.rollback()
    response.headers["Cache-Control"] = "no-store"
    return view
