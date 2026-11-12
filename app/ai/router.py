from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.config import AISettings
from app.ai.contracts import JobResponse, RewriteRequest
from app.ai.jobs import poll_job, submit_job
from app.db import get_session
from app.security import get_current_user_id

router = APIRouter(prefix="/api/v1", tags=["ai"])
Owner = Annotated[UUID, Depends(get_current_user_id)]
Session = Annotated[AsyncSession, Depends(get_session)]
Key = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
]


def _settings(request: Request) -> AISettings:
    settings = request.app.state.ai_settings
    if settings is None:
        raise HTTPException(503, "Rewriting is disabled", headers={"Retry-After": "60"})
    return settings


@router.post(
    "/items/{item_id}/ai/rewrite",
    response_model=JobResponse,
    status_code=202,
    responses={200: {"model": JobResponse}, 404: {}, 409: {}, 410: {}, 429: {}, 503: {}},
)
async def rewrite_item(
    item_id: UUID,
    payload: RewriteRequest,
    request: Request,
    response: Response,
    owner_id: Owner,
    session: Session,
    idempotency_key: Key,
) -> JobResponse:
    settings = _settings(request)
    telemetry = request.app.state.telemetry
    telemetry.increment("ai.requests", operation="submit")
    with telemetry.span("ai.submit"):
        result, status_code = await submit_job(
            session=session,
            settings=settings,
            owner_id=owner_id,
            item_id=item_id,
            payload=payload,
            idempotency_key=idempotency_key,
            trace_context=telemetry.capture_context(),
            telemetry=telemetry,
        )
    response.status_code = status_code
    response.headers["Location"] = result.status_url
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("/ai/jobs/{job_id}", response_model=JobResponse, responses={404: {}, 410: {}, 503: {}})
async def read_job(
    job_id: UUID, request: Request, response: Response, owner_id: Owner, session: Session
) -> JobResponse:
    _settings(request)
    request.app.state.telemetry.increment("ai.requests", operation="poll")
    with request.app.state.telemetry.span("ai.poll"):
        result = await poll_job(session=session, owner_id=owner_id, job_id=job_id)
    response.headers["Cache-Control"] = "no-store"
    return result
