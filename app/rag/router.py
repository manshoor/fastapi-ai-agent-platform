from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.rag.contracts import DocumentList, DocumentResponse
from app.rag.documents import delete_document, get_document, list_documents, submit_document
from app.rag.runtime import RAGRuntime
from app.rag.uploads import read_upload
from app.security import get_current_user_id

UPLOAD_PATH = "/api/v1/rag/documents"
router = APIRouter(prefix="/api/v1/rag", tags=["rag"])
Owner = Annotated[UUID, Depends(get_current_user_id)]
Session = Annotated[AsyncSession, Depends(get_session)]
Key = Annotated[
    str,
    Header(alias="Idempotency-Key", min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._-]+$"),
]


def _runtime(request: Request) -> RAGRuntime:
    runtime = request.app.state.rag
    if runtime is None:
        raise HTTPException(503, "RAG is disabled", headers={"Retry-After": "60"})
    return runtime


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=202,
    responses={
        200: {"model": DocumentResponse},
        400: {},
        408: {},
        409: {},
        413: {},
        415: {},
        422: {},
        429: {},
        503: {},
    },
)
async def upload_document(
    request: Request,
    response: Response,
    owner_id: Owner,
    session: Session,
    idempotency_key: Key,
) -> DocumentResponse:
    # No Form/File parameters: FastAPI would read the body before authentication.
    runtime = _runtime(request)
    telemetry = runtime.telemetry
    telemetry.increment("rag.requests", operation="upload")
    async with runtime.upload_slot(owner_id):
        upload = await read_upload(request, runtime.settings)
        with telemetry.span("rag.upload"):
            result, status_code = await submit_document(
                session=session,
                settings=runtime.settings,
                owner_id=owner_id,
                upload=upload,
                idempotency_key=idempotency_key,
                trace_context=telemetry.capture_context(),
                telemetry=telemetry,
            )
    response.status_code = status_code
    response.headers["Location"] = result.status_url
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("/documents", response_model=DocumentList, responses={503: {}})
async def read_documents(
    request: Request, response: Response, owner_id: Owner, session: Session
) -> DocumentList:
    runtime = _runtime(request)
    items = await list_documents(session=session, settings=runtime.settings, owner_id=owner_id)
    response.headers["Cache-Control"] = "no-store"
    return DocumentList(items=items)


@router.get(
    "/documents/{document_id}", response_model=DocumentResponse, responses={404: {}, 503: {}}
)
async def read_document(
    document_id: UUID, request: Request, response: Response, owner_id: Owner, session: Session
) -> DocumentResponse:
    _runtime(request)
    result = await get_document(session=session, owner_id=owner_id, document_id=document_id)
    response.headers["Cache-Control"] = "no-store"
    return result


@router.delete("/documents/{document_id}", status_code=204, responses={404: {}, 503: {}})
async def remove_document(
    document_id: UUID, request: Request, owner_id: Owner, session: Session
) -> Response:
    _runtime(request)
    await delete_document(session=session, owner_id=owner_id, document_id=document_id)
    return Response(status_code=204, headers={"Cache-Control": "no-store"})
