"""Bounded multipart reading. Called only after authentication has succeeded."""

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from app.rag.config import RAGSettings
from app.rag.contracts import SUFFIXES, ContentType, UploadTitle

_FIELDS = frozenset({"title", "file"})
_PDF_MAGIC = b"%PDF-"
_UTF8_BOM = b"\xef\xbb\xbf"
# The title is the only text field; its limit is far below this.
_FIELD_BYTES = 4096


@dataclass(frozen=True)
class Upload:
    title: str
    content_type: ContentType
    data: bytes


def _invalid(field: str) -> RequestValidationError:
    # Fixed location and type only: never echo the submitted value.
    return RequestValidationError([{"type": "value_error", "loc": ("body", field)}])


def _check_text(data: bytes) -> None:
    try:
        text = data.removeprefix(_UTF8_BOM).decode("utf-8")
    except UnicodeDecodeError:
        raise _invalid("file") from None
    if "\x00" in text:
        raise _invalid("file")


async def _read_file(file: UploadFile, settings: RAGSettings) -> tuple[ContentType, bytes]:
    suffix = PurePosixPath(file.filename or "").suffix.lower()
    content_type = SUFFIXES.get(suffix)
    if content_type is None:
        raise HTTPException(415, "Upload a .txt, .md or .pdf file")
    limit = (
        settings.rag_pdf_max_bytes
        if content_type is ContentType.PDF
        else settings.rag_text_max_bytes
    )
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, "File too large")
    if not data:
        raise _invalid("file")
    if content_type is ContentType.PDF:
        if not data.startswith(_PDF_MAGIC):
            raise _invalid("file")
    else:
        _check_text(data)
    return content_type, data


async def read_upload(request: Request, settings: RAGSettings) -> Upload:
    """Parse `title` + one `file`; reject anything else with a fixed, content-free error."""
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise HTTPException(415, "Use multipart/form-data")
    try:
        async with asyncio.timeout(settings.rag_upload_timeout_seconds):
            async with request.form(max_files=1, max_fields=1, max_part_size=_FIELD_BYTES) as form:
                if set(form.keys()) != _FIELDS or len(form.multi_items()) != len(_FIELDS):
                    raise HTTPException(400, "Send exactly one title and one file")
                title, file = form.get("title"), form.get("file")
                if not isinstance(title, str) or not isinstance(file, UploadFile):
                    raise HTTPException(400, "Send exactly one title and one file")
                try:
                    valid = UploadTitle.model_validate({"title": title})
                except ValidationError as exc:
                    raise RequestValidationError(
                        [
                            {"type": error["type"], "loc": ("body", *error["loc"])}
                            for error in exc.errors()
                        ]
                    ) from None
                content_type, data = await _read_file(file, settings)
    except TimeoutError:
        raise HTTPException(408, "Upload body too slow") from None
    except ClientDisconnect:
        # The client left: answer with a fixed error instead of an unhandled 500.
        raise HTTPException(400, "Upload body incomplete") from None
    except UnicodeError:
        # A client-chosen multipart charset that Python cannot decode (e.g. "undefined").
        raise HTTPException(400, "Malformed multipart body") from None
    except StarletteHTTPException as exc:
        if isinstance(exc, HTTPException):
            raise
        # Starlette's multipart errors: replace their message with a fixed one.
        raise HTTPException(400, "Malformed multipart body") from None
    return Upload(title=valid.title, content_type=content_type, data=data)
