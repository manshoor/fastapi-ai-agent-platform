"""Turn uploaded bytes into page texts. PDFs are parsed in a limited child process.

worker --stdin bytes--> python -I -m app.rag.pdf_extract --stdout JSON--> worker
         scrubbed env, own session, wall-clock timeout, killed on timeout/cancel
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import signal
import sys
from dataclasses import dataclass

from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError

_UTF8_BOM = b"\xef\xbb\xbf"
# Only what the interpreter needs; no database, provider, or telemetry settings.
_CHILD_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
_CHILD_CODES = {
    code.value: code
    for code in (
        IngestError.PDF_ENCRYPTED,
        IngestError.PDF_PARSE_FAILED,
        IngestError.PDF_PAGE_LIMIT,
        IngestError.TEXT_LIMIT,
    )
}
# ensure_ascii JSON spends up to 12 bytes on one character (a non-BMP surrogate pair)
# and up to 4 bytes of framing per page.
_OUTPUT_FACTOR = 12
_PAGE_FRAMING = 4
_OUTPUT_SLACK = 1024
_READ_BYTES = 65536
# The child exits with this code when it cannot set its limits. Kept in step with
# app.rag.pdf_extract, which never imports application modules.
SANDBOX_UNAVAILABLE_EXIT = 3
logger = logging.getLogger(__name__)


class SandboxUnavailable(RuntimeError):
    """The worker cannot parse PDFs safely. Not a property of the document."""


class ParseFailure(Exception):
    """A permanent, content-free reason the document cannot be ingested."""

    def __init__(self, code: IngestError):
        super().__init__(str(code))
        self.code = code


@dataclass(frozen=True)
class ParsedDocument:
    # (page number or None for plain text, text) in reading order.
    pages: list[tuple[int | None, str]]
    page_count: int | None


def _kill(process: asyncio.subprocess.Process) -> None:
    # The child leads its own process group; kill it and anything it spawned.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _run_child(data: bytes, settings: RAGSettings) -> dict:
    cpu_seconds = math.ceil(settings.rag_pdf_timeout_seconds) + 1
    arguments = [
        str(settings.rag_pdf_max_pages),
        str(settings.rag_max_characters),
        str(settings.rag_pdf_memory_bytes),
        str(cpu_seconds),
        str(settings.rag_pdf_max_bytes),
    ]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-m",
        "app.rag.pdf_extract",
        *arguments,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_CHILD_ENV,
        start_new_session=True,
    )
    limit = (
        settings.rag_max_characters * _OUTPUT_FACTOR
        + settings.rag_pdf_max_pages * _PAGE_FRAMING
        + _OUTPUT_SLACK
    )
    try:
        async with asyncio.timeout(settings.rag_pdf_timeout_seconds):
            _, stdout = await asyncio.gather(
                _write_input(process, data), _read_bounded(process, limit)
            )
            await process.wait()
    except TimeoutError:
        raise ParseFailure(IngestError.PDF_PARSE_TIMEOUT) from None
    finally:
        if process.returncode is None:
            _kill(process)
            await process.wait()
    if process.returncode == SANDBOX_UNAVAILABLE_EXIT:
        logger.error("RAG PDF sandbox limits could not be set")
        raise SandboxUnavailable("PDF sandbox limits could not be set")
    if process.returncode != 0 or stdout is None:
        raise ParseFailure(IngestError.PDF_PARSE_FAILED)
    try:
        result = json.loads(stdout)
    except ValueError:
        raise ParseFailure(IngestError.PDF_PARSE_FAILED) from None
    return result if isinstance(result, dict) else {}


async def _write_input(process: asyncio.subprocess.Process, data: bytes) -> None:
    # A child that exits early closes its stdin; its exit code decides the outcome.
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        process.stdin.write(data)
        await process.stdin.drain()
        process.stdin.close()


async def _read_bounded(process: asyncio.subprocess.Process, limit: int) -> bytes | None:
    """Read stdout until EOF; past `limit` bytes, kill the child and return None."""
    output = bytearray()
    while chunk := await process.stdout.read(_READ_BYTES):
        output.extend(chunk)
        if len(output) > limit:
            _kill(process)
            return None
    return bytes(output)


def _valid_text(text: object) -> bool:
    # The child already strips NUL and lone surrogates; never trust that it did.
    if not isinstance(text, str) or "\x00" in text:
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


async def _parse_pdf(data: bytes, settings: RAGSettings) -> ParsedDocument:
    result = await _run_child(data, settings)
    error = result.get("error")
    if error is not None:
        # The child may only report its own codes; anything else is a parse failure.
        code = _CHILD_CODES.get(error) if isinstance(error, str) else None
        raise ParseFailure(code or IngestError.PDF_PARSE_FAILED)
    pages = result.get("pages")
    if (
        not isinstance(pages, list)
        or len(pages) > settings.rag_pdf_max_pages
        or not all(_valid_text(text) for text in pages)
        or sum(len(text) for text in pages) > settings.rag_max_characters
    ):
        raise ParseFailure(IngestError.PDF_PARSE_FAILED)
    return ParsedDocument(
        pages=[(number, text) for number, text in enumerate(pages, start=1)],
        page_count=len(pages),
    )


def _parse_text(data: bytes, settings: RAGSettings) -> ParsedDocument:
    # Upload already proved UTF-8 without NUL; decode the same way.
    text = data.removeprefix(_UTF8_BOM).decode("utf-8")
    if len(text) > settings.rag_max_characters:
        raise ParseFailure(IngestError.TEXT_LIMIT)
    return ParsedDocument(pages=[(None, text)], page_count=None)


async def parse_document(
    data: bytes, content_type: ContentType, settings: RAGSettings
) -> ParsedDocument:
    if content_type is ContentType.PDF:
        parsed = await _parse_pdf(data, settings)
    else:
        parsed = _parse_text(data, settings)
    if not any(text.strip() for _, text in parsed.pages):
        raise ParseFailure(IngestError.NO_TEXT)
    return parsed
