"""PDF text extraction child process. Run only as `python -I -m app.rag.pdf_extract`.

Reads PDF bytes on stdin and writes one JSON object on stdout:
    {"pages": ["page 1 text", ...]}  or  {"error": "<fixed code>"}

Imports only the standard library and pypdf. Resource limits are set before any
untrusted byte is parsed, so a hostile file can exhaust only this process.
"""

import io
import json
import resource
import sys

# Fixed codes shared with app.rag.contracts.IngestError; duplicated so this child
# never imports application modules.
ENCRYPTED = "pdf_encrypted"
PARSE_FAILED = "pdf_parse_failed"
PAGE_LIMIT = "pdf_page_limit"
TEXT_LIMIT = "text_limit"
# Kept in step with app.rag.parsing.SANDBOX_UNAVAILABLE_EXIT.
SANDBOX_UNAVAILABLE_EXIT = 3


def _limit(kind: int, value: int) -> None:
    resource.setrlimit(kind, (value, value))


def _clean(text: str) -> str:
    # PostgreSQL text cannot hold NUL, and lone surrogates cannot be encoded.
    return text.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")


def extract(data: bytes, max_pages: int, max_characters: int) -> dict:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return {"error": ENCRYPTED}
        if len(reader.pages) > max_pages:
            return {"error": PAGE_LIMIT}
        pages = []
        total = 0
        for page in reader.pages:
            text = _clean(page.extract_text() or "")
            total += len(text)
            if total > max_characters:
                return {"error": TEXT_LIMIT}
            pages.append(text)
        return {"pages": pages}
    except Exception:
        # Never report parser messages: they can quote document content.
        return {"error": PARSE_FAILED}


def main(argv: list[str]) -> int:
    max_pages, max_characters, memory_bytes, cpu_seconds, max_input = map(int, argv)
    try:
        _limit(resource.RLIMIT_AS, memory_bytes)
        _limit(resource.RLIMIT_CPU, cpu_seconds)
        _limit(resource.RLIMIT_FSIZE, 0)
        _limit(resource.RLIMIT_CORE, 0)
    except (ValueError, OSError):
        # Fail closed, and say so: this is the worker's fault, not the document's.
        return SANDBOX_UNAVAILABLE_EXIT
    data = sys.stdin.buffer.read(max_input + 1)
    if len(data) > max_input:
        result = {"error": PARSE_FAILED}
    else:
        try:
            result = extract(data, max_pages, max_characters)
        except MemoryError:
            result = {"error": PARSE_FAILED}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
