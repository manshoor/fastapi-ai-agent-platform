"""Text extraction child process. Run only as `python -I -m app.rag.extract <format> ...`.

Reads document bytes on stdin and writes one JSON object on stdout:
    {"pages": ["page 1 text", ...]}  or  {"error": "<fixed code>"}

    pdf   pypdf, one entry per page
    docx  python-docx, one entry: paragraphs and table rows in order, headings as "#"
    html  lxml.html, one entry: script/style/noscript/template and comments dropped,
          headings as "#", table rows as "a | b"

Imports only the standard library and its parsers. Resource limits are set before any
untrusted byte is parsed, so a hostile file can exhaust only this process. Never
imports application modules: the upload route imports `docx_archive_ok` from here.
"""

import io
import json
import resource
import sys
import zipfile

# Fixed codes shared with app.rag.contracts.IngestError; duplicated so this child
# never imports application modules.
ENCRYPTED = "pdf_encrypted"
PDF_PARSE_FAILED = "pdf_parse_failed"
PAGE_LIMIT = "pdf_page_limit"
TEXT_LIMIT = "text_limit"
DOCX_PARSE_FAILED = "docx_parse_failed"
HTML_PARSE_FAILED = "html_parse_failed"
# Kept in step with app.rag.parsing.SANDBOX_UNAVAILABLE_EXIT.
SANDBOX_UNAVAILABLE_EXIT = 3

# Zip-bomb bounds, read from the central directory before anything is inflated.
DOCX_MAX_ENTRIES = 1000
DOCX_MAX_UNCOMPRESSED = 50 * 1024 * 1024
DOCX_BODY = "word/document.xml"

_HTML_DROPPED = ("script", "style", "noscript", "template")
_HTML_HEADINGS = {f"h{level}": level for level in range(1, 7)}
# Elements whose start and end break a line; everything else is inline text.
_HTML_BLOCKS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "caption", "dd", "details",
        "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "header",
        "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table",
        "tbody", "tfoot", "thead", "ul", "br",
    }
)  # fmt: skip


def _limit(kind: int, value: int) -> None:
    resource.setrlimit(kind, (value, value))


def _clean(text: str) -> str:
    # PostgreSQL text cannot hold NUL, and lone surrogates cannot be encoded.
    return text.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")


def _squash(text: str) -> str:
    return " ".join(text.split())


def docx_archive_ok(data: bytes) -> bool:
    """A Word body inside a bounded ZIP. Reads only the central directory."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, ValueError, OSError, EOFError):
        return False
    if len(entries) > DOCX_MAX_ENTRIES:
        return False
    if sum(entry.file_size for entry in entries) > DOCX_MAX_UNCOMPRESSED:
        return False
    return any(entry.filename == DOCX_BODY for entry in entries)


def _pdf(data: bytes, max_pages: int, max_characters: int) -> dict:
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
        return {"error": PDF_PARSE_FAILED}


def _docx_heading(style_name: str) -> int | None:
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading "):
        level = style_name.removeprefix("Heading ")
        return min(int(level), 6) if level.isdigit() and int(level) >= 1 else None
    return None


def _docx_lines(data: bytes) -> list[str]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(io.BytesIO(data))
    lines = []
    # Body order: paragraphs and tables interleaved as the author wrote them.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = _squash(paragraph.text)
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            level = _docx_heading(style or "")
            if level is not None:
                lines.append(f"{'#' * level} {text}")
            elif (style or "").startswith("List"):
                lines.append(f"- {text}")
            else:
                lines.append(text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                cells = [_squash(cell.text) for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
    return lines


def _docx(data: bytes, max_characters: int) -> dict:
    if not docx_archive_ok(data):
        return {"error": DOCX_PARSE_FAILED}
    try:
        text = _clean("\n\n".join(_docx_lines(data)))
    except Exception:
        return {"error": DOCX_PARSE_FAILED}
    if len(text) > max_characters:
        return {"error": TEXT_LIMIT}
    return {"pages": [text]}


class _HTMLText:
    """Walks an lxml tree, flushing a line at each block boundary."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.buffer: list[str] = []

    def flush(self) -> None:
        text = _squash("".join(self.buffer))
        self.buffer = []
        if text and text != "-":
            self.lines.append(text)

    def walk(self, element) -> None:
        tag = element.tag.lower() if isinstance(element.tag, str) else ""
        if tag in _HTML_HEADINGS:
            self.flush()
            text = _squash(element.text_content())
            if text:
                self.lines.append(f"{'#' * _HTML_HEADINGS[tag]} {text}")
            return
        if tag == "tr":
            self.flush()
            cells = [
                _squash(cell.text_content())
                for cell in element
                if isinstance(cell.tag, str) and cell.tag.lower() in ("td", "th")
            ]
            if any(cells):
                self.lines.append(" | ".join(cells))
            return
        block = tag in _HTML_BLOCKS
        if block:
            self.flush()
        if tag == "li":
            self.buffer.append("- ")
        if element.text:
            self.buffer.append(element.text)
        for child in element:
            self.walk(child)
            if child.tail:
                self.buffer.append(child.tail)
        if block:
            self.flush()


def _html(data: bytes, max_characters: int) -> dict:
    import lxml.html

    try:
        # No network, no comments or processing instructions. The HTML parser never
        # loads external entities or DTDs.
        parser = lxml.html.HTMLParser(
            encoding="utf-8", no_network=True, remove_comments=True, remove_pis=True
        )
        root = lxml.html.document_fromstring(data, parser=parser)
        for dropped in root.xpath("|".join(f"//{tag}" for tag in _HTML_DROPPED)):
            dropped.drop_tree()
        body = root.find("body")
        walker = _HTMLText()
        walker.walk(body if body is not None else root)
        walker.flush()
        text = _clean("\n\n".join(walker.lines))
    except Exception:
        return {"error": HTML_PARSE_FAILED}
    if len(text) > max_characters:
        return {"error": TEXT_LIMIT}
    return {"pages": [text]}


def extract(kind: str, data: bytes, max_pages: int, max_characters: int) -> dict:
    if kind == "pdf":
        return _pdf(data, max_pages, max_characters)
    if kind == "docx":
        return _docx(data, max_characters)
    if kind == "html":
        return _html(data, max_characters)
    raise ValueError("unknown format")


def main(argv: list[str]) -> int:
    kind = argv[0]
    max_pages, max_characters, memory_bytes, cpu_seconds, max_input = map(int, argv[1:])
    try:
        _limit(resource.RLIMIT_AS, memory_bytes)
        _limit(resource.RLIMIT_CPU, cpu_seconds)
        _limit(resource.RLIMIT_FSIZE, 0)
        _limit(resource.RLIMIT_CORE, 0)
    except (ValueError, OSError):
        # Fail closed, and say so: this is the worker's fault, not the document's.
        return SANDBOX_UNAVAILABLE_EXIT
    failed = {"pdf": PDF_PARSE_FAILED, "docx": DOCX_PARSE_FAILED, "html": HTML_PARSE_FAILED}[kind]
    data = sys.stdin.buffer.read(max_input + 1)
    if len(data) > max_input:
        result = {"error": failed}
    else:
        try:
            result = extract(kind, data, max_pages, max_characters)
        except MemoryError:
            result = {"error": failed}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
