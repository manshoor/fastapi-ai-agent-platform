"""OCR of scanned PDF pages inside the sandboxed child (pdftoppm + tesseract)."""

import pytest

from app.rag.contracts import ContentType, IngestError
from app.rag.parsing import ParseFailure, parse_document
from tests.support.pdfs import scanned_pdf, text_pdf
from tests.test_rag_formats import parse_settings

SCAN = "Pump flow is forty litres per minute"


def merged(text_page: bytes, scan: bytes) -> bytes:
    """One text page followed by one scanned page."""
    import io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for data in (text_page, scan):
        for page in PdfReader(io.BytesIO(data)).pages:
            writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


async def test_scanned_page_is_read_with_ocr():
    parsed = await parse_document(scanned_pdf([SCAN]), ContentType.PDF, parse_settings())
    assert "forty litres" in parsed.pages[0][1].lower()
    assert (parsed.ocr_pages, parsed.ocr_truncated) == (1, False)


async def test_only_pages_without_text_are_ocr_read():
    data = merged(text_pdf(["A text layer page about valves and filters."]), scanned_pdf([SCAN]))
    parsed = await parse_document(data, ContentType.PDF, parse_settings())
    assert parsed.pages[0] == (1, "A text layer page about valves and filters.")
    assert "forty" in parsed.pages[1][1].lower() and parsed.ocr_pages == 1


async def test_page_cap_reads_the_first_pages_and_records_truncation():
    data = scanned_pdf([SCAN, "Second scanned page about noise"])
    parsed = await parse_document(data, ContentType.PDF, parse_settings(rag_ocr_max_pages=1))
    assert "forty" in parsed.pages[0][1].lower() and parsed.pages[1][1].strip() == ""
    assert (parsed.ocr_pages, parsed.ocr_truncated) == (1, True)


async def test_ocr_off_keeps_the_no_text_failure():
    with pytest.raises(ParseFailure) as error:
        await parse_document(
            scanned_pdf([SCAN]), ContentType.PDF, parse_settings(rag_ocr_enabled=False)
        )
    assert error.value.code is IngestError.NO_TEXT


async def test_page_timeout_skips_the_page():
    with pytest.raises(ParseFailure) as error:
        await parse_document(
            scanned_pdf([SCAN]), ContentType.PDF, parse_settings(rag_ocr_page_seconds=0.001)
        )
    assert error.value.code is IngestError.NO_TEXT


def test_render_is_bounded_whatever_the_page_box():
    """A huge page box renders at most OCR_MAX_SIDE pixels on its longest side."""
    import subprocess

    from app.rag.extract import OCR_DPI, OCR_MAX_SIDE

    huge = text_pdf(["x"]).replace(b"/MediaBox [0 0 612 792]", b"/MediaBox [0 0 14400 14400]")
    pgm = subprocess.run(
        ["pdftoppm", "-r", str(OCR_DPI), "-scale-to", str(OCR_MAX_SIDE), "-gray",
         "-singlefile", "fd://0"],
        input=huge, capture_output=True, check=True, timeout=30,
    ).stdout  # fmt: skip
    width, height = map(int, pgm.split(b"\n", 2)[1].split())
    assert max(width, height) == OCR_MAX_SIDE
