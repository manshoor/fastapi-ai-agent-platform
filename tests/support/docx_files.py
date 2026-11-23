"""Build small Word files and hostile ZIPs in memory for tests. No binary fixtures."""

import io
import zipfile

from docx import Document


def word_file() -> bytes:
    """Title, headings, a paragraph, a list item and a table, in that order."""
    document = Document()
    document.add_heading("Pump manual", level=0)
    document.add_heading("Flow", level=1)
    document.add_paragraph("The pump runs at 40 litres per minute.")
    document.add_paragraph("Check the filter weekly.", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Mode", "Litres"
    table.cell(1, 0).text, table.cell(1, 1).text = "Eco", "25"
    document.add_heading("Noise", level=2)
    document.add_paragraph("It stays under 50 dB.")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def zip_file(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def many_entries(count: int) -> bytes:
    return zip_file({"word/document.xml": b"<x/>"} | {f"e{i}": b"" for i in range(count)})


def inflating(uncompressed: int) -> bytes:
    """A small archive that claims (and inflates to) `uncompressed` bytes."""
    return zip_file({"word/document.xml": b"<x/>", "pad.bin": b"\0" * uncompressed})
