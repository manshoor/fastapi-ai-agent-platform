"""Build small, valid text PDFs in memory for tests. No binary fixtures in the repo."""

import io


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def text_pdf(pages: list[str]) -> bytes:
    """One Helvetica text line per page. Text must be Latin-1."""
    count = len(pages)
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET"
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * index} 0 R >>"
        )
        length = len(stream.encode("latin-1"))
        objects.append(f"<< /Length {length} >>\nstream\n{stream}\nendstream")
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1"))
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return output.getvalue()


def encrypted_pdf() -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("synthetic-user-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _rendered_gray(text: str, dpi: int) -> tuple[int, int, bytes]:
    """Render one text page to raw 8-bit gray pixels with pdftoppm (in the image)."""
    import subprocess

    pgm = subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-singlefile", "fd://0"],
        input=text_pdf([text]),
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    # Binary PGM: "P5\n<width> <height>\n255\n" then pixels.
    magic, size, depth, pixels = pgm.split(b"\n", 3)
    assert magic == b"P5" and depth == b"255"
    width, height = map(int, size.split())
    return width, height, pixels


def scanned_pdf(pages: list[str], dpi: int = 150) -> bytes:
    """Image-only pages: each page is a picture of its text, with no text layer."""
    import zlib

    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        width, height, pixels = _rendered_gray(text, dpi)
        image = zlib.compress(pixels)
        page_number = len(objects) + 1
        kids.append(f"{page_number} 0 R")
        draw = b"q 612 0 0 792 0 0 cm /Im0 Do Q"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /XObject << /Im0 {page_number + 1} 0 R >> >> "
            f"/Contents {page_number + 2} 0 R >>"
        )
        objects.append(
            (
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
                f"/Length {len(image)} >>\nstream\n"
            ).encode("latin-1")
            + image
            + b"\nendstream"
        )
        objects.append(f"<< /Length {len(draw)} >>\nstream\n{draw.decode()}\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    output = io.BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(output.tell())
        raw = body if isinstance(body, bytes) else body.encode("latin-1")
        output.write(f"{number} 0 obj\n".encode() + raw + b"\nendobj\n")
    xref = output.tell()
    output.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return output.getvalue()
