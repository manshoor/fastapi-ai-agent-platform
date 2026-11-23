"""Word and HTML: upload checks, sandboxed parsing, and heading-aware chunks."""

import signal

import httpx
import pytest

from app.config import Settings
from app.rag.chunking import chunk_document
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError
from app.rag.extract import DOCX_MAX_ENTRIES, DOCX_MAX_UNCOMPRESSED, docx_archive_ok
from app.rag.parsing import ParseFailure, parse_document
from tests.support.docx_files import inflating, many_entries, word_file, zip_file
from tests.test_rag_documents import upload
from tests.test_rag_parsing import spawned  # noqa: F401

HTML = b"""<!DOCTYPE html>
<html><head><title>Ignored head</title><style>body{color:red}</style></head>
<body>
<h1>Pump manual</h1>
<!-- hidden comment -->
<script>alert("script text")</script>
<noscript>noscript text</noscript>
<template><p>template text</p></template>
<h2>Flow</h2>
<p>The pump runs at <b>40</b> litres per minute.</p>
<ul><li>Check the filter weekly.</li></ul>
<table><tr><th>Mode</th><th>Litres</th></tr><tr><td>Eco</td><td>25</td></tr></table>
</body></html>"""


def parse_settings(**updates) -> RAGSettings:
    return RAGSettings(ai_enabled=True, rag_enabled=True).model_copy(update=updates)


@pytest.fixture
def settings(monkeypatch) -> Settings:
    # The app's RAG runtime reads its settings from the environment.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    return Settings(ai_enabled=True, rag_enabled=True)


async def test_word_keeps_order_headings_lists_and_tables(spawned):  # noqa: F811
    parsed = await parse_document(word_file(), ContentType.DOCX, parse_settings())
    assert parsed.page_count is None and len(parsed.pages) == 1
    page, text = parsed.pages[0]
    assert page is None
    assert text.split("\n\n") == [
        "# Pump manual",
        "# Flow",
        "The pump runs at 40 litres per minute.",
        "- Check the filter weekly.",
        "Mode | Litres",
        "Eco | 25",
        "## Noise",
        "It stays under 50 dB.",
    ]
    args, kwargs, _ = spawned[0]
    assert args[1:5] == ("-I", "-m", "app.rag.extract", "docx")
    assert kwargs["start_new_session"] is True


async def test_word_headings_become_chunk_heading_paths():
    parsed = await parse_document(word_file(), ContentType.DOCX, parse_settings())
    chunks = chunk_document(
        parsed, ContentType.DOCX, parse_settings(rag_chunk_size=100, rag_chunk_overlap=0)
    )
    noise = next(chunk for chunk in chunks if "50 dB" in chunk.content)
    assert noise.heading == "Flow > Noise"


async def test_html_resolves_no_external_entity():
    hostile = (
        b'<!DOCTYPE html [<!ENTITY leak SYSTEM "file:///etc/passwd">]>'
        b"<html><body><p>Before &leak; after.</p></body></html>"
    )
    text = (await parse_document(hostile, ContentType.HTML, parse_settings())).pages[0][1]
    assert "Before" in text and "after." in text
    assert "root:" not in text


async def test_html_drops_active_content():
    parsed = await parse_document(HTML, ContentType.HTML, parse_settings())
    text = parsed.pages[0][1]
    assert text.split("\n\n") == [
        "# Pump manual",
        "## Flow",
        "The pump runs at 40 litres per minute.",
        "- Check the filter weekly.",
        "Mode | Litres",
        "Eco | 25",
    ]
    for hidden in (
        "Ignored head",
        "color:red",
        "hidden comment",
        "script text",
        "noscript",
        "template text",
    ):
        assert hidden not in text


@pytest.mark.parametrize(
    ("data", "valid"),
    [
        (word_file(), True),
        (b"not a zip", False),
        (zip_file({"other.xml": b"<x/>"}), False),
        (many_entries(DOCX_MAX_ENTRIES), False),
        (inflating(DOCX_MAX_UNCOMPRESSED + 1), False),
    ],
    ids=["word", "not-zip", "no-body", "entries", "inflated"],
)
def test_docx_archive_bounds(data, valid):
    assert docx_archive_ok(data) is valid


async def test_hostile_word_fails_with_a_fixed_code():
    with pytest.raises(ParseFailure) as error:
        await parse_document(
            inflating(DOCX_MAX_UNCOMPRESSED + 1), ContentType.DOCX, parse_settings()
        )
    assert error.value.code is IngestError.DOCX_PARSE_FAILED


async def test_word_text_limit():
    with pytest.raises(ParseFailure) as error:
        await parse_document(word_file(), ContentType.DOCX, parse_settings(rag_max_characters=20))
    assert error.value.code is IngestError.TEXT_LIMIT


async def test_slow_html_child_is_killed(spawned):  # noqa: F811
    with pytest.raises(ParseFailure) as error:
        await parse_document(HTML, ContentType.HTML, parse_settings(rag_pdf_timeout_seconds=0.01))
    assert error.value.code is IngestError.PARSE_TIMEOUT
    assert spawned[0][2].returncode == -signal.SIGKILL


@pytest.mark.parametrize(
    ("name", "data", "status", "content_type"),
    [
        (
            "Manual.DOCX",
            word_file(),
            202,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("page.html", HTML, 202, "text/html"),
        ("page.htm", b"<p>Short page.</p>", 202, "text/html"),
        ("bomb.docx", many_entries(DOCX_MAX_ENTRIES), 422, None),
        ("fake.docx", b"plain text", 422, None),
        ("page.html", b"\xff\xfe<p>x</p>", 422, None),
    ],
    ids=["docx", "html", "htm", "bomb", "not-zip", "not-utf8"],
)
async def test_word_and_html_uploads(
    client: httpx.AsyncClient, account, name, data, status, content_type
):
    response = await upload(client, await account(), name=name, data=data)
    assert response.status_code == status
    if content_type is not None:
        assert response.json()["content_type"] == content_type
