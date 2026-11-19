"""LangChain-based chunking: size, overlap, page attribution, limits, Unicode."""

import pytest

from app.rag.chunking import chunk_document
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError
from app.rag.parsing import ParsedDocument, ParseFailure


def settings(**updates) -> RAGSettings:
    return RAGSettings(ai_enabled=True, rag_enabled=True).model_copy(update=updates)


def words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_chunks_respect_size_and_overlap():
    text = words(400)
    chunks = chunk_document(
        ParsedDocument(pages=[(None, text)], page_count=None),
        ContentType.TEXT,
        settings(rag_chunk_size=200, rag_chunk_overlap=50),
    )
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 200 for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    # Consecutive chunks share text, and together they cover every word.
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert left.content.split()[-1] in right.content.split()
    covered = set(" ".join(chunk.content for chunk in chunks).split())
    assert covered == set(text.split())


def test_pdf_chunks_never_cross_pages():
    parsed = ParsedDocument(pages=[(1, words(60, "alpha")), (2, words(60, "beta"))], page_count=2)
    chunks = chunk_document(
        parsed, ContentType.PDF, settings(rag_chunk_size=150, rag_chunk_overlap=20)
    )
    assert {chunk.page for chunk in chunks} == {1, 2}
    for chunk in chunks:
        expected = "alpha" if chunk.page == 1 else "beta"
        assert all(word.startswith(expected) for word in chunk.content.split())


def test_markdown_prefers_heading_boundaries():
    text = "# Pumps\n\n" + words(20, "pump") + "\n\n# Filters\n\n" + words(20, "filter")
    chunks = chunk_document(
        ParsedDocument(pages=[(None, text)], page_count=None),
        ContentType.MARKDOWN,
        settings(rag_chunk_size=200, rag_chunk_overlap=0),
    )
    assert [chunk.content.splitlines()[0] for chunk in chunks] == ["# Pumps", "# Filters"]


def test_unicode_is_preserved():
    text = "東京の天気は晴れです。 Café crème ✓ " * 20
    chunks = chunk_document(
        ParsedDocument(pages=[(None, text)], page_count=None),
        ContentType.TEXT,
        settings(rag_chunk_size=100, rag_chunk_overlap=10),
    )
    assert "東京" in chunks[0].content and all(len(chunk.content) <= 100 for chunk in chunks)


def test_chunk_limit_and_blank_pages():
    many = ParsedDocument(pages=[(None, words(500))], page_count=None)
    with pytest.raises(ParseFailure) as error:
        chunk_document(
            many,
            ContentType.TEXT,
            settings(rag_chunk_size=100, rag_chunk_overlap=10, rag_max_chunks=3),
        )
    assert error.value.code is IngestError.CHUNK_LIMIT
    blank = ParsedDocument(pages=[(1, "  "), (2, "\n")], page_count=2)
    with pytest.raises(ParseFailure) as error:
        chunk_document(blank, ContentType.PDF, settings())
    assert error.value.code is IngestError.NO_TEXT


def _markdown(text: str, size: int = 200, overlap: int = 0):
    parsed = ParsedDocument([(None, text)], None)
    return chunk_document(
        parsed, ContentType.MARKDOWN, settings(rag_chunk_size=size, rag_chunk_overlap=overlap)
    )


def test_markdown_chunks_carry_their_heading_path():
    text = (
        "# House rules\n\nIntro text.\n\n## Pets\n\n### Dogs\n\n"
        + words(60, "dog")
        + "\n\n## Waste\n\n"
        + words(30, "bin")
    )
    chunks = _markdown(text)
    by_word = {
        word: next(chunk.heading for chunk in chunks if word in chunk.content)
        for word in ("Intro", "dog59", "bin1")
    }
    assert by_word == {
        "Intro": "House rules",
        "dog59": "House rules > Pets > Dogs",
        "bin1": "House rules > Waste",
    }


def test_headings_inside_code_fences_are_ignored():
    text = "## Setup\n\n```\n# not a heading\n```\n\n" + words(80, "step")
    assert {chunk.heading for chunk in _markdown(text)} == {"Setup"}


def test_heading_path_is_capped_and_absent_outside_markdown():
    text = "# " + "h" * 400 + "\n\nBody text here."
    (chunk,) = _markdown(text, size=1200)
    assert len(chunk.heading) == 300
    plain = chunk_document(
        ParsedDocument([(None, "# Not markdown\n\nBody.")], None), ContentType.TEXT, settings()
    )
    assert {chunk.heading for chunk in plain} == {None}


def test_text_before_any_heading_has_no_path():
    (chunk, *_) = _markdown("Preface words.\n\n# Later\n\n" + words(5))
    assert chunk.heading is None
