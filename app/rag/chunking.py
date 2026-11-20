"""Split page texts into overlapping chunks with LangChain text splitters.

Markdown chunks also record their heading path ("Pets > Dogs"): the headings in force
where the chunk starts, ignoring "#" lines inside code fences. The path goes into the
answer prompt only; embeddings stay on the chunk text.

CPU-bound: callers run `chunk_document` in a worker thread.
"""

import re
from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.rag.config import RAGSettings
from app.rag.contracts import HEADED_TYPES, ContentType, IngestError
from app.rag.parsing import ParsedDocument, ParseFailure

HEADING_CHARACTERS = 300
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    page: int | None
    content: str
    heading: str | None = None


def _headings(text: str) -> list[tuple[int, int, str]]:
    """(offset, level, title) of every Markdown heading outside code fences."""
    found = []
    fenced = False
    offset = 0
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (match := _HEADING.match(line.rstrip("\r\n"))):
            found.append((offset, len(match.group(1)), match.group(2).strip()))
        offset += len(line)
    return found


def _path(headings: list[tuple[int, int, str]], start: int) -> str | None:
    """The heading stack in force at `start`, joined with " > ", capped in length."""
    stack: list[tuple[int, str]] = []
    for offset, level, title in headings:
        if offset > start:
            break
        stack = [entry for entry in stack if entry[0] < level]
        stack.append((level, title))
    if not stack:
        return None
    return " > ".join(title for _, title in stack)[:HEADING_CHARACTERS]


def _splitter(content_type: ContentType, settings: RAGSettings) -> RecursiveCharacterTextSplitter:
    size, overlap = settings.rag_chunk_size, settings.rag_chunk_overlap
    if content_type in HEADED_TYPES:
        # Prefers heading, code-fence and paragraph boundaries before plain text breaks.
        # Word and HTML arrive from the child with "#" headings, so they split the same.
        return RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def chunk_document(
    parsed: ParsedDocument, content_type: ContentType, settings: RAGSettings
) -> list[Chunk]:
    """Chunk each page separately, so every chunk cites exactly one page."""
    splitter = _splitter(content_type, settings)
    markdown = content_type in HEADED_TYPES
    chunks: list[Chunk] = []
    for page, text in parsed.pages:
        headings = _headings(text) if markdown else []
        # Chunks come in text order; overlap means the next one may start before this ends.
        cursor = 0
        for piece in splitter.split_text(text):
            content = piece.strip()
            if not content:
                continue
            if len(chunks) >= settings.rag_max_chunks:
                raise ParseFailure(IngestError.CHUNK_LIMIT)
            start = text.find(content, cursor)
            if start < 0:
                start = max(text.find(content), 0)
            cursor = start + 1
            heading = _path(headings, start) if markdown else None
            chunks.append(Chunk(ordinal=len(chunks), page=page, content=content, heading=heading))
    if not chunks:
        raise ParseFailure(IngestError.NO_TEXT)
    return chunks
