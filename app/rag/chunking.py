"""Split page texts into overlapping chunks with LangChain text splitters.

CPU-bound: callers run `chunk_document` in a worker thread.
"""

from dataclasses import dataclass

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError
from app.rag.parsing import ParsedDocument, ParseFailure


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    page: int | None
    content: str


def _splitter(content_type: ContentType, settings: RAGSettings) -> RecursiveCharacterTextSplitter:
    size, overlap = settings.rag_chunk_size, settings.rag_chunk_overlap
    if content_type is ContentType.MARKDOWN:
        # Prefers heading, code-fence and paragraph boundaries before plain text breaks.
        return RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, chunk_size=size, chunk_overlap=overlap
        )
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def chunk_document(
    parsed: ParsedDocument, content_type: ContentType, settings: RAGSettings
) -> list[Chunk]:
    """Chunk each page separately, so every chunk cites exactly one page."""
    splitter = _splitter(content_type, settings)
    chunks: list[Chunk] = []
    for page, text in parsed.pages:
        for piece in splitter.split_text(text):
            content = piece.strip()
            if not content:
                continue
            if len(chunks) >= settings.rag_max_chunks:
                raise ParseFailure(IngestError.CHUNK_LIMIT)
            chunks.append(Chunk(ordinal=len(chunks), page=page, content=content))
    if not chunks:
        raise ParseFailure(IngestError.NO_TEXT)
    return chunks
