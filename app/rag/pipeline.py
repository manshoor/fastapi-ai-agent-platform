"""Synchronous RAG query: admit, retrieve, answer, cite, record usage.

    slot -> admit (per-owner lock, window count, usage row) -> any searchable chunk?
      no  -> no_context (no provider call)
      yes -> LCEL chain: retriever (embed question + owner KNN) -> passages -> answer
          -> citations mapped back to chunks -> usage row -> 200

No database transaction is open during provider calls. The chain runs with LangSmith
tracing disabled, so no question, passage or answer text can leave through it.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langsmith import tracing_context
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import SQLAlchemyError

from app.ai.processing import SessionFactory
from app.ai.provider import RAG_PROMPT_VERSION, AnswerGeneration, Passage, ProviderFailure
from app.models import RAGQuery
from app.rag.contracts import Citation, QueryRequest, QueryResponse
from app.rag.retrieval import OwnerScopedRetriever, has_context
from app.rag.runtime import RAGRuntime

logger = logging.getLogger(__name__)
# Two-key advisory lock namespace for per-owner query admission ("RAGQ").
_QUERY_LOCK_NAMESPACE = 0x52414751
SNIPPET_CHARACTERS = 300
NO_CONTEXT_ANSWER = "No relevant passages were found in your documents."
_UNAVAILABLE_CODES = frozenset({"provider_rate_limit"})
_TIMEOUT_CODES = frozenset({"provider_timeout"})
DEADLINE_CODE = "query_deadline"
DATABASE_CODE = "database_unavailable"
CANCELLED_CODE = "query_cancelled"


@dataclass(frozen=True)
class _Outcome:
    status: str
    answer: str
    citations: list[Citation]
    retrieved: int
    generation: AnswerGeneration | None = None


async def _admit(
    *, session_factory: SessionFactory, runtime: RAGRuntime, owner_id: UUID, top_k: int
) -> UUID:
    """Count this owner's recent queries under a per-owner lock, then record this one."""
    settings = runtime.settings
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:owner))"),
                {"namespace": _QUERY_LOCK_NAMESPACE, "owner": str(owner_id)},
            )
            recent = await session.scalar(
                select(func.count())
                .select_from(RAGQuery)
                .where(
                    RAGQuery.owner_id == owner_id,
                    RAGQuery.created_at
                    > func.clock_timestamp() - timedelta(seconds=settings.rag_query_window_seconds),
                )
            )
            if recent >= settings.rag_query_limit:
                retry_after = max(1, round(settings.rag_query_window_seconds))
                raise HTTPException(
                    429, "Query limit reached", headers={"Retry-After": str(retry_after)}
                )
            query_id = uuid4()
            session.add(
                RAGQuery(
                    id=query_id,
                    owner_id=owner_id,
                    status="admitted",
                    top_k=top_k,
                    embedding_model=settings.ai_embedding_model,
                    model=settings.ai_model,
                    prompt_version=RAG_PROMPT_VERSION,
                    created_at=func.clock_timestamp(),
                )
            )
    return query_id


async def _record(
    *,
    session_factory: SessionFactory,
    query_id: UUID,
    status: str,
    retrieved: int | None = None,
    embedding_tokens: int | None = None,
    generation: AnswerGeneration | None = None,
    error_code: str | None = None,
) -> None:
    """Best effort: a failed usage write never turns a paid answer into an error."""
    values = {
        "status": status,
        "retrieved_count": retrieved,
        "embedding_tokens": embedding_tokens,
        "error_code": error_code,
        "finished_at": func.clock_timestamp(),
    }
    if generation is not None:
        values.update(
            response_model=generation.model,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            provider_request_id=generation.request_id,
        )
    try:
        async with session_factory() as session:
            await session.execute(
                update(RAGQuery)
                .where(RAGQuery.id == query_id, RAGQuery.status == "admitted")
                .values(**values)
            )
            await session.commit()
    except (SQLAlchemyError, OSError) as exc:
        logger.error("RAG usage update failure exception=%s", type(exc).__name__)


def _citations(documents: list[Document], generation: AnswerGeneration) -> list[Citation]:
    return [
        Citation(
            document_id=documents[index].metadata["document_id"],
            title=documents[index].metadata["title"],
            ordinal=documents[index].metadata["ordinal"],
            page=documents[index].metadata["page"],
            snippet=documents[index].page_content[:SNIPPET_CHARACTERS],
        )
        for index in generation.citations
    ]


def _build_chain(runtime: RAGRuntime, retriever: OwnerScopedRetriever):
    async def generate(inputs: dict) -> _Outcome:
        documents: list[Document] = inputs["documents"]
        if not documents:
            return _Outcome("no_context", NO_CONTEXT_ANSWER, [], 0)
        passages = [
            Passage(
                title=document.metadata["title"],
                page=document.metadata["page"],
                text=document.page_content,
            )
            for document in documents
        ]
        generation = await runtime.answerer.answer(inputs["question"], passages)
        if not generation.citations:
            # Uncited text is not grounded: never show it as an answer.
            return _Outcome("no_context", NO_CONTEXT_ANSWER, [], len(documents), generation)
        return _Outcome(
            "answered",
            generation.answer,
            _citations(documents, generation),
            len(documents),
            generation,
        )

    return RunnableParallel(documents=retriever, question=RunnablePassthrough()) | RunnableLambda(
        generate
    )


def _http_error(code: str) -> HTTPException:
    if code in _UNAVAILABLE_CODES:
        return HTTPException(503, "Answer provider busy", headers={"Retry-After": "5"})
    if code in _TIMEOUT_CODES or code == DEADLINE_CODE:
        return HTTPException(504, "Answer timed out")
    return HTTPException(502, "Answer provider failed")


async def answer_query(
    *,
    session_factory: SessionFactory,
    runtime: RAGRuntime,
    owner_id: UUID,
    request: QueryRequest,
) -> QueryResponse:
    async with runtime.query_slot():
        query_id = await _admit(
            session_factory=session_factory, runtime=runtime, owner_id=owner_id, top_k=request.top_k
        )
        telemetry = runtime.telemetry
        started = time.monotonic()
        with telemetry.span("rag.query", query_id=str(query_id), top_k=request.top_k):
            retriever = OwnerScopedRetriever(
                session_factory=session_factory,
                embedder=runtime.embedder,
                owner_id=owner_id,
                embedding_model=runtime.settings.ai_embedding_model,
                top_k=request.top_k,
                document_ids=request.document_ids,
                telemetry=telemetry,
            )

            # Every exit after admission closes the usage row, paid or not.
            async def fail(code: str) -> None:
                await _record(
                    session_factory=session_factory,
                    query_id=query_id,
                    status="failed",
                    embedding_tokens=retriever.embedding_tokens,
                    error_code=code,
                )
                telemetry.increment("rag.queries", status="failed", error_code=code)

            try:
                outcome = await _answer(runtime, retriever, request)
            except (ProviderFailure, TimeoutError) as exc:
                code = exc.code if isinstance(exc, ProviderFailure) else DEADLINE_CODE
                await fail(code)
                raise _http_error(code) from None
            except (SQLAlchemyError, OSError):
                # The app's database handler maps the error itself to a fixed response.
                await fail(DATABASE_CODE)
                raise
            except asyncio.CancelledError:
                await asyncio.shield(fail(CANCELLED_CODE))
                raise
            telemetry.annotate(retrieved_count=outcome.retrieved)
            await _record(
                session_factory=session_factory,
                query_id=query_id,
                status=outcome.status,
                retrieved=outcome.retrieved,
                embedding_tokens=retriever.embedding_tokens,
                generation=outcome.generation,
            )
            telemetry.increment("rag.queries", status=outcome.status)
            telemetry.observe(
                "rag.query.duration", time.monotonic() - started, status=outcome.status
            )
            return QueryResponse(
                status=outcome.status, answer=outcome.answer, citations=outcome.citations
            )


async def _answer(
    runtime: RAGRuntime, retriever: OwnerScopedRetriever, request: QueryRequest
) -> _Outcome:
    """No searchable chunk: no provider call. Otherwise run the chain under the deadline."""
    if not await has_context(
        session_factory=retriever.session_factory,
        owner_id=retriever.owner_id,
        embedding_model=retriever.embedding_model,
        document_ids=request.document_ids,
    ):
        return _Outcome("no_context", NO_CONTEXT_ANSWER, [], 0)
    chain = _build_chain(runtime, retriever)
    async with asyncio.timeout(runtime.settings.rag_query_deadline_seconds):
        with tracing_context(enabled=False):
            return await chain.ainvoke(request.question)
