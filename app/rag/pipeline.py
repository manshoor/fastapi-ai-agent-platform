"""Synchronous RAG query: admit, retrieve, answer in quoted claims, check, cite, record usage.

    slot -> admit (per-owner lock, window count, usage row) -> any searchable chunk?
      no  -> no_context, reason no_chunks (no provider call)
      yes -> LCEL chain: retriever (embed question + owner KNN)
          -> relevance margin -> optional LLM rerank (empty -> no_context, rerank_empty)
          -> neighbor windows (same owner scope) -> passages -> claims
          -> no claims                    -> no_context, reason no_claims
          -> any claim fails its check    -> no_context, reason = that check
          -> all claims verified          -> answered, citations with their quotes
          -> usage row (summed chat usage, reason, effort) -> 200

The answer is the claim statements joined with spaces. Refusal is all-or-nothing: one
failed claim refuses the whole answer, since dropping it could flip the meaning.

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
from app.ai.provider import (
    RAG_PROMPT_VERSION,
    AnswerGeneration,
    Passage,
    ProviderFailure,
    Ranking,
)
from app.models import RAGQuery
from app.rag.config import RERANK_LLM, RAGSettings, RerankMode, query_deadline_floor
from app.rag.contracts import (
    Citation,
    DiagnosticClaim,
    DiagnosticPassage,
    DiagnosticsRequest,
    DiagnosticsResponse,
    DiagnosticUsage,
    QueryRequest,
    QueryResponse,
)
from app.rag.retrieval import OwnerScopedRetriever, has_context
from app.rag.runtime import RAGRuntime
from app.rag.verification import check_claims

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
# The answer call runs without reasoning until the effort setting exists.
_REASONING_EFFORT = "none"
# No-context reasons decided here; claim check reasons come from app.rag.verification.
NO_CHUNKS = "no_chunks"
NO_PASSAGES = "no_passages"
NO_CLAIMS = "no_claims"
RERANK_EMPTY = "rerank_empty"
# Rerank reads up to twice the requested passages, within the prompt's label range.
MAX_CANDIDATES = 8
# Diagnostics: a claim that passed, or one after the first failure (never checked).
_VERIFIED = "verified"
_NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class _Options:
    """Answer-context settings for one query: the server's, or eval overrides."""

    neighbor_chunks: int
    relevance_margin: float | None
    rerank: RerankMode

    def candidates(self, top_k: int) -> int:
        return min(MAX_CANDIDATES, 2 * top_k) if self.rerank == RERANK_LLM else top_k


def _options(settings: RAGSettings, request: QueryRequest) -> _Options:
    overrides = getattr(request, "overrides", None)
    options = _Options(
        neighbor_chunks=settings.rag_neighbor_chunks,
        relevance_margin=settings.rag_relevance_margin,
        rerank=settings.rag_rerank,
    )
    if overrides is None:
        return options
    return _Options(
        neighbor_chunks=_pick(overrides.neighbor_chunks, options.neighbor_chunks),
        relevance_margin=_pick(overrides.relevance_margin, options.relevance_margin),
        rerank=_pick(overrides.rerank, options.rerank),
    )


def _pick(override, default):
    return default if override is None else override


@dataclass(frozen=True)
class _Outcome:
    status: str
    answer: str
    citations: list[Citation]
    retrieved: int
    generation: AnswerGeneration | None = None
    # The passages sent to the model, in label order (P1 first).
    documents: tuple[Document, ...] = ()
    no_context_reason: str | None = None


def _add(total: int | None, tokens: int | None) -> int | None:
    if tokens is None:
        return total
    return (total or 0) + tokens


@dataclass
class _Usage:
    """Billed usage of one query: embeddings plus every completed chat call.

    Filled as calls complete, so a failure after a billed answer still records it.
    A call that raises is not counted, even when the provider billed it (truncated or
    invalid output): the adapter does not return that call's usage.
    """

    embedding_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    chat_calls: int = 0
    response_model: str | None = None
    request_id: str | None = None

    def add(self, generation: AnswerGeneration | Ranking) -> None:
        self.chat_calls += 1
        self.input_tokens = _add(self.input_tokens, generation.input_tokens)
        self.output_tokens = _add(self.output_tokens, generation.output_tokens)
        self.response_model = generation.model
        self.request_id = generation.request_id


@dataclass(frozen=True)
class _Result:
    outcome: _Outcome
    usage: _Usage


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
    usage: _Usage,
    retrieved: int | None = None,
    no_context_reason: str | None = None,
    error_code: str | None = None,
) -> None:
    """Best effort: a failed usage write never turns a paid answer into an error."""
    values = {
        "status": status,
        "retrieved_count": retrieved,
        "embedding_tokens": usage.embedding_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "chat_calls": usage.chat_calls,
        "response_model": usage.response_model,
        "provider_request_id": usage.request_id,
        "reasoning_effort": _REASONING_EFFORT,
        "no_context_reason": no_context_reason,
        "error_code": error_code,
        "finished_at": func.clock_timestamp(),
    }
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
    """One citation per cited passage, in claim order, with that passage's quotes."""
    citations = []
    for index in generation.citations:
        label = f"P{index + 1}"
        metadata = documents[index].metadata
        citations.append(
            Citation(
                document_id=metadata["document_id"],
                title=metadata["title"],
                ordinal=metadata["ordinal"],
                page=metadata["page"],
                # A window cites its anchor chunk: the snippet is that chunk's text.
                snippet=metadata.get("anchor_text", documents[index].page_content)[
                    :SNIPPET_CHARACTERS
                ],
                quotes=[claim.quote for claim in generation.claims if claim.citation == label],
            )
        )
    return citations


def _passage(document: Document) -> Passage:
    metadata = document.metadata
    return Passage(
        title=metadata["title"],
        page=metadata["page"],
        text=document.page_content,
        section=metadata.get("heading"),
    )


def _within_margin(documents: list[Document], margin: float | None) -> list[Document]:
    """Drop candidates much farther than the closest one; the closest always stays."""
    if margin is None or not documents:
        return documents
    limit = documents[0].metadata["distance"] + margin
    return [document for document in documents if document.metadata["distance"] <= limit]


def _build_chain(
    runtime: RAGRuntime,
    retriever: OwnerScopedRetriever,
    usage: _Usage,
    options: _Options,
    top_k: int,
):
    async def generate(inputs: dict) -> _Outcome:
        documents: list[Document] = _within_margin(inputs["documents"], options.relevance_margin)
        if not documents:
            return _Outcome("no_context", NO_CONTEXT_ANSWER, [], 0, no_context_reason=NO_PASSAGES)

        # Rerank picks and orders retrieved chunks before any window widens them.
        if options.rerank == RERANK_LLM:
            ranking = await runtime.reranker.rerank(
                inputs["question"],
                [_passage(document) for document in documents],
                min(top_k, len(documents)),
            )
            usage.add(ranking)
            documents = [documents[index] for index in ranking.order]
            if not documents:
                return _Outcome(
                    "no_context", NO_CONTEXT_ANSWER, [], 0, no_context_reason=RERANK_EMPTY
                )
        documents = await retriever.expand(documents, options.neighbor_chunks)
        passages = [_passage(document) for document in documents]
        generation = await runtime.answerer.answer(inputs["question"], passages)
        usage.add(generation)
        sent = tuple(documents)

        # Unquoted or misquoted text is not grounded: never show it as an answer.
        failure = (
            check_claims(generation.claims, [passage.text for passage in passages])
            if generation.claims
            else NO_CLAIMS
        )
        if failure is not None:
            return _Outcome(
                "no_context",
                NO_CONTEXT_ANSWER,
                [],
                len(documents),
                generation,
                sent,
                no_context_reason=str(failure),
            )
        return _Outcome(
            "answered",
            generation.answer,
            _citations(documents, generation),
            len(documents),
            generation,
            sent,
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
    result = await _run_query(
        session_factory=session_factory, runtime=runtime, owner_id=owner_id, request=request
    )
    outcome = result.outcome
    return QueryResponse(status=outcome.status, answer=outcome.answer, citations=outcome.citations)


async def diagnose_query(
    *,
    session_factory: SessionFactory,
    runtime: RAGRuntime,
    owner_id: UUID,
    request: DiagnosticsRequest,
) -> DiagnosticsResponse:
    """Eval-only: the same query, plus the exact passages sent to the model and the usage.

    The passage text returns only to its owner, in the response. It never reaches a span,
    a log line or the usage row. Overrides that the query deadline cannot fit are refused
    before admission.
    """
    options = _options(runtime.settings, request)
    if runtime.settings.rag_query_deadline_seconds <= query_deadline_floor(
        runtime.settings, options.rerank
    ):
        raise HTTPException(422, "Overrides do not fit the query deadline")
    result = await _run_query(
        session_factory=session_factory, runtime=runtime, owner_id=owner_id, request=request
    )
    outcome = result.outcome
    usage = result.usage
    passages = [
        DiagnosticPassage(
            label=f"P{number}",
            document_id=document.metadata["document_id"],
            title=document.metadata["title"],
            ordinal=document.metadata["ordinal"],
            page=document.metadata["page"],
            distance=document.metadata["distance"],
            section=document.metadata.get("heading"),
            text=document.page_content,
        )
        for number, document in enumerate(outcome.documents, start=1)
    ]
    return DiagnosticsResponse(
        status=outcome.status,
        answer=outcome.answer,
        citations=outcome.citations,
        no_context_reason=outcome.no_context_reason,
        passages=passages,
        claims=_diagnostic_claims(outcome),
        verdicts=None,
        usage=DiagnosticUsage(
            embedding_tokens=usage.embedding_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            chat_calls=usage.chat_calls,
            reasoning_effort=_REASONING_EFFORT,
        ),
    )


def _diagnostic_claims(outcome: _Outcome) -> list[DiagnosticClaim] | None:
    """Each claim with its own check result, up to and including the first failure."""
    generation = outcome.generation
    if generation is None:
        return None
    texts = [document.page_content for document in outcome.documents]
    claims = []
    failed = False
    for claim in generation.claims:
        check = _NOT_CHECKED
        if not failed:
            failure = check_claims([claim], texts)
            failed = failure is not None
            check = str(failure) if failed else _VERIFIED
        claims.append(
            DiagnosticClaim(
                statement=claim.statement, citation=claim.citation, quote=claim.quote, check=check
            )
        )
    return claims


async def _run_query(
    *,
    session_factory: SessionFactory,
    runtime: RAGRuntime,
    owner_id: UUID,
    request: QueryRequest,
) -> _Result:
    async with runtime.query_slot():
        query_id = await _admit(
            session_factory=session_factory, runtime=runtime, owner_id=owner_id, top_k=request.top_k
        )
        telemetry = runtime.telemetry
        started = time.monotonic()
        options = _options(runtime.settings, request)
        with telemetry.span("rag.query", query_id=str(query_id), top_k=request.top_k):
            retriever = OwnerScopedRetriever(
                session_factory=session_factory,
                embedder=runtime.embedder,
                owner_id=owner_id,
                embedding_model=runtime.settings.ai_embedding_model,
                top_k=options.candidates(request.top_k),
                document_ids=request.document_ids,
                telemetry=telemetry,
            )

            usage = _Usage()

            # Every exit after admission closes the usage row, paid or not.
            async def fail(code: str) -> None:
                usage.embedding_tokens = retriever.embedding_tokens
                await _record(
                    session_factory=session_factory,
                    query_id=query_id,
                    status="failed",
                    usage=usage,
                    error_code=code,
                )
                telemetry.increment("rag.queries", status="failed", error_code=code)

            try:
                outcome = await _answer(runtime, retriever, request, usage, options)
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
            usage.embedding_tokens = retriever.embedding_tokens
            await _record(
                session_factory=session_factory,
                query_id=query_id,
                status=outcome.status,
                usage=usage,
                retrieved=outcome.retrieved,
                no_context_reason=outcome.no_context_reason,
            )
            # The no-context reason is a fixed code: safe as a metric tag.
            tags = {"outcome": outcome.no_context_reason} if outcome.no_context_reason else {}
            telemetry.increment("rag.queries", status=outcome.status, **tags)
            telemetry.observe(
                "rag.query.duration", time.monotonic() - started, status=outcome.status
            )
            return _Result(outcome, usage)


async def _answer(
    runtime: RAGRuntime,
    retriever: OwnerScopedRetriever,
    request: QueryRequest,
    usage: _Usage,
    options: _Options,
) -> _Outcome:
    """No searchable chunk: no provider call. Otherwise run the chain under the deadline."""
    if not await has_context(
        session_factory=retriever.session_factory,
        owner_id=retriever.owner_id,
        embedding_model=retriever.embedding_model,
        document_ids=request.document_ids,
    ):
        return _Outcome("no_context", NO_CONTEXT_ANSWER, [], 0, no_context_reason=NO_CHUNKS)
    chain = _build_chain(runtime, retriever, usage, options, request.top_k)
    async with asyncio.timeout(runtime.settings.rag_query_deadline_seconds):
        with tracing_context(enabled=False):
            return await chain.ainvoke(request.question)
