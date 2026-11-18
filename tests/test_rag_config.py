"""RAG settings rules: enforced only when RAG is on, and safe for leases and deadlines."""

import pytest
from pydantic import ValidationError

from app.rag.config import RAGSettings


def build(**values) -> RAGSettings:
    return RAGSettings(ai_enabled=True, **values)


def test_rag_rules_do_not_block_the_rewrite_dispatcher_when_off():
    # Valid for rewrites, invalid for the vector column: must not matter while RAG is off.
    assert build(rag_enabled=False, ai_embedding_dimensions=3072).rag_enabled is False


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"ai_enabled": False}, "requires AI_ENABLED"),
        ({"ai_embedding_dimensions": 3072}, "vector column"),
        (
            {
                "rag_pdf_timeout_seconds": 100,
                "ai_embedding_timeout_seconds": 20,
                "rag_lease_seconds": 110,
            },
            "RAG_LEASE_SECONDS",
        ),
        (
            {
                # Small values keep AI_LEASE_SECONDS from the QA environment valid.
                "ai_embedding_timeout_seconds": 2,
                "ai_provider_timeout_seconds": 2,
                "rag_query_deadline_seconds": 3,
            },
            "RAG_QUERY_DEADLINE_SECONDS",
        ),
        ({"rag_embedding_batch_size": 257}, "rag_embedding_batch_size"),
    ],
    ids=["needs-ai", "dimensions", "lease-sum", "deadline", "batch-size"],
)
def test_unsafe_rag_settings_are_rejected(values, message):
    with pytest.raises(ValidationError) as error:
        RAGSettings(**{"ai_enabled": True, "rag_enabled": True, **values})
    assert message in str(error.value)


def test_defaults_are_valid():
    settings = build(rag_enabled=True)
    assert settings.rag_lease_seconds > (
        settings.rag_pdf_timeout_seconds + settings.ai_embedding_timeout_seconds + 5
    )
    assert settings.rag_query_deadline_seconds > (
        settings.ai_embedding_timeout_seconds + settings.ai_provider_timeout_seconds
    )
