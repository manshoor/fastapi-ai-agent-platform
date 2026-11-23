"""Server-side claim checks: pure functions, no database or provider."""

import pytest

from app.ai.contracts import Claim
from app.models import CLAIM_FAILURES, RAG_NO_CONTEXT_REASONS
from app.rag.verification import ClaimFailure, check_claims, check_quote, normalize

PASSAGES = [
    "Quiet hours run from 22:00 to 07:00 on weekdays.",
    "A replacement apartment key costs €25. Order it from the management office.",
]


def _claim(statement: str, quote: str, citation: str = "P1") -> Claim:
    return Claim(statement=statement, citation=citation, quote=quote)


def test_normalize_folds_typography_and_whitespace():
    assert normalize("Quiet  Hours – “22:00”­\n") == 'quiet hours - "22:00"'


def test_verbatim_claims_pass():
    claims = [
        _claim("Quiet hours start at 22:00.", "Quiet hours run from 22:00"),
        _claim("A new key costs €25.", "replacement apartment key costs €25", "P2"),
    ]
    assert check_claims(claims, PASSAGES) is None


@pytest.mark.parametrize(
    "quote",
    [
        "“Quiet hours run from 22:00”",
        "...quiet  hours run from 22:00…",
        "QUIET HOURS RUN FROM 22:00",
    ],
    ids=["curly-quotes", "ellipsis-and-spaces", "case"],
)
def test_quotes_match_after_normalizing(quote):
    assert check_claims([_claim("Quiet hours start at 22:00.", quote)], PASSAGES) is None


@pytest.mark.parametrize(
    ("claim", "failure"),
    [
        (_claim("Quiet hours start at 22:00.", "22:00 to"), ClaimFailure.QUOTE_TOO_SHORT),
        (
            _claim("Quiet hours start at 23:00.", "Quiet hours run from 23:00"),
            ClaimFailure.QUOTE_NOT_FOUND,
        ),
        (
            _claim("A key costs €25.", "replacement apartment key costs €25"),
            ClaimFailure.QUOTE_NOT_FOUND,
        ),
        (
            _claim("Quiet hours start at 23:00.", "Quiet hours run from 22:00"),
            ClaimFailure.NUMBER_NOT_IN_PASSAGE,
        ),
        (
            _claim("A key costs €2.", "replacement apartment key costs €25", "P2"),
            ClaimFailure.NUMBER_NOT_IN_PASSAGE,
        ),
    ],
    ids=["two-words", "invented-quote", "wrong-passage", "changed-time", "split-number"],
)
def test_claim_failures_are_named(claim, failure):
    assert check_claims([claim], PASSAGES) is failure


def test_one_failed_claim_fails_the_answer():
    claims = [
        _claim("Quiet hours start at 22:00.", "Quiet hours run from 22:00"),
        _claim("Keys are free.", "keys are free of charge", "P2"),
    ]
    assert check_claims(claims, PASSAGES) is ClaimFailure.QUOTE_NOT_FOUND


def test_a_label_outside_the_passages_is_rejected():
    with pytest.raises(ValueError):
        check_claims([_claim("x y z", "Quiet hours run", "P3")], PASSAGES)


def test_failure_codes_match_the_database_checks():
    assert tuple(ClaimFailure) == CLAIM_FAILURES
    claim_checks = {str(failure) for failure in ClaimFailure} - {"tool_claim_ungrounded"}
    assert claim_checks <= set(RAG_NO_CONTEXT_REASONS)


def _verdict(passage: str, statement: str, quote: str) -> ClaimFailure | None:
    return check_claims([_claim(statement, quote)], [passage])


@pytest.mark.parametrize(
    ("passage", "statement", "quote", "failure"),
    [
        # A quote must start and end on word edges: no word or number is cut in half.
        (
            "Subletting is illegal without written consent.",
            "Subletting is legal with consent.",
            "legal without written consent",
            ClaimFailure.QUOTE_NOT_FOUND,
        ),
        (
            "Pets are disallowed in the building.",
            "Pets are allowed.",
            "allowed in the building",
            ClaimFailure.QUOTE_NOT_FOUND,
        ),
        (
            "A replacement key costs €25 at the office.",
            "A key costs two euros.",
            "key costs €2",
            ClaimFailure.QUOTE_NOT_FOUND,
        ),
        # A space-grouped number is one number, not two.
        (
            "The deposit is 1 200 € per flat.",
            "The deposit is 200 €.",
            "The deposit is 1 200 €",
            ClaimFailure.NUMBER_NOT_IN_PASSAGE,
        ),
        ("The deposit is 1 200 € per flat.", "It is 1200 €.", "The deposit is 1 200 €", None),
        ("The deposit is 1 200 € per flat.", "It is 1 200 €.", "The deposit is 1 200", None),
        # Markdown emphasis, hyphen variants and zero-width characters are not text.
        (
            "**Pets** need *written* approval.",
            "Pets need approval.",
            "Pets need written approval",
            None,
        ),
        ("A two‑year lease is standard.", "Leases run two years.", "A two-year lease is", None),
        (
            "Lifts​ are serviced monthly.",
            "Lifts get a monthly service.",
            "Lifts are serviced monthly",
            None,
        ),
        # A year inside a date may be stated alone.
        (
            "Rules valid from 01.03.2025 onwards.",
            "The rules start in 2025.",
            "Rules valid from 01.03.2025",
            None,
        ),
    ],
    ids=[
        "illegal",
        "disallowed",
        "cut-number",
        "grouped-part",
        "grouped-joined",
        "narrow-space",
        "markdown",
        "non-breaking-hyphen",
        "zero-width",
        "date-year",
    ],
)
def test_review_edge_cases(passage, statement, quote, failure):
    assert _verdict(passage, statement, quote) is failure


BOOKINGS_FACT = "Upcoming bookings: Gym on 2026-11-21 from 18:00 to 19:00 UTC."


@pytest.mark.parametrize(
    ("statement", "quote", "failure"),
    [
        ("Gym on 2026-11-21 from 18:00 to 19:00 UTC.", "Gym on 2026-11-21", None),
        ("Your gym session is on 21 November at 18:00.", "Upcoming bookings: Gym", None),
        # A time the fact does not write is refused, even when it means the same.
        ("Your gym session starts at 6 pm.", "Upcoming bookings: Gym", "number_not_in_passage"),
        ("No bookings.", "No upcoming bookings", "quote_not_found"),
    ],
)
def test_tool_fact_claims_are_checked_like_passages(statement, quote, failure):
    assert check_quote(statement, quote, BOOKINGS_FACT) == failure


def test_three_word_fact_can_be_quoted_whole():
    assert (
        check_quote("No upcoming bookings.", "No upcoming bookings", "No upcoming bookings.")
        is None
    )
