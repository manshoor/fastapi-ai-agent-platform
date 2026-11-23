"""Server-side checks that every claim is quoted word for word from the text it cites.

    claim (statement, citation P<n>, quote)
      -> quote has at least 3 words                       else quote_too_short
      -> normalized quote is inside the cited passage,
         starting and ending on word edges                else quote_not_found
      -> every number in the statement is in the passage  else number_not_in_passage

All-or-nothing: the first failed claim fails the whole answer, because dropping one
claim can change the meaning of the rest (for example, by losing an exception).
The passage text is exactly the text the model received. The assistant checks its
tool claims the same way, against the server-written fact of the cited tool result
(`check_quote`). Pure functions only.
"""

import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum

from app.ai.contracts import Claim

MIN_QUOTE_WORDS = 3
_WHITESPACE = re.compile(r"\s+")
# Numbers as written: 25, 2.50, 1,200, 22:00, and space-grouped 1 200 (one number).
_NUMBER = re.compile(r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,:]\d+)*")
_SEPARATOR = re.compile(r"[.,:]")
# Characters that differ between a passage and a faithful quote of it: typographic
# quotes, hyphens and dashes become ASCII; soft hyphens, zero-width characters and
# Markdown emphasis or code marks are dropped.
_ASCII = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": None,
        "\u200b": None,
        "\u2060": None,
        "\ufeff": None,
        "*": None,
        "`": None,
    }
)
# Wrapping a model may add around a quote: quote marks and ellipses ("..." after NFKC).
_QUOTE_EDGES = re.compile(r"""^(?:["'\s]|\.\.\.)+|(?:["'\s]|\.\.\.)+$""")


class ClaimFailure(StrEnum):
    """Fixed, content-free reasons an answer is refused. Stored on usage and turn rows."""

    QUOTE_TOO_SHORT = "quote_too_short"
    QUOTE_NOT_FOUND = "quote_not_found"
    NUMBER_NOT_IN_PASSAGE = "number_not_in_passage"
    TOOL_CLAIM_UNGROUNDED = "tool_claim_ungrounded"


def normalize(text: str) -> str:
    """NFKC, casefolded, ASCII quotes and dashes, no soft hyphens, single spaces."""
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_ASCII)
    return _WHITESPACE.sub(" ", folded).strip()


def _quote(text: str) -> str:
    return _QUOTE_EDGES.sub("", normalize(text))


def _quoted_in(quote: str, passage: str) -> bool:
    """Substring match that never cuts a word or number: "legal" is not in "illegal"."""
    pattern = re.escape(quote)
    if quote[:1].isalnum():
        pattern = r"(?<!\w)" + pattern
    if quote[-1:].isalnum():
        pattern += r"(?!\w)"
    return re.search(pattern, passage) is not None


def appears_in(phrase: str, text: str) -> bool:
    """`phrase` occurs in `text` after normalizing both, starting and ending on word edges."""
    quoted = _quote(phrase)
    return bool(quoted) and _quoted_in(quoted, normalize(text))


# A quote never starts or ends inside a number: "1" is not in "1,200", "19" not in "19:00".
_NOT_AFTER_NUMBER = r"(?<!\d[.,:])"
_NOT_BEFORE_NUMBER = r"(?![.,:]\d)"


def whole_quote(quote: str, text: str) -> bool:
    """`quote` is at least three words of `text`, cutting no word and no number."""
    quoted = _quote(quote)
    if len(quoted.split()) < MIN_QUOTE_WORDS:
        return False
    pattern = re.escape(quoted)
    if quoted[:1].isalnum():
        pattern = r"(?<!\w)" + _NOT_AFTER_NUMBER + pattern
    if quoted[-1:].isalnum():
        pattern += r"(?!\w)" + _NOT_BEFORE_NUMBER
    return re.search(pattern, normalize(text)) is not None


def _numbers(text: str) -> set[str]:
    """Numbers in normalized text. A date's parts count too: 01.03.2025 allows 2025."""
    numbers = set()
    for match in _NUMBER.findall(text):
        number = match.replace(" ", "")
        numbers.add(number)
        parts = _SEPARATOR.split(number)
        if len(parts) > 2:
            numbers.update(parts)
    return numbers


def check_quote(statement: str, quote: str, text: str) -> ClaimFailure | None:
    """One claim against the one text it cites: a passage, or a tool fact."""
    source = normalize(text)
    quoted = _quote(quote)

    # The quote must be a real, contiguous span of the cited text.
    if len(quoted.split()) < MIN_QUOTE_WORDS:
        return ClaimFailure.QUOTE_TOO_SHORT
    if not _quoted_in(quoted, source):
        return ClaimFailure.QUOTE_NOT_FOUND

    # Numbers are where paraphrase goes wrong most: each must appear as written.
    stated = {match.replace(" ", "") for match in _NUMBER.findall(normalize(statement))}
    if not stated <= _numbers(source):
        return ClaimFailure.NUMBER_NOT_IN_PASSAGE
    return None


def _check(claim: Claim, passages: Sequence[str]) -> ClaimFailure | None:
    index = int(claim.citation[1:]) - 1
    if not 0 <= index < len(passages):
        raise ValueError("Claim cites a passage that was not supplied")
    return check_quote(claim.statement, claim.quote, passages[index])


def check_claims(claims: Sequence[Claim], passages: Sequence[str]) -> ClaimFailure | None:
    """None when every claim checks out; otherwise the first claim's failure."""
    for claim in claims:
        failure = _check(claim, passages)
        if failure is not None:
            return failure
    return None
