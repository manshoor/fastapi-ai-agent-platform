"""Golden set and scoring for RAG answers, shared by tests and scripts/evaluate_rag.py.

Each case is scored on the diagnostics view of one query, so retrieval and answer quality
are measured apart:

    retrieval  every evidence string is in some passage sent to the model
    citations  every evidence string is in some cited passage
    status     answered or no_context, as expected
    facts      every fact group has one alternative in the answer
    forbidden  no forbidden string is in the answer

Matching is on normalized text (NFKC, casefolded, ASCII quotes and dashes, single spaces,
euro sign before the amount) and never splits a number or a word: "€2" does not match "€25"
or "€2.50", "30" does not match "22:30", and "four" does not match "fourteen".
"""

import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/rag_v1.jsonl"
DOCUMENTS = ROOT / "evals/rag_documents"
CONTROLS = ROOT / "evals/rag_judge_controls.jsonl"
CHECKS = ("retrieval", "citations", "status", "facts", "forbidden")
CATEGORIES = frozenset(
    {
        "single",
        "numeric",
        "condition",
        "multi",
        "heading",
        "split",
        "distractor",
        "not_covered",
        "injection",
        # Two facts in two documents, linked by one entity: graph expansion's target.
        "multi_hop",
    }
)
STATUSES = frozenset({"answered", "no_context"})
_CASE_ID = re.compile(r"[a-z0-9-]{1,64}\Z")
_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
# "€ 25", "25 €" and "25€" all become "€25", the form the documents use.
_EURO_AFTER = re.compile(r"(?<![\d€.,])(\d+(?:[.,]\d+)*) ?€")
_EURO_SPACE = re.compile(r"€ (?=\d)")
# Typographic characters a model may emit for the ASCII ones in the documents.
_ASCII = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "­": None,
    }
)


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_ASCII)
    spaced = _WHITESPACE.sub(" ", folded).strip()
    return _EURO_SPACE.sub("€", _EURO_AFTER.sub(r"€\1", spaced))


def contains(text: str, item: str) -> bool:
    """Normalized substring match that never splits a number or a word.

    A number may end in zero decimals ("€25" matches "€25.00") but not in other digits
    ("€2" does not match "€2.50"). A word may take a plural ("monday" matches "Mondays").
    """
    needle = normalize(item)
    pattern = re.escape(needle)

    # Leading edge: no word character before; for a number, no "2." or "22:" either.
    if needle[:1].isdigit():
        pattern = r"(?<!\w)(?<!\d[.,:])" + pattern
    elif needle[:1].isalpha():
        pattern = r"(?<!\w)" + pattern

    # Trailing edge: a number may not go on; a word may only take a plural ending.
    if needle[-1:].isdigit():
        pattern += r"(?!\d)(?![.,]0*[1-9])(?!:\d)"
    elif needle[-1:].isalpha():
        pattern += r"(?:s|es)?(?!\w)"
    return re.search(pattern, normalize(text)) is not None


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def load_cases(path: Path = DATASET) -> list[dict]:
    """Golden cases. Answerable cases need evidence and facts; not-covered cases have none."""
    cases = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if not _CASE_ID.fullmatch(case["id"]) or case["id"] in seen:
            raise ValueError("Invalid or duplicate case ID")
        seen.add(case["id"])
        if case["category"] not in CATEGORIES or case["expected_status"] not in STATUSES:
            raise ValueError("Invalid case category or status")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError("Every case needs a question")
        groups = case["facts"]
        if not (
            _strings(case["evidence"])
            and isinstance(groups, list)
            and all(_strings(group) and group for group in groups)
            and _strings(case["forbidden"])
        ):
            raise ValueError("Invalid case expectations")
        answerable = case["expected_status"] == "answered"
        if answerable != bool(case["evidence"]) or answerable != bool(groups):
            raise ValueError("Only answerable cases carry evidence and facts")
        if (case["category"] == "heading") != isinstance(case.get("heading"), str):
            raise ValueError("Heading cases, and only they, name their section")
        cases.append(case)
    return cases


def load_documents(directory: Path = DOCUMENTS) -> list[dict]:
    """Each fixture document: file name, title from its first heading, and full text."""
    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({"name": path.name, "title": _TITLE.search(text).group(1), "text": text})
    return documents


def load_controls(path: Path = CONTROLS) -> list[dict]:
    """Judge controls: answers with known verdicts, so a judge that always passes is caught."""
    controls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        control = json.loads(line)
        if not (
            _CASE_ID.fullmatch(control["id"])
            and isinstance(control["question"], str)
            and isinstance(control["answer"], str)
            and _strings(control["passages"])
            and isinstance(control["grounded"], bool)
            and isinstance(control["answers_question"], bool)
        ):
            raise ValueError("Invalid judge control")
        controls.append(control)
    return controls


def cited_texts(body: dict) -> list[str]:
    """Texts of the passages the answer cites, matched by document and chunk ordinal."""
    by_chunk = {
        (passage["document_id"], passage["ordinal"]): passage["text"]
        for passage in body["passages"]
    }
    return [
        by_chunk[key]
        for citation in body["citations"]
        if (key := (citation["document_id"], citation["ordinal"])) in by_chunk
    ]


def _all_found(evidence: list[str], texts: list[str]) -> bool:
    return all(any(contains(text, item) for text in texts) for item in evidence)


def score(case: dict, body: dict | None) -> dict[str, bool]:
    """Checks for one diagnostics response; a failed request (None) fails every check."""
    if body is None:
        return dict.fromkeys(CHECKS, False)
    answer = body["answer"]
    if case["expected_status"] != "answered":
        # Nothing to retrieve or state; only a refusal without citations passes.
        return {
            "retrieval": True,
            "citations": not body["citations"],
            "status": body["status"] == "no_context",
            "facts": True,
            "forbidden": not any(contains(answer, item) for item in case["forbidden"]),
        }
    return {
        "retrieval": _all_found(case["evidence"], [p["text"] for p in body["passages"]]),
        "citations": _all_found(case["evidence"], cited_texts(body)),
        "status": body["status"] == "answered",
        "facts": all(any(contains(answer, item) for item in group) for group in case["facts"]),
        "forbidden": not any(contains(answer, item) for item in case["forbidden"]),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def summarize(runs: list[dict]) -> dict[str, float | int | None]:
    """Rates over runs of {expected_status, scores, status, error, judge, tokens, latency}.

    `wrong_given_hit` is the problem this set measures: retrieval found the evidence,
    yet the answer is wrong. Errors (5xx, timeouts) are counted apart from wrong answers.
    """
    finished = [run for run in runs if run["error"] is None]
    answerable = [run for run in finished if run["expected_status"] == "answered"]
    not_covered = [run for run in finished if run["expected_status"] == "no_context"]
    hits = [run for run in answerable if run["scores"]["retrieval"]]

    def correct(run: dict) -> bool:
        scores = run["scores"]
        return scores["status"] and scores["facts"] and scores["forbidden"]

    judged = [run["judge"] for run in finished if run["judge"] is not None]
    return {
        "runs": len(runs),
        "error_rate": _rate(len(runs) - len(finished), len(runs)),
        "retrieval_hit": _rate(len(hits), len(answerable)),
        "answer_correct": _rate(sum(correct(run) for run in answerable), len(answerable)),
        "wrong_given_hit": _rate(
            sum(run["status"] == "answered" and not correct(run) for run in hits), len(hits)
        ),
        "false_refusal": _rate(
            sum(run["status"] == "no_context" for run in answerable), len(answerable)
        ),
        "correct_refusal": _rate(
            sum(run["status"] == "no_context" for run in not_covered), len(not_covered)
        ),
        "faithful": _rate(sum(verdict["grounded"] for verdict in judged), len(judged)),
        "mean_tokens": _mean([run["tokens"] for run in finished if run["tokens"] is not None]),
        "mean_latency": _mean([run["latency"] for run in finished]),
    }
