"""RAG golden set: data integrity and scoring. No provider calls; live runs are opt-in.

`scripts/evaluate_rag.py` asks these cases through the hidden diagnostics route and scores
retrieval apart from the answer. CI cannot measure answer quality (the simulator echoes
the first passage), so it gates the data, the scorer and the script's safety checks.
"""

import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.rag.chunking import chunk_document
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType
from app.rag.parsing import ParsedDocument
from tests.support.rag_evals import (
    CHECKS,
    cited_texts,
    contains,
    load_cases,
    load_controls,
    load_documents,
    normalize,
    score,
    summarize,
)

CASES = load_cases()
DOCUMENTS = load_documents()
BY_ID = {case["id"]: case for case in CASES}
EXPECTED_CATEGORIES = {
    "single": 6,
    "numeric": 5,
    "condition": 4,
    "multi": 3,
    "heading": 3,
    "split": 3,
    "distractor": 3,
    "not_covered": 3,
    "injection": 2,
}


def _holders(item: str) -> list[dict]:
    return [document for document in DOCUMENTS if contains(document["text"], item)]


def _chunks(document: dict) -> list[str]:
    """Chunk texts under the default chunk settings."""
    fields = RAGSettings.model_fields
    settings = SimpleNamespace(
        rag_chunk_size=fields["rag_chunk_size"].default,
        rag_chunk_overlap=fields["rag_chunk_overlap"].default,
        rag_max_chunks=fields["rag_max_chunks"].default,
    )
    parsed = ParsedDocument([(None, document["text"])], None)
    chunks = chunk_document(parsed, ContentType.MARKDOWN, settings)
    return [chunk.content for chunk in chunks]


def _cases(category: str) -> list[dict]:
    return [case for case in CASES if case["category"] == category]


def test_dataset_covers_every_category():
    assert Counter(case["category"] for case in CASES) == EXPECTED_CATEGORIES
    # Five documents fit the per-owner limit of unsettled uploads.
    assert len(DOCUMENTS) == 5


def test_evidence_is_in_exactly_one_document():
    for case in CASES:
        for evidence in case["evidence"]:
            assert len(_holders(evidence)) == 1, case["id"]


def test_facts_are_stated_by_the_evidence_documents():
    for case in CASES:
        text = "\n".join(_holders(item)[0]["text"] for item in case["evidence"])
        for group in case["facts"]:
            assert any(contains(text, item) for item in group), case["id"]


def test_distractor_answers_come_from_another_document():
    for case in _cases("distractor"):
        source = _holders(case["evidence"][0])[0]["name"]
        holders = {document["name"] for item in case["forbidden"] for document in _holders(item)}
        assert holders and source not in holders, case["id"]


def test_split_evidence_needs_two_adjacent_chunks():
    for case in _cases("split"):
        (name,) = {_holders(item)[0]["name"] for item in case["evidence"]}
        chunks = _chunks(next(document for document in DOCUMENTS if document["name"] == name))
        evidence = case["evidence"]
        assert not any(all(contains(chunk, item) for item in evidence) for chunk in chunks), case[
            "id"
        ]
        assert any(
            all(contains(first, item) or contains(second, item) for item in evidence)
            for first, second in zip(chunks, chunks[1:], strict=False)
        ), case["id"]


def test_heading_evidence_chunks_lack_their_heading():
    for case in _cases("heading"):
        (evidence,) = case["evidence"]
        (document,) = _holders(evidence)
        pattern = rf"^#{{2,6}} {re.escape(case['heading'])}$"
        assert re.search(pattern, document["text"], re.MULTILINE), case["id"]
        holding = [chunk for chunk in _chunks(document) if contains(chunk, evidence)]
        assert holding, case["id"]
        assert not any(contains(chunk, case["heading"]) for chunk in holding), case["id"]


def test_judge_controls_cover_both_failure_kinds():
    controls = load_controls()
    assert len(controls) == 6
    assert sum(not control["grounded"] for control in controls) == 3
    assert sum(not control["answers_question"] for control in controls) == 1


def _body(*, answer: str, texts: list[str], cited: list[int], status: str = "answered") -> dict:
    """A diagnostics body: passages from texts, citations by passage index."""
    passages = [
        {"document_id": "doc", "ordinal": index, "text": text} for index, text in enumerate(texts)
    ]
    citations = [{"document_id": "doc", "ordinal": index} for index in cited]
    return {"status": status, "answer": answer, "passages": passages, "citations": citations}


KEY = "Keys. A replacement apartment key costs €25. Order it from the management office."
OTHER = "Glass is collected on Mondays."


def test_correct_answer_passes_every_check():
    body = _body(answer="A replacement key costs  €25.", texts=[OTHER, KEY], cited=[1])
    assert score(BY_ID["numeric-key-price"], body) == dict.fromkeys(CHECKS, True)
    assert cited_texts(body) == [KEY]


@pytest.mark.parametrize(
    ("case_id", "answer", "texts", "cited", "failed"),
    [
        ("numeric-key-price", "A replacement key costs €35.", [KEY], [0], {"facts"}),
        ("numeric-key-price", "It costs €25.", [OTHER], [0], {"retrieval", "citations"}),
        ("numeric-key-price", "It costs €25.", [OTHER, KEY], [0], {"citations"}),
        (
            "distractor-visitor-price",
            "Visitor parking costs €2 per hour, or €3 per hour.",
            ["Visitor parking costs €2 per hour, paid at the machine."],
            [0],
            {"forbidden"},
        ),
    ],
    ids=["wrong-number", "retrieval-miss", "cited-wrong-passage", "distractor-number"],
)
def test_scorer_names_the_failed_check(case_id, answer, texts, cited, failed):
    scores = score(BY_ID[case_id], _body(answer=answer, texts=texts, cited=cited))
    assert {name for name, ok in scores.items() if not ok} == failed


def test_not_covered_cases_pass_only_a_clean_refusal():
    case = BY_ID["not-covered-pool"]
    refusal = _body(answer="Nothing found.", texts=[OTHER], cited=[], status="no_context")
    assert score(case, refusal) == dict.fromkeys(CHECKS, True)
    invented = _body(answer="The pool opens at 07:00.", texts=[OTHER], cited=[0])
    assert {name for name, ok in score(case, invented).items() if not ok} == {
        "citations",
        "status",
    }


def test_failed_request_fails_every_check():
    assert score(BY_ID["single-bikes"], None) == dict.fromkeys(CHECKS, False)


def test_normalize_matches_typographic_variants():
    assert normalize("Quiet  hours – “22:00”\n") == 'quiet hours - "22:00"'


@pytest.mark.parametrize(
    ("text", "item", "found"),
    [
        ("It costs €25.", "€2", False),
        ("It costs €2.50.", "€2", False),
        ("Book 24 hours ahead.", "4 hours", False),
        ("It costs €2 per hour.", "€2", True),
        ("Glass goes out on Mondays.", "monday", True),
        ("It costs €25.00.", "€25", True),
        ("It costs 25 €.", "€25", True),
        ("It costs € 25.", "€25", True),
        ("Events end at 22:30.", "30", False),
        ("Events end at 22:30.", "22", False),
        ("Pay within fourteen days.", "four", False),
    ],
)
def test_matching_never_splits_a_number(text, item, found):
    assert contains(text, item) is found


def _run(expected: str, *, error=None, status=None, judge=None, **scores) -> dict:
    checks = dict.fromkeys(CHECKS, True) | scores
    return {
        "expected_status": expected,
        "scores": checks,
        "status": status,
        "error": error,
        "judge": judge,
        "tokens": 100,
        "latency": 1.0,
    }


def test_summary_separates_retrieval_answers_refusals_and_errors():
    grounded = {"grounded": True, "answers_question": True}
    ungrounded = {"grounded": False, "answers_question": True}
    runs = [
        _run("answered", status="answered", judge=grounded),
        _run("answered", status="answered", judge=ungrounded, facts=False),
        _run("answered", status="no_context", retrieval=False),
        _run("no_context", status="no_context"),
        _run("answered", error="http_502"),
    ]
    runs[2]["scores"]["status"] = False
    summary = summarize(runs)
    assert summary["runs"] == 5
    assert summary["error_rate"] == pytest.approx(1 / 5)
    assert summary["retrieval_hit"] == pytest.approx(2 / 3)
    assert summary["answer_correct"] == pytest.approx(1 / 3)
    assert summary["wrong_given_hit"] == pytest.approx(1 / 2)
    assert summary["false_refusal"] == pytest.approx(1 / 3)
    assert summary["correct_refusal"] == 1
    assert summary["faithful"] == pytest.approx(1 / 2)


def _script():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_rag.py"
    spec = importlib.util.spec_from_file_location("evaluate_rag", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--allow-paid-requests", "--base-url", "http://127.0.0.1:8000"],
        ["--allow-paid-requests", "--base-url", "https://example.com"],
        ["--allow-paid-requests", "--base-url", "http://api:8000"],
        ["--allow-paid-requests", "--judge-model", "m"],
        ["--allow-paid-requests", "--repeat", "6"],
        ["--allow-paid-requests", "--top-k", "9"],
        ["--allow-paid-requests", "--config", "Bad Name"],
        ["--allow-paid-requests", "--config", "fast=unknown:1"],
        ["--allow-paid-requests", "--config", "a", "--config", "a"],
        ["--allow-paid-requests", "--output", "/tmp/rag-eval.json"],
    ],
    ids=[
        "no-consent",
        "normal-port",
        "remote",
        "compose-host",
        "judge-model-alone",
        "too-many-repeats",
        "top-k-limit",
        "bad-config-name",
        "unknown-override",
        "duplicate-config",
        "output-outside-artifacts",
    ],
)
def test_evaluation_refuses_unsafe_or_invalid_runs(argv, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        _script().parse_args(argv)
    assert exit_info.value.code == 2


def test_default_run_uses_server_settings():
    args = _script().parse_args(["--allow-paid-requests"])
    assert args.configs == [("server", {})]
    assert args.output is None


@pytest.mark.parametrize(
    ("run", "ok"),
    [
        ({"error": None, "scores": dict.fromkeys(CHECKS, True), "judge": None}, True),
        (
            {
                "error": None,
                "scores": dict.fromkeys(CHECKS, True),
                "judge": {"grounded": False, "answers_question": True},
            },
            False,
        ),
        ({"error": "http_504", "scores": dict.fromkeys(CHECKS, False), "judge": None}, False),
        (
            {
                "error": None,
                "scores": dict.fromkeys(CHECKS, True),
                "judge": None,
                "judge_error": "provider_timeout",
            },
            False,
        ),
    ],
    ids=["checks-pass", "judge-rejects", "error", "judge-error"],
)
def test_a_run_passes_only_with_every_check_and_verdict(run, ok):
    assert _script().passed(run) is ok


def test_cases_need_a_question(tmp_path):
    case = dict(BY_ID["single-bikes"], question="  ")
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_cases(path)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """The script's ROOT moved to a temporary tree with a private artifacts directory."""
    script = _script()
    (tmp_path / "artifacts").mkdir(mode=0o700)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    return script, tmp_path / "artifacts"


def test_output_must_be_a_new_file_under_artifacts(artifacts):
    script, directory = artifacts
    assert script.output_path(str(directory / "run.json")) == (directory / "run.json").resolve()
    (directory / "old.json").write_text("{}", encoding="utf-8")
    for value in (directory / "old.json", directory / "../run.json", directory / "x/run.json"):
        with pytest.raises(ValueError):
            script.output_path(str(value))


class _Client:
    """Replays fixed responses for the diagnostics route and the login form."""

    def __init__(self, *statuses: int):
        self.statuses = list(statuses)
        self.logins = 0

    def post(self, path, **kwargs):
        request = httpx.Request("POST", f"http://127.0.0.1:8002{path}")
        if path.endswith("/login"):
            self.logins += 1
            return httpx.Response(200, json={"access_token": "new"}, request=request)
        status = self.statuses.pop(0)
        if status == -1:
            raise httpx.ConnectError("refused", request=request)
        body = {"status": "answered"} if status == 200 else {}
        return httpx.Response(status, json=body, request=request)


def _account(script, client):
    account = object.__new__(script.Account)
    account.client = client
    account._email = "eval@example.com"
    account._password = "synthetic"
    account.headers = {"Authorization": "Bearer old"}
    return account


def test_expired_token_logs_in_again_once():
    script = _script()
    client = _Client(401, 200)
    account = _account(script, client)
    body, error, _ = script.ask(account, {"question": "q"})
    assert (body, error, client.logins) == ({"status": "answered"}, None, 1)
    assert account.headers == {"Authorization": "Bearer new"}
    client = _Client(401, 401)
    assert script.ask(_account(script, client), {"question": "q"})[:2] == (None, "http_401")


def test_transport_failures_are_recorded_not_raised():
    script = _script()
    assert script.ask(_account(script, _Client(-1)), {"question": "q"})[:2] == (
        None,
        "transport_error",
    )


@pytest.mark.parametrize(("status", "message"), [(404, "is off"), (500, "did not answer")])
def test_setup_stops_when_the_route_is_not_reachable(status, message):
    script = _script()
    with pytest.raises(script.SetupError, match=message):
        script.check_route(_Client(status))
    script.check_route(_Client(401))
