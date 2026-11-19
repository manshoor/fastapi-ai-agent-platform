"""Opt-in evaluation of RAG answers over real HTTP, through the hidden diagnostics route.

    check the route is on -> register one account -> upload the fixture documents
      -> wait until they are ready
      -> per configuration and repeat: ask every golden case -> score -> summarize

The route returns the passages sent to the model, so retrieval and answers are scored
apart (see tests/support/rag_evals.py). The API must run with RAG_DIAGNOSTICS_ENABLED.
Every run is treated as paid: --allow-paid-requests is required. --judge grades cited
answers with an LLM judge, after the judge passes known-verdict controls; it calls the
provider directly and needs OPENAI_API_KEY. Against the provider simulator the run
exits 1 by design: the simulator echoes the first passage, so it checks wiring only.

A failed query, a transport error or a failed verdict is recorded on its run and the
evaluation goes on, so a late failure never discards finished results. Setup failures
(route off, account, upload, ingestion) stop the evaluation before any query.

Example (isolated AI stack on 8002, after explicit authorization):
  uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --repeat 3 \
    --judge --output artifacts/rag-baseline.json

Compare answer-context settings in one run (each --config is a name and its overrides):
  ... --config base --config wide=neighbor_chunks:2 --config ranked=rerank:llm
  ... --config checked=verifier:on --config thinking=reasoning_effort:low

Exit 0 when every run passes, 1 for failed runs, 2 for invalid configuration, failed
judge controls or failed setup. Prints only case IDs, check names, rates and fixed
codes: never questions, answers, document text or credentials.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS = "/api/v1/rag/query/diagnostics"
DOCUMENTS = "/api/v1/rag/documents"
SETTLED = frozenset({"ready", "failed"})
# The isolated AI API on this machine only: never a Compose or remote host name.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_ACCEPTED = 202
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
# A 429 is refused before admission, so waiting and asking again costs nothing.
RETRY_LIMIT = 30
DEFAULT_RETRY_SECONDS = 5
MAX_RETRY_SECONDS = 120
RERANK_MODES = frozenset({"off", "llm"})
REASONING_EFFORTS = frozenset({"none", "low", "medium"})
VERIFIER_MODES = frozenset({"off", "on"})


def _one_of(values: frozenset[str]) -> Callable[[str], str]:
    def parse(value: str) -> str:
        if value not in values:
            raise ValueError("Unknown override value")
        return value

    return parse


# Server settings a configuration may override (the diagnostics route's `overrides`).
OVERRIDES: dict[str, Callable[[str], object]] = {
    "neighbor_chunks": int,
    "relevance_margin": float,
    "rerank": _one_of(RERANK_MODES),
    "reasoning_effort": _one_of(REASONING_EFFORTS),
    "verifier": _one_of(VERIFIER_MODES),
}
_NAME = re.compile(r"[a-z0-9-]{1,32}\Z")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke_ai = _load("smoke_ai", "scripts/smoke_ai.py")
_evals = _load("rag_evals", "tests/support/rag_evals.py")
expect = _smoke_ai.expect


class SetupError(Exception):
    """The evaluation cannot start; the message is a fixed, content-free sentence."""


def parse_config(value: str) -> tuple[str, dict]:
    """`name` or `name=key:value,...`, where every key is a known override."""
    name, _, pairs = value.partition("=")
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid configuration name")
    overrides: dict[str, object] = {}
    for pair in filter(None, pairs.split(",")):
        key, _, raw = pair.partition(":")
        if key not in OVERRIDES or key in overrides or not raw:
            raise ValueError("Unknown or repeated override")
        overrides[key] = OVERRIDES[key](raw)
    return name, overrides


def output_path(value: str) -> Path:
    """A file that does not exist yet, directly under the private artifacts directory.

    Checked before any request, so a name clash never discards a finished evaluation.
    """
    artifacts = ROOT / "artifacts"
    path = Path(value).resolve()
    if artifacts.is_symlink() or not artifacts.is_dir() or path.parent != artifacts.resolve():
        raise ValueError("Output must be directly under artifacts/")
    if path.exists() or path.is_symlink():
        raise ValueError("Output file already exists")
    return path


def validate_loopback(base_url: str) -> None:
    _smoke_ai.validate_destination(base_url)
    if urlsplit(base_url).hostname not in LOOPBACK_HOSTS:
        raise ValueError("Evaluation runs against a loopback API only")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--allow-paid-requests", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not args.allow_paid_requests:
        parser.error("Live evaluation requires --allow-paid-requests")
    if args.judge and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--judge requires OPENAI_API_KEY")
    if args.judge_model and not args.judge:
        parser.error("--judge-model requires --judge")
    if not 1 <= args.repeat <= 5:
        parser.error("repeat must be 1..5")
    if not 1 <= args.top_k <= 8:
        parser.error("top-k must be 1..8")
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be 1..600 seconds")
    try:
        validate_loopback(args.base_url)
    except ValueError:
        parser.error("unsafe destination; select the isolated AI API on loopback")
    try:
        args.configs = [parse_config(value) for value in args.config] or [("server", {})]
    except ValueError:
        parser.error("invalid --config; use name or name=key:value,...")
    if len({name for name, _ in args.configs}) != len(args.configs):
        parser.error("configuration names must be unique")
    try:
        args.output = output_path(args.output) if args.output else None
    except ValueError:
        parser.error("--output must name a new file directly under artifacts/")
    return args


class Account:
    """One synthetic account. It logs in again when its token expires during a long run."""

    def __init__(self, client: httpx.Client):
        self.client = client
        self._email = f"rag-eval-{uuid4().hex}@example.com"
        self._password = f"synthetic-{uuid4().hex}!"
        credentials = {"email": self._email, "password": self._password}
        expect(client.post("/api/v1/auth/register", json=credentials), HTTP_CREATED)
        self.headers = self._login()

    def _login(self) -> dict:
        form = {"username": self._email, "password": self._password}
        login = expect(self.client.post("/api/v1/auth/login", data=form), HTTP_OK).json()
        return {"Authorization": f"Bearer {login['access_token']}"}

    def refresh(self) -> None:
        self.headers = self._login()


def check_route(client: httpx.Client) -> None:
    """Before any setup: 401 means the route is on; 404 means the flag is off."""
    probe = client.post(DIAGNOSTICS, json={"question": "probe"})
    if probe.status_code == HTTP_NOT_FOUND:
        raise SetupError("The diagnostics route is off (set RAG_DIAGNOSTICS_ENABLED)")
    if probe.status_code != HTTP_UNAUTHORIZED:
        raise SetupError("The diagnostics route did not answer as expected")


def upload_documents(account: Account, timeout: float) -> None:
    # One upload at a time: the API admits one upload per owner.
    client = account.client
    ids = []
    for document in _evals.load_documents():
        response = client.post(
            DOCUMENTS,
            headers={**account.headers, "Idempotency-Key": f"rag-eval-{uuid4().hex}"},
            data={"title": document["title"]},
            files={"file": (document["name"], document["text"].encode())},
        )
        ids.append(expect(response, HTTP_ACCEPTED).json()["id"])

    # Wait for every document to settle; a failed ingestion stops the evaluation.
    deadline = time.monotonic() + timeout
    for document_id in ids:
        while True:
            response = client.get(f"{DOCUMENTS}/{document_id}", headers=account.headers)
            status = expect(response, HTTP_OK).json()["status"]
            if status == "ready":
                break
            if status in SETTLED:
                raise SetupError("Fixture ingestion failed")
            if time.monotonic() >= deadline:
                raise SetupError("Fixture ingestion deadline exceeded")
            time.sleep(0.5)


def _retry_after(response: httpx.Response) -> float:
    try:
        seconds = int(response.headers.get("Retry-After", DEFAULT_RETRY_SECONDS))
    except ValueError:
        seconds = DEFAULT_RETRY_SECONDS
    return min(max(seconds, 1), MAX_RETRY_SECONDS)


def ask(account: Account, request: dict) -> tuple[dict | None, str | None, float]:
    """One diagnostics query: (body, None, seconds) or (None, fixed error code, seconds).

    A 429 waits and asks again; a 401 logs in again once. Neither reached the model.
    """
    refreshed = False
    latency = 0.0
    for _ in range(RETRY_LIMIT):
        started = time.monotonic()
        try:
            response = account.client.post(DIAGNOSTICS, headers=account.headers, json=request)
        except httpx.TimeoutException:
            return None, "transport_timeout", time.monotonic() - started
        except httpx.HTTPError:
            return None, "transport_error", time.monotonic() - started
        latency = time.monotonic() - started
        if response.status_code == HTTP_TOO_MANY_REQUESTS:
            time.sleep(_retry_after(response))
            continue
        if response.status_code == HTTP_UNAUTHORIZED and not refreshed:
            refreshed = True
            try:
                account.refresh()
            except (httpx.HTTPError, AssertionError):
                return None, "login_failed", latency
            continue
        if response.status_code != HTTP_OK:
            return None, f"http_{response.status_code}", latency
        return response.json(), None, latency
    return None, "rate_limited", latency


async def _verdict(settings, question: str, answer: str, passages: list[str], model) -> dict:
    from app.ai.provider import OpenAIJudge

    # One client per event loop: each verdict runs in its own asyncio.run.
    judge = OpenAIJudge(settings)
    try:
        verdict = await judge.judge(question, answer, passages, model=model)
    finally:
        await judge.aclose()
    return {"grounded": verdict.grounded, "answers_question": verdict.answers_question}


def controls_pass(settings, model) -> bool:
    """The judge must grade every known-verdict control correctly before its scores count."""
    for control in _evals.load_controls():
        verdict = asyncio.run(
            _verdict(settings, control["question"], control["answer"], control["passages"], model)
        )
        expected = {
            "grounded": control["grounded"],
            "answers_question": control["answers_question"],
        }
        if verdict != expected:
            print(f"{control['id']}: judge control failed", file=sys.stderr)
            return False
    return True


def _tokens(body: dict | None) -> int | None:
    if body is None:
        return None
    usage = body["usage"]
    counts = [usage["input_tokens"], usage["output_tokens"]]
    return sum(count for count in counts if count is not None) if any(counts) else None


def _percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def _summary_line(name: str, summary: dict) -> str:
    rates = " ".join(
        f"{key}={_percent(summary[key])}"
        for key in (
            "error_rate",
            "retrieval_hit",
            "answer_correct",
            "wrong_given_hit",
            "false_refusal",
            "correct_refusal",
            "faithful",
        )
    )
    tokens = summary["mean_tokens"]
    latency = summary["mean_latency"]
    return (
        f"{name}: runs={summary['runs']} {rates}"
        f" mean_tokens={'-' if tokens is None else round(tokens)}"
        f" mean_latency={'-' if latency is None else f'{latency:.1f}s'}"
    )


def _write(path: Path, payload: dict) -> None:
    # Exclusive create with private permissions: never overwrite or follow a link.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def judge_run(judge, case: dict, body: dict | None) -> tuple[dict | None, str | None]:
    """Grade a cited answer: (verdict, None), (None, fixed error code), or (None, None)."""
    if judge is None or body is None or body["status"] != "answered" or not body["citations"]:
        return None, None
    settings, model, failures = judge
    try:
        verdict = asyncio.run(
            _verdict(settings, case["question"], body["answer"], _evals.cited_texts(body), model)
        )
    except failures as exc:
        return None, exc.code
    return verdict, None


def run_case(account: Account, case: dict, args, overrides: dict, judge) -> dict:
    """Ask one case and score it. The record holds codes and scores, never text."""
    request = {"question": case["question"], "top_k": args.top_k}
    if overrides:
        request["overrides"] = overrides
    body, error, latency = ask(account, request)
    verdict, judge_error = judge_run(judge, case, body)
    return {
        "case": case["id"],
        "category": case["category"],
        "expected_status": case["expected_status"],
        "scores": _evals.score(case, body),
        "status": body["status"] if body else None,
        "no_context_reason": body["no_context_reason"] if body else None,
        "error": error,
        "judge": verdict,
        "judge_error": judge_error,
        "tokens": _tokens(body),
        "latency": round(latency, 3),
    }


def passed(run: dict) -> bool:
    """Every check passes, the judge (if any) accepts, and nothing failed on the way."""
    verdict = run["judge"]
    judged = verdict is None or (verdict["grounded"] and verdict["answers_question"])
    clean = run["error"] is None and run.get("judge_error") is None
    return clean and all(run["scores"].values()) and judged


def _line(name: str, repeat: int, run: dict) -> str:
    misses = sorted(check for check, ok in run["scores"].items() if not ok)
    verdict = run["judge"]
    if run.get("judge_error"):
        judged = run["judge_error"]
    elif verdict is None:
        judged = "-"
    else:
        judged = ",".join(key for key, ok in verdict.items() if ok) or "none"
    return (
        f"{name} r{repeat} {run['case']}: {'PASS' if passed(run) else 'FAIL'}"
        f" failed={','.join(misses) or '-'} judge={judged} error={run['error'] or '-'}"
    )


def _setup(client: httpx.Client, timeout: float) -> Account:
    expect(client.get("/health/ready"), HTTP_OK)
    check_route(client)
    account = Account(client)
    upload_documents(account, timeout)
    return account


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = _evals.load_cases()
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            print("Unknown case ID", file=sys.stderr)
            return 2
        cases = [case for case in cases if case["id"] in args.case]

    # The judge must pass its controls before any query is paid for.
    judge = None
    if args.judge:
        from app.ai.config import ProviderSettings
        from app.ai.provider import ProviderFailure

        settings = ProviderSettings()
        judge = (settings, args.judge_model, ProviderFailure)
        try:
            if not controls_pass(settings, args.judge_model):
                return 2
        except ProviderFailure as exc:
            print(f"Judge controls failed: {exc.code}", file=sys.stderr)
            return 2

    # Setup failures stop here; per-query failures are recorded on their runs.
    runs: list[dict] = []
    summaries: dict[str, dict] = {}
    with httpx.Client(base_url=args.base_url, timeout=args.timeout, trust_env=False) as client:
        try:
            account = _setup(client, args.timeout)
        except SetupError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except (httpx.HTTPError, AssertionError):
            print("Evaluation setup failed (details withheld)", file=sys.stderr)
            return 2
        for name, overrides in args.configs:
            config_runs = []
            for repeat in range(1, args.repeat + 1):
                for case in cases:
                    run = run_case(account, case, args, overrides, judge)
                    run.update(config=name, repeat=repeat)
                    config_runs.append(run)
                    print(_line(name, repeat, run))
            summaries[name] = _evals.summarize(config_runs)
            runs.extend(config_runs)

    for name, summary in summaries.items():
        print(_summary_line(name, summary))
    if args.output is not None:
        try:
            _write(args.output, {"summaries": summaries, "runs": runs})
        except OSError:
            print("Output file could not be created", file=sys.stderr)
            return 2
    failed = sum(not passed(run) for run in runs)
    print(f"{len(runs) - failed}/{len(runs)} runs passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
