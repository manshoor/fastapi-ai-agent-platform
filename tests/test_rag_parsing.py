"""PDF and text parsing, including the limited child process."""

import asyncio
import json
import signal
import sys

import pytest

from app.rag import parsing
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError
from app.rag.parsing import ParseFailure, parse_document
from tests.support.pdfs import encrypted_pdf, text_pdf

SECRET = "synthetic-secret-must-not-reach-child"


def settings(**updates) -> RAGSettings:
    return RAGSettings(ai_enabled=True, rag_enabled=True).model_copy(update=updates)


@pytest.fixture
def spawned(monkeypatch):
    """Record every child process and the environment it was given."""
    calls = []
    original = asyncio.create_subprocess_exec

    async def recording(*args, **kwargs):
        process = await original(*args, **kwargs)
        calls.append((args, kwargs, process))
        return process

    monkeypatch.setattr(parsing.asyncio, "create_subprocess_exec", recording)
    return calls


async def test_pdf_pages_are_extracted_in_order(spawned):
    data = text_pdf(["First page about pumps.", "Second page (filters)."])
    parsed = await parse_document(data, ContentType.PDF, settings())
    assert parsed.page_count == 2
    assert [page for page, _ in parsed.pages] == [1, 2]
    assert "pumps" in parsed.pages[0][1] and "(filters)" in parsed.pages[1][1]
    args, kwargs, process = spawned[0]
    assert args[1:5] == ("-I", "-m", "app.rag.extract", "pdf")
    assert kwargs["start_new_session"] is True
    assert process.returncode == 0


@pytest.fixture
def child(monkeypatch):
    """Replace the PDF child with a script, keeping the real spawn options."""
    original = asyncio.create_subprocess_exec
    processes = []

    def use(script: str) -> list:
        async def scripted(*args, **kwargs):
            process = await original(sys.executable, "-I", "-c", script, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(parsing.asyncio, "create_subprocess_exec", scripted)
        return processes

    return use


def reply(output: str, exit_code: int = 0) -> str:
    """Child script: drain stdin, print `output` verbatim, exit with `exit_code`."""
    return (
        "import sys; sys.stdin.buffer.read(); "
        f"sys.stdout.write({output!r}); sys.stdout.flush(); sys.exit({exit_code})"
    )


async def test_child_environment_is_scrubbed(child, monkeypatch):
    for name in ("DB_PASSWORD", "OPENAI_API_KEY", "JWT_SECRET", "LANGSMITH_API_KEY"):
        monkeypatch.setenv(name, SECRET)
    # The real child process reports the environment it actually received.
    child(
        "import json, os, sys; sys.stdin.buffer.read(); "
        "sys.stdout.write(json.dumps({'pages': [json.dumps(dict(os.environ))]}))"
    )
    parsed = await parse_document(text_pdf(["Scrubbed."]), ContentType.PDF, settings())
    environment = json.loads(parsed.pages[0][1])
    assert SECRET not in parsed.pages[0][1]
    assert set(environment) - {"LC_CTYPE"} == {"PATH", "LANG"}


async def test_unavailable_sandbox_is_not_a_document_failure(child):
    child(reply("", exit_code=parsing.SANDBOX_UNAVAILABLE_EXIT))
    with pytest.raises(parsing.SandboxUnavailable):
        await parse_document(text_pdf(["Valid."]), ContentType.PDF, settings())


@pytest.mark.parametrize(
    ("output", "exit_code"),
    [
        ("not json", 0),
        ("[]", 0),
        ('{"error": []}', 0),
        ('{"error": {}}', 0),
        ('{"pages": [1]}', 0),
        ('{"pages": ["nul \\u0000"]}', 0),
        ('{"pages": ["lone \\ud800"]}', 0),
        (json.dumps({"pages": ["x" * 60, "y" * 60]}), 0),
        (json.dumps({"pages": ["fine"]}), 1),
    ],
    ids=[
        "not-json",
        "not-object",
        "error-list",
        "error-object",
        "page-number",
        "nul",
        "lone-surrogate",
        "too-many-characters",
        "nonzero-exit",
    ],
)
async def test_forged_child_output_is_a_parse_failure(child, output, exit_code):
    child(reply(output, exit_code))
    with pytest.raises(ParseFailure) as error:
        await parse_document(text_pdf(["x"]), ContentType.PDF, settings(rag_max_characters=100))
    assert error.value.code is IngestError.PDF_PARSE_FAILED


async def test_oversize_child_output_is_cut_off_and_killed(child):
    # Writes far past the bound and never exits by itself.
    processes = child(
        "import sys, time; sys.stdin.buffer.read(); "
        "[sys.stdout.write('x' * 65536) for _ in range(64)]; sys.stdout.flush(); time.sleep(60)"
    )
    async with asyncio.timeout(10):
        with pytest.raises(ParseFailure) as error:
            await parse_document(
                text_pdf(["x"]), ContentType.PDF, settings(rag_max_characters=1000)
            )
    assert error.value.code is IngestError.PDF_PARSE_FAILED
    assert processes[0].returncode == -signal.SIGKILL


async def test_largest_valid_output_fits_the_bound(child):
    # ensure_ascii JSON spends 12 bytes on each non-BMP character.
    pages = ["\U0001f600" * 10] * 300
    child(reply(json.dumps({"pages": pages}, ensure_ascii=True)))
    parsed = await parse_document(
        text_pdf(["x"]), ContentType.PDF, settings(rag_max_characters=3000)
    )
    assert parsed.page_count == 300


@pytest.mark.parametrize(
    ("data", "overrides", "code"),
    [
        (encrypted_pdf(), {}, IngestError.PDF_ENCRYPTED),
        (text_pdf(["a", "b", "c"]), {"rag_pdf_max_pages": 2}, IngestError.PDF_PAGE_LIMIT),
        (text_pdf(["x" * 80, "y" * 80]), {"rag_max_characters": 100}, IngestError.TEXT_LIMIT),
        (b"%PDF-1.4\n" + b"\x00garbage" * 200, {}, IngestError.PDF_PARSE_FAILED),
        (text_pdf([" "]), {}, IngestError.NO_TEXT),
    ],
    ids=["encrypted", "pages", "characters", "garbage", "blank"],
)
async def test_pdf_limits_fail_with_fixed_codes(data, overrides, code):
    with pytest.raises(ParseFailure) as error:
        await parse_document(data, ContentType.PDF, settings(**overrides))
    assert error.value.code is code
    assert str(error.value) == code.value


async def test_slow_child_is_killed_on_timeout(spawned):
    # Interpreter start-up alone outlasts this deadline.
    async with asyncio.timeout(5):
        with pytest.raises(ParseFailure) as error:
            await parse_document(
                text_pdf(["Slow."]), ContentType.PDF, settings(rag_pdf_timeout_seconds=0.01)
            )
    assert error.value.code is IngestError.PDF_PARSE_TIMEOUT
    _, _, process = spawned[0]
    assert process.returncode == -signal.SIGKILL


async def test_cancelled_parse_kills_the_child(spawned):
    task = asyncio.create_task(parse_document(text_pdf(["Cancel."]), ContentType.PDF, settings()))
    async with asyncio.timeout(5):
        while not spawned:
            await asyncio.sleep(0.001)
    task.cancel()
    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await task
    _, _, process = spawned[0]
    assert process.returncode == -signal.SIGKILL


async def test_text_and_markdown_decode_with_bom_and_unicode():
    parsed = await parse_document("Café 東京 ✓".encode("utf-8-sig"), ContentType.TEXT, settings())
    assert parsed.pages == [(None, "Café 東京 ✓")] and parsed.page_count is None


@pytest.mark.parametrize(
    ("data", "code"),
    [(b"   \n\t ", IngestError.NO_TEXT), (b"x" * 101, IngestError.TEXT_LIMIT)],
)
async def test_text_limits(data, code):
    with pytest.raises(ParseFailure) as error:
        await parse_document(data, ContentType.MARKDOWN, settings(rag_max_characters=100))
    assert error.value.code is code
