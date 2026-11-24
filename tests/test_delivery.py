"""Image delivery workflow: pins, permissions and publish gates. No connections."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
CI = ROOT / ".github" / "workflows" / "ci.yml"
SCRIPT = ".github/scripts/publish_images.sh"

PINNED_USES = re.compile(r"^[\w.-]+(/[\w.-]+)+@[0-9a-f]{40} # v\d+\.\d+\.\d+$")
READ_ONLY = {"contents": "read"}
# The pinned actions/attest README lists the last three.
PUBLISH_PERMISSIONS = {
    "contents": "read",
    "packages": "write",
    "id-token": "write",
    "attestations": "write",
    "artifact-metadata": "write",
}
PUBLISH_IF = (
    "github.event_name == 'push' && github.ref == 'refs/heads/main' "
    "&& github.repository == 'manshoor/fastapi-ai-agent-platform'"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _jobs() -> dict:
    return _load(CI)["jobs"]


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


# Every workflow


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_action_is_pinned_to_a_commit_with_its_version(path):
    found = re.findall(r"^\s*(?:- )?uses:\s*(.+)$", path.read_text(), re.MULTILINE)
    assert found
    for reference in found:
        assert PINNED_USES.match(reference), reference


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_the_default_token_is_read_only(path):
    assert _load(path)["permissions"] == READ_ONLY


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_no_stored_secret_is_used(path):
    assert not re.search(r"\bsecrets\b", re.sub(r"#.*", "", path.read_text()))


# Permissions and gates in ci.yml


def test_only_publish_sets_permissions():
    jobs = _jobs()
    assert {name for name, job in jobs.items() if "permissions" in job} == {"publish"}
    assert jobs["publish"]["permissions"] == PUBLISH_PERMISSIONS


def test_publish_runs_only_on_main_pushes_in_this_repository():
    assert _jobs()["publish"]["if"] == PUBLISH_IF


def test_publish_needs_every_gate():
    # publish-check is skipped on main, so needing it would skip publish too.
    assert set(_jobs()["publish"]["needs"]) == {"lint", "test", "ai-qa", "image"}


def test_publish_check_runs_only_on_pull_requests():
    job = _jobs()["publish-check"]
    assert job["if"] == "github.event_name == 'pull_request'"
    assert job["needs"] == "image"


def test_no_gate_can_be_skipped_or_ignored():
    for name in ("lint", "test", "ai-qa", "image"):
        job = _jobs()[name]
        assert "if" not in job
        assert "continue-on-error" not in job
        assert all("continue-on-error" not in step for step in job["steps"])


def test_only_pull_request_runs_are_cancelled():
    # A cancelled main run could cut off a publish.
    concurrency = _load(CI)["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_checkouts_keep_no_credentials():
    for job in _jobs().values():
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


# Publishing


def test_publish_neither_builds_nor_runs_code():
    commands = _commands(_jobs()["publish"])
    assert not re.search(r"\b(docker|buildx|compose|uv|python3?|pip|make)\b", commands)
    assert [line for line in commands.splitlines() if ".github" in line] == [
        f'{SCRIPT} "$REGISTRY" "$GITHUB_SHA" archives >> "$GITHUB_OUTPUT"'
    ]


def test_pull_requests_run_the_script_that_publishes():
    assert SCRIPT in _commands(_jobs()["publish-check"])


def test_every_published_image_is_attested_by_digest():
    steps = [
        step
        for step in _jobs()["publish"]["steps"]
        if step.get("uses", "").startswith("actions/attest@")
    ]
    base = "ghcr.io/manshoor/fastapi-ai-agent-platform"
    assert [step["with"] for step in steps] == [
        {
            "subject-name": base + suffix,
            "subject-digest": f"${{{{ steps.publish.outputs.{name} }}}}",
        }
        for name, suffix in (("app", ""), ("postgres", "-postgres"), ("simulator", "-simulator"))
    ]


def test_the_throwaway_registry_is_pinned_by_digest():
    assert re.search(
        r"\bregistry:[\w.]+@sha256:[0-9a-f]{64}\b", _commands(_jobs()["publish-check"])
    )


# Image build


def test_images_are_built_natively_on_both_platforms():
    job = _jobs()["image"]
    assert job["runs-on"] == "${{ matrix.runner }}"
    assert job["strategy"]["matrix"]["include"] == [
        {"runner": "ubuntu-24.04", "arch": "amd64"},
        {"runner": "ubuntu-24.04-arm", "arch": "arm64"},
    ]


def test_image_build_uses_no_shared_cache():
    job = _jobs()["image"]
    assert not re.search(r"cache-(from|to)|type=gha", _commands(job))
    assert not [step for step in job["steps"] if "actions/cache" in step.get("uses", "")]
    for step in job["steps"]:
        if step.get("uses", "").startswith("astral-sh/setup-uv@"):
            assert step["with"]["enable-cache"] is False


def test_image_archives_are_kept_one_day():
    (upload,) = [
        step
        for step in _jobs()["image"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert upload["with"]["retention-days"] == 1
    assert upload["with"]["if-no-files-found"] == "error"


def test_every_entry_point_is_imported():
    commands = _commands(_jobs()["image"])
    for module in (
        "app.main",
        "app.ai.dispatcher",
        "app.rag.sinks",
        "app.ai.worker",
        "app.rag.worker",
        "app.rag.sqs_worker",
        "app.rag.connectors.__main__",
    ):
        assert re.search(rf"\b{re.escape(module)}\b", commands), module
