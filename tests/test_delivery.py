"""Image delivery workflow: pins, permissions and publish gates. No connections.

Each `rule_*` raises on a bad workflow. It runs on the real file, and the mutation tests
run it on edited copies that it must reject.
"""

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
CI = WORKFLOW_DIR / "ci.yml"
SCRIPT = ".github/scripts/publish_images.sh"
GATES = ("lint", "test", "ai-qa", "image")

PINNED = re.compile(r"[\w.-]+(/[\w.-]+)+@[0-9a-f]{40}")
VERSIONED = re.compile(r"[\w.-]+(/[\w.-]+)+@[0-9a-f]{40} # v\d+\.\d+\.\d+")
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
PUBLISH_RUN = f'{SCRIPT} "$REGISTRY" "$GITHUB_SHA" archives >> "$GITHUB_OUTPUT"'
LOGIN_RUN = (
    'printf \'%s\' "$GHCR_TOKEN" | skopeo login --username "$GITHUB_ACTOR" --password-stdin ghcr.io'
)
SMOKE_STEP = "Smoke-test the app and PostgreSQL images"
SMOKE_RUN = "$COMPOSE up --detach --wait api\nuv run --locked python scripts/smoke.py\n"
SAVE_RUN = (
    'docker save --output "images-${{ matrix.arch }}.tar" '
    '"$APP_IMAGE" "$POSTGRES_IMAGE" "$SIMULATOR_IMAGE"'
)
BUILD_COMMAND = re.compile(
    r"\bdocker (image )?(build|buildx|builder|commit|tag|load|import|pull)\b"
    r"|\b(compose|\$COMPOSE|buildx) (build|bake|pull)\b|\$COMPOSE build\b|--build\b"
)
# The only expression a shell command may hold; anything else could carry event text.
RUN_EXPRESSIONS = {"matrix.arch"}


def _workflows(folder: Path = WORKFLOW_DIR) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix in {".yml", ".yaml"})


WORKFLOWS = _workflows()


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _jobs() -> dict:
    return _load(CI)["jobs"]


def _commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _steps(workflow: dict) -> Iterator[dict]:
    for job in workflow["jobs"].values():
        yield from job.get("steps", [])


def _action(step: dict) -> str:
    return step.get("uses", "").split("@")[0]


# Rules for every workflow


def rule_read_only(workflow: dict, name: str = "ci.yml") -> None:
    assert workflow["permissions"] == READ_ONLY
    writers = {job for job, body in workflow["jobs"].items() if "permissions" in body}
    assert writers == ({"publish"} if name == "ci.yml" else set())


def rule_pins(workflow: dict) -> None:
    found = [step["uses"] for step in _steps(workflow) if "uses" in step]
    found += [job["uses"] for job in workflow["jobs"].values() if "uses" in job]
    for reference in found:
        assert PINNED.fullmatch(reference), reference


def rule_no_event_text(workflow: dict) -> None:
    for step in _steps(workflow):
        run = step.get("run", "")
        assert run.count("${{") == len(re.findall(r"\$\{\{.*?\}\}", run, re.DOTALL))
        found = {match.strip() for match in re.findall(r"\$\{\{(.*?)\}\}", run, re.DOTALL)}
        assert found <= RUN_EXPRESSIONS, found


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_workflow_is_read_only_pinned_and_free_of_event_text(path):
    workflow = _load(path)
    rule_read_only(workflow, path.name)
    rule_pins(workflow)
    rule_no_event_text(workflow)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_pin_names_its_version(path):
    found = re.findall(r"^\s*(?:- )?uses:\s*(.+)$", path.read_text(), re.MULTILINE)
    parsed = [step for step in _steps(_load(path)) if "uses" in step]
    assert len(found) == len(parsed)
    for reference in found:
        assert VERSIONED.fullmatch(reference), reference


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_no_stored_secret_is_used(path):
    assert not re.search(r"\bsecrets\b", re.sub(r"#.*", "", path.read_text()))


def test_both_workflow_suffixes_are_covered(tmp_path):
    for name in ("a.yml", "b.yaml", "notes.md"):
        (tmp_path / name).write_text("")
    assert [path.name for path in _workflows(tmp_path)] == ["a.yml", "b.yaml"]


# Rules for ci.yml


def rule_triggers(workflow: dict) -> None:
    # YAML 1.1 reads the key `on` as true.
    assert workflow[True] == {
        "push": {"branches": ["main"]},
        "pull_request": None,
        "workflow_dispatch": None,
    }


def rule_publish_gate(workflow: dict) -> None:
    job = workflow["jobs"]["publish"]
    assert job["permissions"] == PUBLISH_PERMISSIONS
    assert job["if"] == PUBLISH_IF
    # publish-check is skipped on main, so needing it would skip publish too.
    assert set(job["needs"]) == set(GATES)


def rule_publish_steps(workflow: dict) -> None:
    # A shell, container or variable from outside the steps could run other code.
    assert not {"env", "defaults"} & set(workflow)
    job = workflow["jobs"]["publish"]
    assert set(job) == {
        "if",
        "needs",
        "runs-on",
        "timeout-minutes",
        "permissions",
        "env",
        "steps",
    }
    assert job["env"] == {"REGISTRY": "ghcr.io/manshoor"}
    assert job["runs-on"] == "ubuntu-24.04"
    inputs: dict[str, list[dict]] = {}
    runs = []
    for step in job["steps"]:
        assert set(step) <= {"name", "id", "uses", "with", "run", "env"}, step
        if "uses" in step:
            assert "env" not in step
            inputs.setdefault(_action(step), []).append(step.get("with"))
        else:
            runs.append((step["run"], step.get("env")))
    assert set(inputs) == {"actions/checkout", "actions/download-artifact", "actions/attest"}
    # No other ref or repository; no archives from another run.
    assert inputs["actions/checkout"] == [{"persist-credentials": False}]
    assert inputs["actions/download-artifact"] == [
        {"pattern": "images-*", "path": "archives", "merge-multiple": True}
    ]
    assert runs == [
        (LOGIN_RUN, {"GHCR_TOKEN": "${{ github.token }}"}),
        (PUBLISH_RUN, None),
    ]
    # Each image is attested by the digest the script printed.
    base = "ghcr.io/manshoor/fastapi-ai-agent-platform"
    assert inputs["actions/attest"] == [
        {
            "subject-name": base + suffix,
            "subject-digest": f"${{{{ steps.publish.outputs.{name} }}}}",
        }
        for name, suffix in (("app", ""), ("postgres", "-postgres"), ("simulator", "-simulator"))
    ]
    (script,) = [step for step in job["steps"] if step.get("run") == PUBLISH_RUN]
    assert script["id"] == "publish"


def rule_gates_cannot_be_skipped(workflow: dict) -> None:
    for name in GATES:
        job = workflow["jobs"][name]
        assert not {"if", "continue-on-error"} & set(job), name
        for step in job["steps"]:
            assert "continue-on-error" not in step, step
            assert not re.search(r"\|\|\s*(true|:|exit 0)|\bset \+e", step.get("run", "")), step
            # Only logs and teardown may be conditional.
            if "if" in step:
                assert re.fullmatch(r"\$COMPOSE (logs|down) [\w -]+", step["run"]), step


def rule_image_order(workflow: dict) -> None:
    job = workflow["jobs"]["image"]
    assert job["env"]["COMPOSE"].endswith("-f compose.yaml -f compose.image.yaml")
    steps = job["steps"]
    smoke = {"name": SMOKE_STEP, "run": SMOKE_RUN}
    assert steps.count(smoke) == 1
    (save,) = [step for step in steps if "docker save" in step.get("run", "")]
    assert save == {"name": "Save the tested images", "run": SAVE_RUN}
    builds = [i for i, step in enumerate(steps) if BUILD_COMMAND.search(step.get("run", ""))]
    assert builds and max(builds) < steps.index(smoke) < steps.index(save)
    # Nothing runs between the save and the upload.
    assert _action(steps[steps.index(save) + 1]) == "actions/upload-artifact"


CI_RULES: list[Callable[[dict], None]] = [
    rule_read_only,
    rule_pins,
    rule_no_event_text,
    rule_triggers,
    rule_publish_gate,
    rule_publish_steps,
    rule_gates_cannot_be_skipped,
    rule_image_order,
]


@pytest.mark.parametrize("rule", CI_RULES, ids=lambda rule: rule.__name__)
def test_ci_follows(rule):
    rule(_load(CI))


# Bad edits each rule must reject

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
CHECKOUT = f"      - uses: actions/checkout@{SHA} # v7.0.1\n"
LINT_RUN = "      - run: uv run --locked ruff check .\n"
SAVE = "      - name: Save the tested images\n"
SMOKE = f"      - name: {SMOKE_STEP}\n"
LOGIN = "      - name: Log in to GHCR\n"
DISPATCH = "  workflow_dispatch:\n"
PATTERN = "          pattern: images-*\n"
BAD_EDITS = [
    ("write-default", rule_read_only, "permissions:\n  contents: read", "permissions: write-all"),
    (
        "second-writer",
        rule_read_only,
        "  lint:\n",
        "  lint:\n    permissions:\n      packages: write\n",
    ),
    ("quoted-tag", rule_pins, f"uses: actions/checkout@{SHA} # v7.0.1", 'uses: "a/b@v7"'),
    ("flow-style-tag", rule_pins, LINT_RUN, "      - {uses: a/b@main}\n"),
    ("quoted-key-tag", rule_pins, LINT_RUN, '      - "uses": a/b@main\n'),
    ("short-sha", rule_pins, SHA, SHA[:12]),
    ("docker-action", rule_pins, LINT_RUN, "      - uses: docker://alpine:3\n"),
    (
        "job-level-tag",
        rule_pins,
        "jobs:\n",
        "jobs:\n  extra:\n    uses: a/b/.github/workflows/c.yml@main\n",
    ),
    (
        "event-title",
        rule_no_event_text,
        LINT_RUN,
        '      - run: echo "${{ github.event.pull_request.title }}"\n',
    ),
    ("head-ref", rule_no_event_text, LINT_RUN, "      - run: echo ${{github.head_ref}}\n"),
    ("index-style", rule_no_event_text, LINT_RUN, "      - run: echo ${{ github['event'] }}\n"),
    (
        "folded-expression",
        rule_no_event_text,
        LINT_RUN,
        "      - run: |\n          echo ${{\n          github.head_ref }}\n",
    ),
    ("target-trigger", rule_triggers, DISPATCH, DISPATCH + "  pull_request_target:\n"),
    ("run-trigger", rule_triggers, DISPATCH, DISPATCH + "  workflow_run:\n"),
    ("any-branch", rule_triggers, "    branches: [main]", '    branches: ["**"]'),
    ("tag-push", rule_triggers, "    branches: [main]", '    branches: [main]\n    tags: ["*"]'),
    ("no-gate", rule_publish_gate, "needs: [lint, test, ai-qa, image]", "needs: [lint]"),
    ("any-event", rule_publish_gate, "github.event_name == 'push' && ", ""),
    (
        "other-ref",
        rule_publish_steps,
        LOGIN,
        CHECKOUT + "        with:\n          ref: x\n" + LOGIN,
    ),
    (
        "ref-input",
        rule_publish_steps,
        "          persist-credentials: false\n",
        "          persist-credentials: false\n          ref: x\n",
    ),
    ("other-run", rule_publish_steps, PATTERN, PATTERN + '          run-id: "1"\n'),
    ("build-in-publish", rule_publish_steps, LOGIN, "      - run: docker build .\n" + LOGIN),
    ("code-in-publish", rule_publish_steps, LOGIN, "      - run: node x.js\n" + LOGIN),
    ("other-action", rule_publish_steps, LOGIN, f"      - uses: a/b@{SHA}\n" + LOGIN),
    ("appended-code", rule_publish_steps, 'archives >> "$GITHUB_OUTPUT"', "archives && ./x"),
    ("workflow-env", rule_publish_steps, "jobs:\n", "env:\n  BASH_ENV: x.sh\njobs:\n"),
    (
        "other-subject",
        rule_publish_steps,
        "${{ steps.publish.outputs.app }}",
        "${{ steps.publish.outputs.postgres }}",
    ),
    ("smoke-if", rule_image_order, SMOKE, SMOKE + "        if: false\n"),
    ("smoke-ignored", rule_image_order, SMOKE, SMOKE + "        continue-on-error: true\n"),
    ("smoke-or-true", rule_image_order, "scripts/smoke.py\n", "scripts/smoke.py || true\n"),
    ("smoke-renamed", rule_image_order, SMOKE, "      - name: Other\n"),
    ("no-smoke", rule_image_order, "          uv run --locked python scripts/smoke.py\n", ""),
    ("rebuild", rule_image_order, SAVE, '      - run: docker build -t "$APP_IMAGE" .\n' + SAVE),
    ("retag", rule_image_order, SAVE, '      - run: docker tag x "$APP_IMAGE"\n' + SAVE),
    ("compose-build", rule_image_order, SAVE, "      - run: $COMPOSE build\n" + SAVE),
    ("no-overlay", rule_image_order, " -f compose.image.yaml", ""),
    ("gate-if", rule_gates_cannot_be_skipped, "  lint:\n", "  lint:\n    if: false\n"),
    ("gate-step-if", rule_gates_cannot_be_skipped, LINT_RUN, LINT_RUN + "        if: false\n"),
    (
        "gate-or-true",
        rule_gates_cannot_be_skipped,
        "ruff check .\n",
        "ruff check . || true\n",
    ),
]


@pytest.mark.parametrize(
    ("rule", "old", "new"), [edit[1:] for edit in BAD_EDITS], ids=[edit[0] for edit in BAD_EDITS]
)
def test_a_bad_edit_is_rejected(rule, old, new):
    text = CI.read_text()
    assert old in text
    # An edit inside `publish` must hit that job, not an earlier one with the same text.
    start = text.index("\n  publish:\n") if rule is rule_publish_steps else 0
    if old not in text[start:]:
        start = 0
    edited = text[:start] + text[start:].replace(old, new, 1)
    with pytest.raises(AssertionError):
        rule(yaml.safe_load(edited))


def test_a_new_workflow_with_write_access_is_rejected(tmp_path):
    (tmp_path / "rogue.yaml").write_text(
        "on: push\npermissions: {contents: read}\n"
        "jobs: {a: {permissions: {packages: write}, steps: []}}\n"
    )
    (path,) = _workflows(tmp_path)
    with pytest.raises(AssertionError):
        rule_read_only(_load(path), path.name)


def test_publish_check_runs_only_on_pull_requests():
    job = _jobs()["publish-check"]
    assert job["if"] == "github.event_name == 'pull_request'"
    assert job["needs"] == "image"


def test_only_pull_request_runs_are_cancelled():
    # A cancelled main run could cut off a publish.
    concurrency = _load(CI)["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_checkouts_keep_no_credentials():
    for step in _steps(_load(CI)):
        if _action(step) == "actions/checkout":
            assert step["with"]["persist-credentials"] is False


# Publishing


def test_pull_requests_run_the_script_that_publishes():
    assert SCRIPT in _commands(_jobs()["publish-check"])


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
