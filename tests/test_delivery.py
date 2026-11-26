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
GITOPS = WORKFLOW_DIR / "gitops.yml"
SCRIPT = ".github/scripts/publish_images.sh"
GATES = ("lint", "test", "ai-qa", "image")

PINNED = re.compile(r"[\w.-]+(/[\w.-]+)+@[0-9a-f]{40}")
VERSIONED = re.compile(r"[\w.-]+(/[\w.-]+)+@[0-9a-f]{40} # v\d+\.\d+\.\d+")
READ_LEVELS = {"read", "none"}
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
SECRET_WORD = re.compile(r"\bsecrets\b", re.IGNORECASE)
SECRET_REFERENCE = re.compile(
    r"\bsecrets\s*[.\[]|\bsecrets[\"']?\s*:\s*[\"']?inherit\b", re.IGNORECASE
)
SECRET_KEY = re.compile(r"secrets", re.IGNORECASE)
# Expressions an action input may hold: values this repository's workflows make, and the
# commit to check out.
INPUT_EXPRESSIONS = {
    "steps.publish.outputs.app",
    "steps.publish.outputs.postgres",
    "steps.publish.outputs.simulator",
    "matrix.arch",
    "github.event.pull_request.head.sha || github.sha",
}
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


def _only_reads(permissions: object) -> bool:
    # `read-all` and `write-all` are strings, not maps.
    return isinstance(permissions, dict) and all(
        isinstance(level, str) and level in READ_LEVELS for level in permissions.values()
    )


def rule_read_only(workflow: dict, name: str = "ci.yml") -> None:
    # A missing key means the repository default, which may write.
    assert _only_reads(workflow.get("permissions"))
    for job, body in workflow["jobs"].items():
        # The one job that may write; rule_publish_gate fixes its exact permissions.
        if (name, job) == ("ci.yml", "publish"):
            continue
        assert "permissions" not in body or _only_reads(body["permissions"]), job


def _strings(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def _mappings(node: object) -> Iterator[dict]:
    if isinstance(node, dict):
        yield node
        node = list(node.values())
    if isinstance(node, list):
        for item in node:
            yield from _mappings(item)


def rule_no_secrets(workflow: dict) -> None:
    # The bare word is fine in a name or a command; reading a secret is not. Expression
    # names ignore case.
    for text in _strings(workflow):
        assert not SECRET_REFERENCE.search(text), text
        # Also whole-object use, such as toJSON(secrets). To the end of the text: an
        # expression's own strings may hold "}}".
        assert not SECRET_WORD.search(text.partition("${{")[2]), text
    for mapping in _mappings(workflow):
        # A job passing secrets on (`secrets: inherit`) or a workflow asking for them.
        assert not [key for key in mapping if SECRET_KEY.fullmatch(str(key))], mapping
        # `if:` is an expression even without the braces.
        assert not SECRET_WORD.search(str(mapping.get("if", "")))


def rule_pins(workflow: dict) -> None:
    found = [step["uses"] for step in _steps(workflow) if "uses" in step]
    found += [job["uses"] for job in workflow["jobs"].values() if "uses" in job]
    for reference in found:
        assert PINNED.fullmatch(reference), reference
    # Their images would need digests; no job needs them.
    for name, job in workflow["jobs"].items():
        assert not {"container", "services"} & set(job), name


def _check_text(node: object, allowed: set[str]) -> None:
    for text in _strings(node):
        assert text.count("${{") == len(re.findall(r"\$\{\{.*?\}\}", text, re.DOTALL))
        found = {match.strip() for match in re.findall(r"\$\{\{(.*?)\}\}", text, re.DOTALL)}
        assert found <= allowed, found


def rule_no_event_text(workflow: dict) -> None:
    _check_text(workflow.get("defaults"), RUN_EXPRESSIONS)
    for job in workflow["jobs"].values():
        _check_text(job.get("defaults"), RUN_EXPRESSIONS)
        _check_text(job.get("with"), INPUT_EXPRESSIONS)
    for step in _steps(workflow):
        _check_text([step.get("run"), step.get("shell")], RUN_EXPRESSIONS)
        _check_text(step.get("with"), INPUT_EXPRESSIONS)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_workflow_is_read_only_pinned_and_free_of_event_text_and_secrets(path):
    workflow = _load(path)
    rule_read_only(workflow, path.name)
    rule_pins(workflow)
    rule_no_event_text(workflow)
    rule_no_secrets(workflow)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_every_pin_names_its_version(path):
    found = re.findall(r"^\s*(?:- )?uses:\s*(.+)$", path.read_text(), re.MULTILINE)
    parsed = [step for step in _steps(_load(path)) if "uses" in step]
    assert len(found) == len(parsed)
    for reference in found:
        assert VERSIONED.fullmatch(reference), reference


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
    rule_no_secrets,
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
    ("read-all", rule_read_only, "permissions:\n  contents: read", "permissions: read-all"),
    ("no-permissions", rule_read_only, "permissions:\n  contents: read\n", ""),
    ("null-permissions", rule_read_only, "permissions:\n  contents: read", "permissions:"),
    (
        "extra-write",
        rule_read_only,
        "permissions:\n  contents: read",
        "permissions:\n  contents: read\n  packages: write",
    ),
    (
        "admin-level",
        rule_read_only,
        "permissions:\n  contents: read",
        "permissions:\n  contents: admin",
    ),
    (
        "job-mixed-levels",
        rule_read_only,
        "  lint:\n",
        "  lint:\n    permissions: {contents: read, id-token: write}\n",
    ),
    ("job-write-all", rule_read_only, "  lint:\n", "  lint:\n    permissions: write-all\n"),
    ("with-event", rule_no_event_text, PATTERN, "          pattern: ${{ github.head_ref }}\n"),
    (
        "shell-event",
        rule_no_event_text,
        LINT_RUN,
        LINT_RUN + "        shell: bash ${{ github.event.pull_request.title }}\n",
    ),
    (
        "job-with-event",
        rule_no_event_text,
        "jobs:\n",
        f"jobs:\n  extra:\n    uses: a/b/c.yml@{SHA}\n    with: {{a: '${{{{ github.actor }}}}'}}\n",
    ),
    (
        "default-shell-event",
        rule_no_event_text,
        "jobs:\n",
        "defaults: {run: {shell: '${{ github.head_ref }}'}}\njobs:\n",
    ),
    ("container", rule_pins, "  lint:\n", "  lint:\n    container: alpine:3\n"),
    ("services", rule_pins, "  lint:\n", "  lint:\n    services: {a: {image: alpine}}\n"),
    ("secret-env", rule_no_secrets, LINT_RUN, LINT_RUN + "        env: {A: '${{ secrets.A }}'}\n"),
    ("secret-run", rule_no_secrets, LINT_RUN, "      - run: echo ${{ secrets.A }}\n"),
    ("secret-index", rule_no_secrets, LINT_RUN, "      - run: echo ${{ secrets['A'] }}\n"),
    ("secret-case", rule_no_secrets, LINT_RUN, "      - run: echo ${{ SECRETS.A }}\n"),
    ("secret-space", rule_no_secrets, LINT_RUN, "      - run: echo ${{ secrets . A }}\n"),
    ("secret-all", rule_no_secrets, LINT_RUN, "      - run: echo ${{ toJSON(secrets) }}\n"),
    (
        "secret-braces",
        rule_no_secrets,
        LINT_RUN,
        "      - run: echo \"${{ format('}}', toJSON(secrets)) }}\"\n",
    ),
    ("secret-hash", rule_no_secrets, LINT_RUN, "      - run: 'echo \"#${{ secrets.A }}\"'\n"),
    (
        "secret-folded",
        rule_no_secrets,
        LINT_RUN,
        "      - run: |\n          echo ${{ toJSON(\n          secrets) }}\n",
    ),
    ("secret-if", rule_no_secrets, LINT_RUN, LINT_RUN + "        if: toJSON(secrets) != ''\n"),
    ("secret-input", rule_no_secrets, PATTERN, "          pattern: ${{ secrets.A }}\n"),
    (
        "secret-inherit",
        rule_no_secrets,
        "jobs:\n",
        f"jobs:\n  extra:\n    uses: a/b/.github/workflows/c.yml@{SHA}\n    secrets: inherit\n",
    ),
    (
        "secret-inherit-quoted",
        rule_no_secrets,
        "jobs:\n",
        f'jobs:\n  extra:\n    uses: a/b/c.yml@{SHA}\n    "secrets" : "inherit"\n',
    ),
    (
        "secret-passed",
        rule_no_secrets,
        "jobs:\n",
        f"jobs:\n  extra:\n    uses: a/b/c.yml@{SHA}\n    secrets: {{A: b}}\n",
    ),
    (
        "secret-inherit-text",
        rule_no_secrets,
        LINT_RUN,
        "      - run: 'echo \"secrets: inherit\"'\n",
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


# Edits that only read: the loosened rules must accept them.
GOOD_EDITS = [
    (
        "more-reads",
        rule_read_only,
        "permissions:\n  contents: read",
        "permissions:\n  contents: read\n  attestations: read\n  packages: none",
    ),
    (
        "job-reads",
        rule_read_only,
        "  lint:\n",
        "  lint:\n    permissions: {contents: read, packages: none}\n",
    ),
    ("bare-word", rule_no_secrets, LINT_RUN, "      - run: echo no secrets here\n"),
]


@pytest.mark.parametrize(
    ("rule", "old", "new"), [edit[1:] for edit in GOOD_EDITS], ids=[edit[0] for edit in GOOD_EDITS]
)
def test_a_reading_edit_is_accepted(rule, old, new):
    text = CI.read_text()
    assert old in text
    rule(yaml.safe_load(text.replace(old, new, 1)))


@pytest.mark.parametrize(
    ("rule", "old", "new"),
    [
        (rule_read_only, "permissions:\n  contents: read", "permissions:\n  contents: write"),
        (rule_read_only, "  cluster:\n", "  cluster:\n    permissions: {packages: write}\n"),
        # Only ci.yml's publish job may write.
        (rule_read_only, "  cluster:\n", "  publish:\n    permissions: {packages: write}\n"),
        (rule_no_secrets, "${{ github.token }}", "${{ secrets.GITHUB_TOKEN }}"),
        (rule_no_event_text, '--revision "$REVISION"', "--revision ${{ github.head_ref }}"),
        (rule_pins, "actions/checkout@", "actions/checkout@v7 #"),
        (rule_pins, "  cluster:\n", "  cluster:\n    container: alpine:3\n"),
        (rule_no_event_text, "fetch-depth: 0", "fetch-depth: ${{ github.head_ref }}"),
    ],
)
def test_a_bad_gitops_edit_is_rejected(rule, old, new):
    text = GITOPS.read_text()
    assert old in text
    workflow = yaml.safe_load(text.replace(old, new, 1))
    arguments = (GITOPS.name,) if rule is rule_read_only else ()
    with pytest.raises(AssertionError):
        rule(workflow, *arguments)


# Rules for gitops.yml

CHECKED_OUT = "${{ github.event.pull_request.head.sha || github.sha }}"
GITOPS_PATHS = [
    "pyproject.toml",
    "uv.lock",
    "deploy/**",
    "scripts/check_deploy.py",
    "scripts/gitops_cluster.py",
    "scripts/promote_images.py",
    "scripts/smoke.py",
    "scripts/smoke_ai.py",
    "scripts/smoke_assistant.py",
    "scripts/smoke_gitops.py",
    "scripts/smoke_rag.py",
    "app/assistant/totp.py",
    "tests/support/docx_files.py",
    "tests/support/pdfs.py",
    ".github/workflows/gitops.yml",
]
INSTALL_KUBECTL = """\
mkdir -p "$RUNNER_TEMP/bin"
curl --fail --silent --show-error --location --retry 3 --output "$RUNNER_TEMP/bin/kubectl" "$KUBECTL_URL"
echo "$KUBECTL_SHA256  $RUNNER_TEMP/bin/kubectl" | sha256sum --check --strict
chmod +x "$RUNNER_TEMP/bin/kubectl"
echo "$RUNNER_TEMP/bin" >> "$GITHUB_PATH"
"""  # noqa: E501
INSTALL_KIND = """\
curl --fail --silent --show-error --location --retry 3 --output "$RUNNER_TEMP/bin/kind" "$KIND_URL"
echo "$KIND_SHA256  $RUNNER_TEMP/bin/kind" | sha256sum --check --strict
chmod +x "$RUNNER_TEMP/bin/kind"
"""
INSTALL_POPPLER = """\
sudo apt-get update
sudo apt-get install --yes --no-install-recommends poppler-utils
command -v pdftoppm
"""
SETUP_RUNS = [None, None, "uv python install 3.13", "uv sync --locked", INSTALL_KUBECTL]
MANIFESTS_RUNS = [
    *SETUP_RUNS,
    'kubectl kustomize deploy/app > "$RUNNER_TEMP/app.yaml"',
    'uv run --locked python scripts/check_deploy.py < "$RUNNER_TEMP/app.yaml"',
    "uv run --locked python scripts/promote_images.py verify",
]
DUMP_STEP = "Show pods, events and logs"
DUMP_PODS = "db redis kafka provider migrate topic-init api dispatcher worker rag-worker"
DUMP_ARGOCD = "argocd-application-controller argocd-repo-server"
CLUSTER_RUNS = [
    *SETUP_RUNS,
    INSTALL_KIND,
    INSTALL_POPPLER,
    'uv run --locked python scripts/gitops_cluster.py up --revision "$REVISION"',
    'uv run --locked python scripts/smoke_gitops.py --revision "$REVISION"',
    DUMP_STEP,
    "uv run --locked python scripts/gitops_cluster.py down",
]
GITOPS_ENV = {"KUBECTL_URL", "KUBECTL_SHA256", "KIND_URL", "KIND_SHA256"}
# Pods hold references to Secrets, never their values; nothing else here can show one.
DUMP_LINE = re.compile(
    r"kube=\(kubectl --kubeconfig \.gitops-local/kubeconfig --context kind-fastapi-gitops\)"
    r"|for name in [\w -]+; do|done"
    r"|\"\$\{kube\[@\]\}\" (--namespace [\w-]+ )?"
    r"(get pods --all-namespaces --output wide"
    r"|get application fastapi-demo --output yaml"
    r"|get deployments,statefulsets,jobs,services,persistentvolumeclaims,networkpolicies"
    r" --output wide"
    r"|describe pods"
    r"|get events --all-namespaces --sort-by \.lastTimestamp"
    r"|logs --selector \"app\.kubernetes\.io/name=\$name\" --all-containers --prefix --tail \d+"
    r") \|\| true"
)


def rule_gitops_triggers(workflow: dict) -> None:
    assert workflow[True] == {
        "push": {"branches": ["main"], "paths": GITOPS_PATHS},
        "pull_request": {"paths": GITOPS_PATHS},
        "workflow_dispatch": None,
    }


def rule_gitops_steps(workflow: dict) -> None:
    assert set(workflow) == {"name", True, "permissions", "concurrency", "env", "jobs"}
    assert workflow["permissions"] == {"contents": "read"}
    # No variable that a shell would run, such as BASH_ENV.
    assert set(workflow["env"]) == GITOPS_ENV
    jobs = workflow["jobs"]
    assert set(jobs) == {"manifests", "cluster"}
    assert set(jobs["manifests"]) == {"runs-on", "timeout-minutes", "steps"}
    assert set(jobs["cluster"]) == {"runs-on", "timeout-minutes", "env", "steps"}
    assert jobs["cluster"]["env"] == {"REVISION": CHECKED_OUT}
    assert jobs["cluster"]["timeout-minutes"] == 30
    assert {job["runs-on"] for job in jobs.values()} == {"ubuntu-24.04"}

    for job, expected in (("manifests", MANIFESTS_RUNS), ("cluster", CLUSTER_RUNS)):
        runs = []
        for step in jobs[job]["steps"]:
            dump = step.get("name") == DUMP_STEP
            runs.append(DUMP_STEP if dump else step.get("run"))
            assert set(step) <= {"name", "uses", "with", "run", "env", "if", "timeout-minutes"}
            # Only the dump and the teardown are conditional.
            condition = "failure()" if dump else "always()" if "down" in str(runs[-1]) else None
            assert step.get("if") == condition, step
            verify = str(runs[-1]).endswith("promote_images.py verify")
            token = {"GH_TOKEN": "${{ github.token }}"} if verify else None
            assert step.get("env") == token, step
        assert runs == expected

    for job in jobs.values():
        actions = [_action(step) for step in job["steps"] if "uses" in step]
        assert actions == ["actions/checkout", "astral-sh/setup-uv"]
        checkout = job["steps"][0]["with"]
        assert checkout["ref"] == CHECKED_OUT
        assert set(checkout) <= {"persist-credentials", "fetch-depth", "ref"}
        assert job["steps"][1]["with"] == {"version": "0.13.6"}
    # Without history, the pin order could not be checked.
    assert jobs["manifests"]["steps"][0]["with"].get("fetch-depth") == 0


def rule_gitops_dump(workflow: dict) -> None:
    (dump,) = [
        step for step in workflow["jobs"]["cluster"]["steps"] if step.get("name") == DUMP_STEP
    ]
    # Every workload's logs, then Argo CD's.
    loops = [line.strip() for line in dump["run"].splitlines() if "for name in" in line]
    assert loops == [f"for name in {DUMP_PODS}; do", f"for name in {DUMP_ARGOCD}; do"]
    for line in dump["run"].splitlines():
        assert DUMP_LINE.fullmatch(line.strip()), line


def rule_gitops_tools(workflow: dict) -> None:
    for tool, host in (
        ("KUBECTL", "https://dl.k8s.io/release/"),
        ("KIND", "https://github.com/kubernetes-sigs/kind/releases/download/"),
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", workflow["env"][f"{tool}_SHA256"])
        url = workflow["env"][f"{tool}_URL"]
        assert re.fullmatch(re.escape(host) + r"v\d+\.\d+\.\d+/[\w/-]+", url)


KIND_STEP = "      - name: Install kind\n"
SMOKE_LIMIT = (
    "        timeout-minutes: 15\n        run: uv run --locked python scripts/smoke_gitops.py"
)
DUMP_LIMIT = "        if: failure()\n        timeout-minutes: 2\n"
# Every cluster step that runs a command, and its limit. The sum leaves the job limit
# room for the two actions and the runner, so a hung step still fails with its logs shown.
CLUSTER_LIMITS = {
    "uv python install 3.13": 1,
    "uv sync --locked": 1,
    "Install kubectl": 1,
    "Install kind": 1,
    "Install poppler": 2,
    "Start the cluster": 4,
    "Smoke-test the cluster": 15,
    "Show pods, events and logs": 2,
    "Delete the cluster": 1,
}
CLUSTER_MINUTES = 30
ACTION_ROOM_MINUTES = 2


def rule_gitops_limits(workflow: dict) -> None:
    job = workflow["jobs"]["cluster"]
    assert job["timeout-minutes"] == CLUSTER_MINUTES
    limits = {
        step.get("name") or step["run"]: step.get("timeout-minutes")
        for step in job["steps"]
        if "run" in step
    }
    assert limits == CLUSTER_LIMITS
    assert sum(limits.values()) <= CLUSTER_MINUTES - ACTION_ROOM_MINUTES


GITOPS_RULES: list[Callable[[dict], None]] = [
    rule_gitops_limits,
    rule_gitops_triggers,
    rule_gitops_steps,
    rule_gitops_dump,
    rule_gitops_tools,
]


@pytest.mark.parametrize("rule", GITOPS_RULES, ids=lambda rule: rule.__name__)
def test_gitops_follows(rule):
    rule(_load(GITOPS))


VERIFY = "        run: uv run --locked python scripts/promote_images.py verify\n"
CHECK = "        run: uv run --locked python scripts/check_deploy.py"
DUMP_POD = '          "${kube[@]}" --namespace fastapi-demo '
GITOPS_EDITS = [
    ("target", rule_gitops_triggers, "  pull_request:\n", "  pull_request_target:\n"),
    ("run-trigger", rule_gitops_triggers, DISPATCH, DISPATCH + "  workflow_run:\n"),
    ("fewer-paths", rule_gitops_triggers, "      - deploy/**\n", ""),
    (
        "kubectl-checksum",
        rule_gitops_steps,
        '          echo "$KUBECTL_SHA256  $RUNNER_TEMP/bin/kubectl" | sha256sum --check --strict\n',
        "",
    ),
    (
        "kind-checksum",
        rule_gitops_steps,
        '          echo "$KIND_SHA256  $RUNNER_TEMP/bin/kind" | sha256sum --check --strict\n',
        "",
    ),
    ("verify-or-true", rule_gitops_steps, VERIFY, VERIFY[:-1] + " || true\n"),
    ("verify-ignored", rule_gitops_steps, VERIFY, VERIFY + "        continue-on-error: true\n"),
    ("verify-if", rule_gitops_steps, VERIFY, VERIFY + "        if: false\n"),
    ("verify-shell", rule_gitops_steps, VERIFY, VERIFY + "        shell: bash {0} || true\n"),
    ("no-verify", rule_gitops_steps, VERIFY, "        run: 'true'\n"),
    ("check-or-true", rule_gitops_steps, CHECK, "        run: true || python check_deploy.py"),
    ("no-poppler", rule_gitops_steps, "          command -v pdftoppm\n", ""),
    ("no-smoke-path", rule_gitops_triggers, "      - scripts/smoke_rag.py\n", ""),
    ("job-if", rule_gitops_steps, "  manifests:\n", "  manifests:\n    if: false\n"),
    ("job-ignored", rule_gitops_steps, "  cluster:\n", "  cluster:\n    continue-on-error: true\n"),
    ("shell-env", rule_gitops_steps, "\nenv:\n", "\nenv:\n  BASH_ENV: x.sh\n"),
    ("defaults", rule_gitops_steps, "\njobs:\n", "\ndefaults: {run: {shell: sh}}\njobs:\n"),
    ("other-ref", rule_gitops_steps, f"          ref: {CHECKED_OUT}\n", "          ref: main\n"),
    ("shallow", rule_gitops_steps, "          fetch-depth: 0\n", ""),
    (
        "no-smoke",
        rule_gitops_steps,
        "scripts/smoke_gitops.py --revision",
        "scripts/x.py --revision",
    ),
    (
        "extra-step",
        rule_gitops_steps,
        "      - name: Render\n",
        "      - run: ./x\n      - name: Render\n",
    ),
    (
        "dump-exec",
        rule_gitops_dump,
        f"{DUMP_POD}describe",
        f"{DUMP_POD}exec deployment/api -- env || true\n{DUMP_POD}describe",
    ),
    ("dump-secret", rule_gitops_dump, "get deployments,", "get secrets,"),
    (
        "dump-yaml",
        rule_gitops_dump,
        "networkpolicies --output wide",
        "networkpolicies --output yaml",
    ),
    ("dump-env", rule_gitops_dump, "          kube=(", "          env\n          kube=("),
    ("dump-chain", rule_gitops_dump, "describe pods || true", "describe pods || true; env"),
    ("smoke-16", rule_gitops_limits, "timeout-minutes: 15\n", "timeout-minutes: 16\n"),
    ("smoke-25", rule_gitops_limits, "timeout-minutes: 15\n", "timeout-minutes: 25\n"),
    ("smoke-40", rule_gitops_limits, "timeout-minutes: 15\n", "timeout-minutes: 40\n"),
    ("start-20", rule_gitops_limits, "timeout-minutes: 4\n", "timeout-minutes: 20\n"),
    (
        "no-smoke-limit",
        rule_gitops_limits,
        SMOKE_LIMIT,
        SMOKE_LIMIT.replace("        timeout-minutes: 15\n", ""),
    ),
    (
        "no-dump-limit",
        rule_gitops_limits,
        DUMP_LIMIT,
        DUMP_LIMIT.replace("        timeout-minutes: 2\n", ""),
    ),
    ("unlimited-step", rule_gitops_limits, KIND_STEP, "      - run: sleep 1000\n" + KIND_STEP),
    ("long-job", rule_gitops_limits, "    timeout-minutes: 30\n", "    timeout-minutes: 60\n"),
    ("no-lock-path", rule_gitops_triggers, "      - uv.lock\n", ""),
    ("no-project-path", rule_gitops_triggers, "      - pyproject.toml\n", ""),
    ("dump-fewer", rule_gitops_dump, "for name in db redis kafka", "for name in db kafka"),
    ("tool-latest", rule_gitops_tools, "/release/v1.36.4/", "/release/stable/"),
    ("tool-host", rule_gitops_tools, "https://dl.k8s.io/", "https://example.com/"),
]


@pytest.mark.parametrize(
    ("rule", "old", "new"),
    [edit[1:] for edit in GITOPS_EDITS],
    ids=[edit[0] for edit in GITOPS_EDITS],
)
def test_a_bad_gitops_gate_edit_is_rejected(rule, old, new):
    text = GITOPS.read_text()
    assert old in text
    with pytest.raises(AssertionError):
        rule(yaml.safe_load(text.replace(old, new, 1)))


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


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_checkouts_keep_no_credentials(path):
    for step in _steps(_load(path)):
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
