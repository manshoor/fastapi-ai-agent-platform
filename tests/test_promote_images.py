"""Image promotion: pins, ordering and build-record checks. No network."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "promote_images", ROOT / "scripts" / "promote_images.py"
)
assert _SPEC is not None and _SPEC.loader is not None
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

# A straight line of commits, oldest first.
OLD, MID, NEW = ("a" * 40, "b" * 40, "c" * 40)
LINE = [OLD, MID, NEW]
SIDE = "d" * 40


def digest(image: str, commit: str) -> str:
    """One fake digest per image and commit."""
    return (
        "sha256:"
        + commit[0] * 32
        + format(sorted(set(script.IMAGES.values())).index(image), "032x")
    )


def pins(app: str = MID, migrate: str = MID, postgres: str = MID, simulator: str = "") -> dict:
    commits = {
        "app": app,
        "migrate": migrate,
        "postgres": postgres,
        "simulator": simulator or app,
    }
    return {
        name: script.Pin(commit, digest(script.IMAGES[name], commit))
        for name, commit in commits.items()
    }


def record(image: str, value: str) -> dict:
    subject = {"name": image, "digest": {"sha256": value.removeprefix("sha256:")}}
    return {"verificationResult": {"statement": {"subject": [subject]}}}


HEAD = "1" * 40


@pytest.fixture
def base() -> dict:
    """The merge base with main, and the pins file there (None: not there yet)."""
    return {"commit": "2" * 40, "code": 0, "text": script.render_pins(pins(OLD, OLD, OLD))}


@pytest.fixture
def tools(monkeypatch, base) -> list[list[str]]:
    """git knows LINE and a side commit off OLD; gh knows every fake digest."""
    calls: list[list[str]] = []

    def run(command):
        calls.append(command)
        if command[:3] == ["git", "ls-tree", "--name-only"]:
            found = "" if base["text"] is None else script.PINS_PATH + "\n"
            return subprocess.CompletedProcess(command, 0, found, "")
        if command[:2] == ["git", "show"]:
            return subprocess.CompletedProcess(command, 0, base["text"], "")
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, base["code"], f"{HEAD}\n", "")
        if command == ["git", "merge-base", "HEAD", "origin/main"]:
            return subprocess.CompletedProcess(command, base["code"], f"{base['commit']}\n", "")
        if command[0] == "git":
            older, newer = command[-2:]
            if not {older, newer} <= {*LINE, SIDE}:
                return subprocess.CompletedProcess(command, 128, "", "bad object")
            ancestors = [OLD, SIDE] if newer == SIDE else LINE[: LINE.index(newer) + 1]
            return subprocess.CompletedProcess(command, 0 if older in ancestors else 1, "", "")
        image = command[3].removeprefix("oci://").partition("@")[0].partition(":")[0]
        commit = command[command.index("--source-digest") + 1]
        out = json.dumps([record(image, digest(image, commit))])
        return subprocess.CompletedProcess(command, 0, out, "")

    monkeypatch.setattr(script, "_run", run)
    return calls


# The pins file


def test_the_committed_pins_are_what_the_script_writes():
    text = script.PINS_FILE.read_text()
    assert script.render_pins(script.parse_pins(text)) == text


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("name: simulator", "name: app"),
        ("name: simulator", "name: other"),
        ("newName: ghcr.io/manshoor/", "newName: ghcr.io/other/"),
        ("newTag: sha-", "newTag: v"),
        ("digest: sha256:", "digest: sha512:"),
        ("kind: Component", "kind: Component\npatches: []"),
        ("images:", "other:"),
    ],
)
def test_other_pins_are_refused(old, new):
    text = script.render_pins(pins())
    assert old in text
    with pytest.raises(script.PromoteError):
        script.parse_pins(text.replace(old, new))


def test_a_missing_pin_is_refused():
    lines = script.render_pins(pins()).splitlines(keepends=True)
    with pytest.raises(script.PromoteError):
        script.parse_pins("".join(lines[:-4]))


# promote


def test_promote_moves_app_migrate_and_simulator(tools):
    moved = script.promote(pins(), NEW, script.PROMOTED)
    assert moved == pins(app=NEW, migrate=NEW)


def test_promote_moves_postgres_only_when_asked(tools):
    moved = script.promote(pins(), NEW, (*script.PROMOTED, "postgres"))
    assert moved == pins(app=NEW, migrate=NEW, postgres=NEW)


def test_promote_can_undo_a_rollback(tools):
    assert script.promote(pins(app=OLD), MID, script.PROMOTED) == pins()


@pytest.mark.parametrize("commit", [OLD, SIDE])
def test_promote_refuses_a_commit_not_after_migrate(tools, commit):
    with pytest.raises(script.PromoteError, match="does not descend"):
        script.promote(pins(), commit, script.PROMOTED)
    assert not [call for call in tools if call[0] == "gh"]


def test_an_unknown_commit_is_refused(tools):
    with pytest.raises(script.PromoteError, match="fetch main"):
        script.promote(pins(), "e" * 40, script.PROMOTED)


# rollback


def test_rollback_moves_app_and_simulator_only(tools):
    assert script.rollback(pins(), OLD) == pins(app=OLD)


@pytest.mark.parametrize("commit", [NEW, SIDE])
def test_rollback_refuses_a_commit_not_before_app(tools, commit):
    with pytest.raises(script.PromoteError, match="not an ancestor"):
        script.rollback(pins(), commit)


# verify


def test_verify_checks_every_pin_by_digest(tools):
    script.verify(pins(app=OLD))
    checked = {call[3] for call in tools if call[0] == "gh"}
    assert checked == {
        f"oci://{script.IMAGES[name]}@{pin.digest}" for name, pin in pins(app=OLD).items()
    }


def test_verify_refuses_an_app_newer_than_migrate(tools):
    with pytest.raises(script.PromoteError, match="newer than migrate"):
        script.verify(pins(app=NEW))


def test_verify_refuses_a_simulator_from_another_commit(tools):
    with pytest.raises(script.PromoteError, match="different commits"):
        script.verify(pins(simulator=OLD))


def test_verify_refuses_another_digest(tools):
    changed = {**pins(), "postgres": script.Pin(MID, "sha256:" + "9" * 64)}
    with pytest.raises(script.PromoteError, match="postgres is pinned to another digest"):
        script.verify(changed)


@pytest.mark.parametrize("moved", ["migrate", "postgres"])
def test_verify_refuses_moving_migrate_or_postgres_back(tools, base, moved):
    base["text"] = script.render_pins(pins())
    with pytest.raises(script.PromoteError, match=f"{moved} moved back"):
        script.verify(pins(**{"app": OLD, moved: OLD}))


def test_verify_refuses_a_hand_edit_of_every_pin(tools, base):
    base["text"] = script.render_pins(pins())
    with pytest.raises(script.PromoteError, match="moved back"):
        script.verify(pins(OLD, OLD, OLD))


def test_verify_accepts_a_rollback_and_a_promotion(tools, base):
    base["text"] = script.render_pins(pins())
    script.verify(pins(app=OLD))
    script.verify(pins(NEW, NEW, NEW))


def test_verify_accepts_the_first_pins(tools, base):
    base["text"] = None
    script.verify(pins())


def test_verify_on_main_compares_with_the_commit_before(tools, base):
    base["commit"] = HEAD
    script.verify(pins())
    assert ["git", "ls-tree", "--name-only", "HEAD^1", "--", script.PINS_PATH] in tools


def test_verify_needs_the_history(tools, base):
    base["code"] = 128
    with pytest.raises(script.PromoteError, match="fetch main"):
        script.verify(pins())


# Build records


def test_the_build_record_check_names_repo_workflow_branch_commit_and_runner(tools):
    script.verified_digest(script.IMAGE, f":sha-{MID}", MID)
    ((*command,),) = tools
    assert command == [
        "gh",
        "attestation",
        "verify",
        f"oci://ghcr.io/manshoor/fastapi-ai-agent-platform:sha-{MID}",
        "--repo",
        "manshoor/fastapi-ai-agent-platform",
        "--signer-workflow",
        "manshoor/fastapi-ai-agent-platform/.github/workflows/ci.yml",
        "--source-ref",
        "refs/heads/main",
        "--source-digest",
        MID,
        "--deny-self-hosted-runners",
        "--format",
        "json",
    ]


def test_a_failed_check_is_refused(monkeypatch):
    failed = subprocess.CompletedProcess([], 1, "", "no attestations")
    monkeypatch.setattr(script, "_run", lambda _: failed)
    with pytest.raises(script.PromoteError, match="no attestations"):
        script.verified_digest(script.IMAGE, f":sha-{MID}", MID)


def test_a_record_for_another_image_is_refused():
    # Same commit, same workflow: only the subject name tells the images apart.
    records = [record(script.IMAGES["postgres"], digest(script.IMAGE, MID))]
    with pytest.raises(script.PromoteError, match="exactly one"):
        script.record_digest(records, script.IMAGE)


def test_two_digests_for_one_image_are_refused():
    records = [record(script.IMAGE, digest(script.IMAGE, commit)) for commit in (OLD, MID)]
    with pytest.raises(script.PromoteError, match="exactly one"):
        script.record_digest(records, script.IMAGE)


def test_a_repeated_record_is_one_digest():
    records = [record(script.IMAGE, digest(script.IMAGE, MID))] * 2
    assert script.record_digest(records, script.IMAGE) == digest(script.IMAGE, MID)
