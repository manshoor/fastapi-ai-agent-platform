"""Pin published images in deploy/app/images/ after checking their signed build records.

    python scripts/promote_images.py promote <commit> [--postgres]
    python scripts/promote_images.py rollback <commit>
    python scripts/promote_images.py verify

promote   moves app, migrate and simulator (and postgres if asked) to a commit that
          descends from the pinned migrate commit.
rollback  moves app and simulator back. Never migrate or postgres: an older image cannot
          run `alembic upgrade` on a newer schema, so migrations must stay usable by the
          previous app version.
verify    re-checks every pin, that app is not newer than migrate, and that migrate and
          postgres are not older than where this branch left main.

Needs `git` and a signed-in `gh`. Every digest comes from a verified build record.
"""

import argparse
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = "deploy/app/images/kustomization.yaml"
PINS_FILE = ROOT / PINS_PATH
REPOSITORY = "manshoor/fastapi-ai-agent-platform"
WORKFLOW = f"{REPOSITORY}/.github/workflows/ci.yml"
SOURCE_REF = "refs/heads/main"
IMAGE = f"ghcr.io/{REPOSITORY}"
# Kustomize placeholder -> published image.
IMAGES = {
    "app": IMAGE,
    "migrate": IMAGE,
    "postgres": f"{IMAGE}-postgres",
    "simulator": f"{IMAGE}-simulator",
}
PROMOTED = ("app", "migrate", "simulator")
ROLLED_BACK = ("app", "simulator")
NEVER_BACK = ("migrate", "postgres")
MAIN = "origin/main"
COMMIT = re.compile(r"[0-9a-f]{40}")
TAG = re.compile(r"sha-([0-9a-f]{40})")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
# `git merge-base --is-ancestor` exit codes.
ANCESTOR = 0
NOT_ANCESTOR = 1
TIMEOUT_SECONDS = 120
HEADER = """\
# Written by scripts/promote_images.py. Do not edit by hand.
# One name per pin, so `migrate` can stay ahead of `app` after a rollback.
apiVersion: kustomize.config.k8s.io/v1alpha1
kind: Component
images:
"""


class PromoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pin:
    commit: str
    digest: str


Pins = dict[str, Pin]


# The pins file


def parse_pins(text: str) -> Pins:
    try:
        return _parse_pins(text)
    except (yaml.YAMLError, AttributeError, KeyError, TypeError):
        raise PromoteError("the pins file is malformed") from None


def _parse_pins(text: str) -> Pins:
    document = yaml.safe_load(text)
    if not isinstance(document, dict) or set(document) != {"apiVersion", "kind", "images"}:
        raise PromoteError("the pins file is not the one this script writes")
    pins: Pins = {}
    for entry in document["images"]:
        if set(entry) != {"name", "newName", "newTag", "digest"}:
            raise PromoteError("a pin needs exactly name, newName, newTag and digest")
        name = entry["name"]
        tag = TAG.fullmatch(entry["newTag"])
        if name in pins or IMAGES.get(name) != entry["newName"]:
            raise PromoteError(f"unexpected image pin: {name}")
        if not tag or not DIGEST.fullmatch(entry["digest"]):
            raise PromoteError(f"{name} is not pinned by commit and digest")
        pins[name] = Pin(tag[1], entry["digest"])
    if set(pins) != set(IMAGES):
        raise PromoteError("the pins file must pin exactly " + ", ".join(IMAGES))
    return pins


def render_pins(pins: Mapping[str, Pin]) -> str:
    entries = [
        f"  - name: {name}\n    newName: {image}\n"
        f"    newTag: sha-{pins[name].commit}\n    digest: {pins[name].digest}\n"
        for name, image in IMAGES.items()
    ]
    return HEADER + "".join(entries)


# git and gh


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise PromoteError(f"{command[0]} timed out") from None
    except OSError:
        raise PromoteError(f"{command[0]} is not available") from None


def is_ancestor(older: str, newer: str) -> bool:
    """True if `older` is `newer` or an ancestor of it."""
    result = _run(["git", "merge-base", "--is-ancestor", older, newer])
    if result.returncode not in (ANCESTOR, NOT_ANCESTOR):
        raise PromoteError("git does not know one of the commits; fetch main first")
    return result.returncode == ANCESTOR


def _git(*arguments: str) -> str:
    result = _run(["git", *arguments])
    if result.returncode:
        raise PromoteError(f"git {arguments[0]} failed; fetch main with its history first")
    return result.stdout


def _base_pins() -> Pins | None:
    """The pins where this branch left main, or None if main had none yet."""
    base = _git("merge-base", "HEAD", MAIN).strip()
    # On main itself: compare with the commit before.
    if base == _git("rev-parse", "HEAD").strip():
        base = "HEAD^1"
    if not _git("ls-tree", "--name-only", base, "--", PINS_PATH).strip():
        return None
    return parse_pins(_git("show", f"{base}:{PINS_PATH}"))


def record_digest(records: object, image: str) -> str:
    """The one digest the verified records give this image."""
    # gh matches a record by digest alone, so the subject name is checked here.
    digests = {
        subject["digest"]["sha256"]
        for record in records
        for subject in record["verificationResult"]["statement"]["subject"]
        if subject["name"] == image
    }
    if len(digests) != 1:
        raise PromoteError(f"the build record does not name exactly one {image}")
    digest = f"sha256:{digests.pop()}"
    if not DIGEST.fullmatch(digest):
        raise PromoteError(f"the build record of {image} holds no digest")
    return digest


def verified_digest(image: str, reference: str, commit: str) -> str:
    """Check the build record of `<image><reference>` and return the signed digest."""
    result = _run(
        [
            *("gh", "attestation", "verify", f"oci://{image}{reference}"),
            *("--repo", REPOSITORY, "--signer-workflow", WORKFLOW),
            *("--source-ref", SOURCE_REF, "--source-digest", commit),
            *("--deny-self-hosted-runners", "--format", "json"),
        ]
    )
    if result.returncode:
        raise PromoteError(f"no valid build record for {image}{reference}: {result.stderr.strip()}")
    return record_digest(json.loads(result.stdout), image)


# Commands


def _pin(pins: Pins, names: tuple[str, ...], commit: str) -> Pins:
    moved = {
        name: Pin(commit, verified_digest(IMAGES[name], f":sha-{commit}", commit)) for name in names
    }
    return {**pins, **moved}


def promote(pins: Pins, commit: str, names: tuple[str, ...]) -> Pins:
    if not is_ancestor(pins["migrate"].commit, commit):
        raise PromoteError("the commit does not descend from the pinned migrate commit")
    return _pin(pins, names, commit)


def rollback(pins: Pins, commit: str) -> Pins:
    if not is_ancestor(commit, pins["app"].commit):
        raise PromoteError("the commit is not an ancestor of the pinned app commit")
    if not is_ancestor(commit, pins["migrate"].commit):
        raise PromoteError("the commit is not an ancestor of the pinned migrate commit")
    return _pin(pins, ROLLED_BACK, commit)


def verify(pins: Pins) -> None:
    if pins["app"].commit != pins["simulator"].commit:
        raise PromoteError("app and simulator are pinned to different commits")
    if not is_ancestor(pins["app"].commit, pins["migrate"].commit):
        raise PromoteError("app is newer than migrate")
    base = _base_pins()
    for name in NEVER_BACK if base else ():
        if not is_ancestor(base[name].commit, pins[name].commit):
            raise PromoteError(f"{name} moved back")
    for name, pin in pins.items():
        # By digest: a tag could move.
        if verified_digest(IMAGES[name], f"@{pin.digest}", pin.commit) != pin.digest:
            raise PromoteError(f"{name} is pinned to another digest than its build record")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    promoter = commands.add_parser("promote")
    promoter.add_argument("commit")
    promoter.add_argument("--postgres", action="store_true", help="Also move the postgres pin")
    commands.add_parser("rollback").add_argument("commit")
    commands.add_parser("verify")
    args = parser.parse_args()
    if args.command != "verify" and not COMMIT.fullmatch(args.commit):
        parser.error("commit must be 40 lowercase hex characters")
    try:
        pins = parse_pins(PINS_FILE.read_text())
        if args.command == "verify":
            verify(pins)
        elif args.command == "rollback":
            PINS_FILE.write_text(render_pins(rollback(pins, args.commit)))
        else:
            names = (*PROMOTED, "postgres") if args.postgres else PROMOTED
            PINS_FILE.write_text(render_pins(promote(pins, args.commit, names)))
    except PromoteError as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.command}: done")


if __name__ == "__main__":
    main()
