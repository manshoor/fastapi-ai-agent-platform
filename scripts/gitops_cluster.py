"""Create or delete the throwaway GitOps demo cluster (kind + Argo CD).

    python scripts/gitops_cluster.py up [--revision <40-hex commit>]
    python scripts/gitops_cluster.py down

    1. create the kind cluster, then check that its API server is this machine
    2. namespace `fastapi-demo` (Pod Security `restricted`) and its Secrets
    3. Argo CD core, from one exact manifest, its images pulled into the node by digest
    4. the Argo CD project and Application: Argo CD then syncs deploy/app

Only ever touches the cluster `fastapi-gitops` on this machine's Docker, always by its own
context in a private kubeconfig: your own kubeconfig is neither read nor changed. Secret
values go over stdin, are never printed, and existing Secrets are never replaced.
"""

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import stat
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = "fastapi-gitops"
CONTEXT = f"kind-{CLUSTER}"
KUBECONFIG = ROOT / ".gitops-local" / "kubeconfig"
PRIVATE_MODE = 0o700
# kind's other providers have their own endpoints.
PROVIDER = "docker"
LOCAL_DOCKER = "unix:///"
NAMESPACE = "fastapi-demo"
ARGOCD_NAMESPACE = "argocd"
KIND_CONFIG = ROOT / "deploy" / "kind" / "cluster.yaml"
PROJECT_FILE = ROOT / "deploy" / "argocd" / "project.yaml"
APPLICATION_FILE = ROOT / "deploy" / "argocd" / "application.yaml"
# v3.6.1, the newest release at least 14 days old when pinned.
ARGOCD_URL = "https://raw.githubusercontent.com/argoproj/argo-cd/v3.6.1/manifests/core-install.yaml"
ARGOCD_SHA256 = "0b38189d6e1fd76ef5c3db10355026c48a02020d82bf21a1a1b5ab16dffe4f3c"
ARGOCD_MAX_BYTES = 8 * 1024 * 1024
ARGOCD_WORKLOADS = (
    "deployment/argocd-redis",
    "deployment/argocd-repo-server",
    "deployment/argocd-applicationset-controller",
    "statefulset/argocd-application-controller",
)


class PinnedImage(NamedTuple):
    digest: str
    # Registries that serve this digest, the manifest's own first.
    sources: tuple[str, ...]


# The images the manifest names, by tag, and their index digests. `up` pulls each into the
# node with a short retry and a second source, then runs it by digest with pull policy
# Never: a failed pull ends `up` with a reason, not in the kubelet's growing back-off.
ARGOCD_IMAGES = {
    # v3.6.1
    "quay.io/argoproj/argocd:v3.6.1": PinnedImage(
        "sha256:296e3d74f449b2ce45265613fcb68c18ed7366488360c8ea32f5f2a93d2e5c54",
        ("quay.io/argoproj/argocd",),
    ),
    # 8.10.1-alpine; Docker Hub serves the same index.
    "public.ecr.aws/docker/library/redis:8.10.1-alpine": PinnedImage(
        "sha256:ef216d85ec44f4fdf7c0e6df2d1757a4734ed32fbb278cabd3181b5a27353b79",
        ("public.ecr.aws/docker/library/redis", "docker.io/library/redis"),
    ),
}
NODE = f"{CLUSTER}-control-plane"
IMAGE_LINE = re.compile(r"(?P<indent> +)(?P<dash>- )?image: (?P<image>\S+)")
POLICY_LINE = re.compile(r" +imagePullPolicy: \w+")
PULL_ATTEMPTS = 2
PULL_SECONDS = 15
PULL_WAIT_SECONDS = 5
COMMIT = re.compile(r"[0-9a-f]{40}")
TIMEOUT_SECONDS = 120
CREATE_TIMEOUT_SECONDS = 600
ROLLOUT_TIMEOUT_SECONDS = 300
KEY_BYTES = 32


class ClusterError(RuntimeError):
    """A fixed reason, or a tool's own error text; never a Secret value."""


def _hex_key() -> str:
    return secrets.token_hex(KEY_BYTES)


def _base64_key() -> str:
    return base64.b64encode(secrets.token_bytes(KEY_BYTES)).decode("ascii")


# All keys from the start: a key cannot be added later, existing Secrets are never changed.
SECRETS: Mapping[str, Mapping[str, Callable[[], str]]] = {
    "db-credentials": {"password": _hex_key},
    "api-secrets": {
        "jwt-secret": _hex_key,
        "totp-key": _base64_key,
        "memory-key": _base64_key,
    },
}


def run(command: list[str], stdin: str = "", timeout: int = TIMEOUT_SECONDS) -> str:
    try:
        result = subprocess.run(
            command, input=stdin, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise ClusterError(f"{command[0]} timed out") from None
    except OSError:
        raise ClusterError(f"{command[0]} is not available") from None
    if result.returncode:
        raise ClusterError(f"{command[0]} failed: {result.stderr.strip()}")
    return result.stdout


def kubectl_command(*arguments: str) -> list[str]:
    return ["kubectl", "--kubeconfig", str(KUBECONFIG), "--context", CONTEXT, *arguments]


def kubectl(*arguments: str, stdin: str = "", timeout: int = TIMEOUT_SECONDS) -> str:
    return run(kubectl_command(*arguments), stdin, timeout)


def require_local_docker() -> None:
    """Refuse unless kind would use the Docker socket of this machine."""
    if os.environ.get("KIND_EXPERIMENTAL_PROVIDER", PROVIDER) not in ("", PROVIDER):
        raise ClusterError("the kind provider is not docker")
    # DOCKER_HOST beats the context. A TCP address, even loopback, may be a tunnel.
    host = os.environ.get("DOCKER_HOST") or run(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]
    )
    if not host.strip().startswith(LOCAL_DOCKER):
        raise ClusterError("the Docker endpoint is not a local socket")


def check_server(config: Mapping) -> None:
    """Refuse unless the context's one API server is on this machine."""
    clusters = config.get("clusters") or []
    if len(clusters) != 1:
        raise ClusterError(f"expected one cluster behind context {CONTEXT}")
    cluster = clusters[0].get("cluster") or {}
    # A proxy would send a loopback address somewhere else.
    if cluster.get("proxy-url"):
        raise ClusterError(f"context {CONTEXT} uses a proxy")
    try:
        loopback = ipaddress.ip_address(urlsplit(cluster["server"]).hostname).is_loopback
    except (KeyError, ValueError):
        loopback = False
    if not loopback:
        raise ClusterError(f"context {CONTEXT} is not a loopback server")


def require_loopback() -> None:
    check_server(json.loads(kubectl("config", "view", "--minify", "--output", "json")))


def _private_folder() -> None:
    """Create the kubeconfig's folder, or refuse one that others could change."""
    folder = KUBECONFIG.parent
    try:
        folder.mkdir(mode=PRIVATE_MODE)
    except FileExistsError:
        pass
    # lstat: a link is not followed.
    status = folder.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or stat.S_IMODE(status.st_mode) != PRIVATE_MODE
        or status.st_uid != os.getuid()
    ):
        raise ClusterError(f"{folder.name} is not a private folder of this user (mode 0700)")


def _create_cluster() -> None:
    _private_folder()
    target = ["--name", CLUSTER, "--kubeconfig", str(KUBECONFIG)]
    if CLUSTER in run(["kind", "get", "clusters"]).split():
        run(["kind", "export", "kubeconfig", *target])
        return
    command = ["kind", "create", "cluster", *target, "--config", str(KIND_CONFIG)]
    run([*command, "--wait", "180s"], timeout=CREATE_TIMEOUT_SECONDS)


def _apply(manifest: str, *arguments: str) -> None:
    # Server-side: Argo CD's CRDs are too large for the client-side annotation.
    kubectl("apply", "--server-side", "--force-conflicts", *arguments, "-f", "-", stdin=manifest)


def namespace_manifests() -> list[dict]:
    restricted = {
        f"pod-security.kubernetes.io/{mode}": "restricted" for mode in ("enforce", "warn", "audit")
    }
    return [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ARGOCD_NAMESPACE}},
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": NAMESPACE, "labels": restricted},
        },
    ]


def secret_manifest(name: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "type": "Opaque",
        "stringData": {key: make() for key, make in SECRETS[name].items()},
    }


def _existing_keys(name: str) -> set[str] | None:
    """Key names of a Secret, or None if it is missing. The values stay in the cluster."""
    template = r'{{range $key, $value := .data}}{{$key}}{{"\n"}}{{end}}'
    common = ("--namespace", NAMESPACE, "get", "secret", name, "--ignore-not-found")
    if not kubectl(*common, "--output", "name").strip():
        return None
    return set(kubectl(*common, "--output", f"go-template={template}").split())


def ensure_secret(name: str) -> None:
    existing = _existing_keys(name)
    if existing is None:
        try:
            # `create` fails on an existing Secret, so it can never replace one.
            kubectl("create", "-f", "-", stdin=json.dumps(secret_manifest(name)))
        except ClusterError:
            # kubectl may quote its input in an error.
            raise ClusterError(f"Secret {name} could not be created") from None
        return
    missing = sorted(set(SECRETS[name]) - existing)
    if missing:
        raise ClusterError(f"Secret {name} lacks {', '.join(missing)}; it is never changed")


def check_manifest(data: bytes) -> str:
    if hashlib.sha256(data).hexdigest() != ARGOCD_SHA256:
        raise ClusterError("the Argo CD manifest does not match its SHA-256")
    return data.decode()


def _download_manifest() -> str:
    try:
        with urllib.request.urlopen(ARGOCD_URL, timeout=TIMEOUT_SECONDS) as response:
            return check_manifest(response.read(ARGOCD_MAX_BYTES))
    except (urllib.error.URLError, TimeoutError):
        raise ClusterError("the Argo CD manifest could not be downloaded") from None


def _pull(image: PinnedImage) -> str:
    """Pull one image into the node by digest; the reference that worked."""
    for source in image.sources:
        reference = f"{source}@{image.digest}"
        for attempt in range(PULL_ATTEMPTS):
            if attempt:
                time.sleep(PULL_WAIT_SECONDS)
            command = ["docker", "exec", NODE, "crictl", "pull"]
            command += ["--pull-timeout", f"{PULL_SECONDS}s", reference]
            try:
                run(command, timeout=PULL_SECONDS + PULL_WAIT_SECONDS)
                return reference
            except ClusterError as error:
                print(f"pull failed: {error}", flush=True)
    raise ClusterError(f"{image.sources[0]}@{image.digest} could not be pulled")


def pin_images(manifest: str, pulled: Mapping[str, str]) -> str:
    """Point every container at its pulled digest, with pull policy Never."""
    lines = []
    found = set()
    for line in manifest.splitlines():
        if POLICY_LINE.fullmatch(line):
            continue
        match = IMAGE_LINE.fullmatch(line)
        if not match:
            lines.append(line)
            continue
        found.add(match["image"])
        indent = match["indent"] + ("  " if match["dash"] else "")
        lines.append(f"{match['indent']}{match['dash'] or ''}image: {pulled.get(match['image'])}")
        lines.append(f"{indent}imagePullPolicy: Never")
    if found != set(pulled):
        raise ClusterError("the Argo CD manifest names other Argo CD images than the pins")
    return "\n".join(lines) + "\n"


def _install_argocd() -> None:
    manifest = _download_manifest()
    pulled = {tag: _pull(image) for tag, image in ARGOCD_IMAGES.items()}
    _apply(pin_images(manifest, pulled), "--namespace", ARGOCD_NAMESPACE)
    for workload in ARGOCD_WORKLOADS:
        kubectl(
            *("--namespace", ARGOCD_NAMESPACE, "rollout", "status", workload),
            f"--timeout={ROLLOUT_TIMEOUT_SECONDS}s",
            timeout=ROLLOUT_TIMEOUT_SECONDS + TIMEOUT_SECONDS,
        )


def application(revision: str | None) -> dict:
    """The committed Application, pointed at one commit if given."""
    manifest = yaml.safe_load(APPLICATION_FILE.read_text())
    if revision is None:
        return manifest
    if not COMMIT.fullmatch(revision):
        raise ClusterError("revision must be 40 lowercase hex characters")
    manifest["spec"]["source"]["targetRevision"] = revision
    return manifest


def up(revision: str | None) -> None:
    # Read first, so a bad revision creates nothing.
    target = application(revision)
    require_local_docker()
    _create_cluster()
    require_loopback()

    for namespace in namespace_manifests():
        _apply(json.dumps(namespace))
    for name in SECRETS:
        ensure_secret(name)

    _install_argocd()
    _apply(PROJECT_FILE.read_text())
    _apply(json.dumps(target))


def down() -> None:
    require_local_docker()
    _private_folder()
    command = ["kind", "delete", "cluster", "--name", CLUSTER, "--kubeconfig", str(KUBECONFIG)]
    run(command, timeout=CREATE_TIMEOUT_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    up_parser = commands.add_parser("up", help="Create the cluster and start the sync")
    up_parser.add_argument("--revision", help="Sync this commit instead of main")
    commands.add_parser("down", help="Delete the cluster and its data")
    args = parser.parse_args()
    try:
        if args.command == "up":
            up(args.revision)
        else:
            down()
    except ClusterError as error:
        parser.exit(1, f"{error}\n")
    print(f"{args.command}: done ({CLUSTER})")


if __name__ == "__main__":
    main()
